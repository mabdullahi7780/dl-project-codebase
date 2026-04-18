"""Joint MoE training script (Phase 2).

Trains the routing gate (C3), expert bank (C5), and fusion module (C6)
end-to-end on cached image embeddings + ground-truth / pseudo-ground-truth
lesion masks.

The Component 1 encoder is frozen during this phase. Only C3/C5/C6 update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from src.components.component3_routing import (
    Component3RoutingGate,
    RoutingGateConfig,
    routing_load_balance_loss,
)
from src.components.component5_experts import (
    EXPERT_NAMES,
    ExpertBank,
    ExpertDecoderConfig,
)
from src.components.component6_fusion import (
    Component6ExpertFusion,
    FusionConfig,
)
from src.core.device import pick_device
from src.core.seed import seed_everything


def bce_dice_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    bce_weight: float = 0.5,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted BCE + Dice over a batch of binary masks."""

    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    bce = bce.mean(dim=(1, 2, 3))

    pred_prob = torch.sigmoid(pred_logits)
    intersection = (pred_prob * target).sum(dim=(2, 3))
    union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
    dice = dice.mean(dim=1)

    loss_per_sample = (bce_weight * bce) + ((1.0 - bce_weight) * dice)
    if sample_weight is None:
        return loss_per_sample.mean()

    weights = sample_weight.float().view(-1)
    return (loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


class JointMoEDataset(Dataset):
    """Loads the grounded MoE cache.

    Synthetic fallback remains available only when ``cache_dir`` is omitted,
    which keeps smoke tests cheap while preventing silent regressions in real
    training runs.
    """

    def __init__(self, cache_dir: Path | None = None, num_synthetic: int = 200) -> None:
        if cache_dir is not None:
            if not cache_dir.exists():
                raise FileNotFoundError(f"MoE cache directory not found: {cache_dir}")
            self.samples = sorted(cache_dir.glob("*.pt"))
            if not self.samples:
                raise FileNotFoundError(f"MoE cache directory is empty: {cache_dir}")
        else:
            self.samples = list(range(num_synthetic))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        if isinstance(sample, Path):
            data = torch.load(sample, weights_only=False)
            expert_masks = data.get("expert_masks")
            if not isinstance(expert_masks, dict):
                raise KeyError(f"Cache sample {sample} is missing expert_masks.")
            expert_weights = data.get("expert_supervision_weights", {})
            return {
                "image_emb": data["image_emb"].float(),
                "domain_ctx": data.get("domain_ctx", torch.zeros(256)).float(),
                "lesion_mask": data["lesion_mask"].float(),
                "sample_weight": torch.tensor(float(data.get("supervision_weight", 1.0)), dtype=torch.float32),
                "expert_masks": torch.stack([expert_masks[name].float() for name in EXPERT_NAMES], dim=0),
                "expert_weights": torch.tensor(
                    [float(expert_weights.get(name, data.get("supervision_weight", 1.0))) for name in EXPERT_NAMES],
                    dtype=torch.float32,
                ),
            }

        emb = torch.randn(256, 64, 64)
        mask = torch.zeros(1, 256, 256)
        cx, cy = torch.randint(64, 192, (2,)).tolist()
        rx, ry = torch.randint(10, 40, (2,)).tolist()
        yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
        blob = ((xx - cx).float() / rx) ** 2 + ((yy - cy).float() / ry) ** 2
        mask[0] = (blob < 1.0).float()
        return {
            "image_emb": emb,
            "domain_ctx": torch.randn(256),
            "lesion_mask": mask,
            "sample_weight": torch.tensor(1.0, dtype=torch.float32),
            "expert_masks": torch.stack([mask.clone() for _ in EXPERT_NAMES], dim=0),
            "expert_weights": torch.ones(len(EXPERT_NAMES), dtype=torch.float32),
        }


def train_joint(config: dict[str, Any], *, cache_dir: Path | None = None) -> Path:
    moe_cfg = config.get("moe", {})
    train_cfg = config.get("moe_training", {}).get("joint", {})

    epochs = int(train_cfg.get("epochs", 15))
    batch_size = int(train_cfg.get("batch_size", 4))
    lr_gate = float(train_cfg.get("lr_gate", 5e-4))
    lr_experts = float(train_cfg.get("lr_experts", 1e-4))
    lr_fusion = float(train_cfg.get("lr_fusion", 1e-4))
    lb_weight = float(train_cfg.get("load_balance_weight", 0.1))
    expert_aux_weight = float(train_cfg.get("expert_aux_weight", 0.25))
    save_dir = Path(config.get("moe_training", {}).get("save_dir", "outputs/moe_runs"))
    save_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(config.get("moe_training", {}).get("device"))
    seed_everything(1337)

    num_experts = int(moe_cfg.get("num_experts", 4))

    routing_gate = Component3RoutingGate(
        RoutingGateConfig(
            num_experts=num_experts,
            use_domain_ctx=bool(moe_cfg.get("use_domain_ctx", False)),
            domain_ctx_dim=int(moe_cfg.get("domain_ctx_dim", 256)),
            temperature=float(moe_cfg.get("routing_temperature", 1.0)),
            top_k=moe_cfg.get("routing_top_k"),
        )
    ).to(device)

    expert_bank = ExpertBank(
        ExpertDecoderConfig(
            num_experts=num_experts,
            mid_channels=int(moe_cfg.get("expert_mid_channels", 128)),
            dropout=float(moe_cfg.get("expert_dropout", 0.1)),
        )
    ).to(device)

    fusion = Component6ExpertFusion(
        FusionConfig(
            num_experts=num_experts,
            fusion_mode=str(moe_cfg.get("fusion_mode", "weighted_logit")),
            learnable_bias=bool(moe_cfg.get("learnable_fusion_bias", False)),
        )
    ).to(device)

    for i, name in enumerate(EXPERT_NAMES[:num_experts]):
        pretrained = save_dir / f"expert_{name}_best.pt"
        if pretrained.is_file():
            expert_bank.experts[i].load_state_dict(
                torch.load(pretrained, map_location=device, weights_only=False)
            )
            print(f"  Loaded pretrained expert: {name} from {pretrained}")

    optimizer = torch.optim.AdamW(
        [
            {"params": routing_gate.parameters(), "lr": lr_gate},
            {"params": expert_bank.parameters(), "lr": lr_experts},
            {"params": fusion.parameters(), "lr": lr_fusion},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        JointMoEDataset(cache_dir=cache_dir),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config.get("moe_training", {}).get("num_workers", 2)),
        pin_memory=True,
        drop_last=False,
    )

    use_amp = bool(config.get("moe_training", {}).get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    print(f"[Joint MoE] {epochs} epochs, bs={batch_size}")
    print(
        f"  LR: gate={lr_gate}, experts={lr_experts}, fusion={lr_fusion}, "
        f"expert_aux={expert_aux_weight}, use_domain_ctx={routing_gate.use_domain_ctx}"
    )

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    save_every = int(config.get("moe_training", {}).get("save_every", 2))

    for epoch in range(epochs):
        routing_gate.train()
        expert_bank.train()
        fusion.train()

        running = {"mask_loss": 0.0, "expert_loss": 0.0, "lb_loss": 0.0, "total": 0.0}
        n_batches = 0

        for batch in loader:
            emb = batch["image_emb"].to(device)
            domain_ctx = batch["domain_ctx"].to(device)
            target = batch["lesion_mask"].to(device)
            sample_weight = batch["sample_weight"].to(device)
            expert_target = batch["expert_masks"].to(device)[:, :num_experts]
            expert_weight = batch["expert_weights"].to(device)[:, :num_experts]

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                routing_weights = routing_gate(
                    emb,
                    domain_ctx if routing_gate.use_domain_ctx else None,
                )
                expert_logits = expert_bank(emb)
                fusion_out = fusion(expert_logits, routing_weights)

                mask_loss = bce_dice_loss(
                    fusion_out.mask_fused_logits,
                    target,
                    sample_weight=sample_weight,
                )

                expert_loss_terms = [
                    bce_dice_loss(
                        expert_logits[i],
                        expert_target[:, i],
                        sample_weight=expert_weight[:, i],
                    )
                    for i in range(num_experts)
                ]
                expert_loss = torch.stack(expert_loss_terms).mean()

                lb_loss = routing_load_balance_loss(routing_weights, num_experts=num_experts)
                total_loss = mask_loss + (expert_aux_weight * expert_loss) + (lb_weight * lb_loss)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running["mask_loss"] += mask_loss.item()
            running["expert_loss"] += expert_loss.item()
            running["lb_loss"] += lb_loss.item()
            running["total"] += total_loss.item()
            n_batches += 1

        scheduler.step()

        avg = {key: value / max(n_batches, 1) for key, value in running.items()}
        record = {"epoch": epoch, **avg}
        history.append(record)
        print(
            f"  epoch {epoch}: total={avg['total']:.4f}  mask={avg['mask_loss']:.4f}  "
            f"expert={avg['expert_loss']:.4f}  lb={avg['lb_loss']:.4f}"
        )

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            _save_moe_checkpoint(save_dir / "moe_best.pt", routing_gate, expert_bank, fusion, epoch)

        if (epoch + 1) % save_every == 0:
            _save_moe_checkpoint(save_dir / f"moe_epoch{epoch}.pt", routing_gate, expert_bank, fusion, epoch)

    final_path = _save_moe_checkpoint(
        save_dir / "moe_checkpoint.pt",
        routing_gate,
        expert_bank,
        fusion,
        epochs - 1,
    )

    hist_path = save_dir / "moe_joint_history.jsonl"
    with hist_path.open("w", encoding="utf-8") as handle:
        for rec in history:
            handle.write(json.dumps(rec) + "\n")

    print(f"MoE joint training complete. Checkpoint: {final_path}")
    return final_path


def _save_moe_checkpoint(
    path: Path,
    routing_gate: nn.Module,
    expert_bank: nn.Module,
    fusion: nn.Module,
    epoch: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "routing_gate": routing_gate.state_dict(),
            "expert_bank": expert_bank.state_dict(),
            "fusion": fusion.state_dict(),
            "epoch": epoch,
        },
        path,
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint MoE training (Phase 2).")
    parser.add_argument("--config", default="configs/moe.yaml")
    parser.add_argument("--cache-dir", default=None, help="Cached embedding directory.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if args.epochs is not None:
        config.setdefault("moe_training", {}).setdefault("joint", {})["epochs"] = args.epochs

    if args.dry_run:
        print("[Dry run] Joint MoE training would proceed with config:")
        print(json.dumps(config.get("moe_training", {}).get("joint", {}), indent=2))
        return

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    train_joint(config, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
