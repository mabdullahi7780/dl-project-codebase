"""Joint MoE training script (Phase 2).

Trains the routing gate (C3), expert bank (C5), and fusion module (C6)
end-to-end on cached image embeddings + ground-truth lesion masks.

The Component 1 encoder is **frozen** during this phase — only C3/C5/C6
weights are updated.

Usage::

    python -m src.training.train_moe_joint \
        --config configs/moe.yaml \
        --cache-dir outputs/embedding_cache

    # Smoke test:
    python -m src.training.train_moe_joint --config configs/moe.yaml --epochs 1 --dry-run
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
    ExpertBank,
    ExpertDecoderConfig,
    EXPERT_NAMES,
)
from src.components.component6_fusion import (
    Component6ExpertFusion,
    FusionConfig,
)
from src.core.device import pick_device
from src.core.seed import seed_everything


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def bce_dice_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    pred_prob = torch.sigmoid(pred_logits)
    intersection = (pred_prob * target).sum(dim=(2, 3))
    union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
    return bce_weight * bce + (1.0 - bce_weight) * dice.mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JointMoEDataset(Dataset):
    """Loads cached image embeddings + ground-truth lesion masks.

    Each sample is a ``.pt`` file containing:
        - ``image_emb``: [256, 64, 64] — Component 1 output
        - ``lesion_mask``: [1, 256, 256] — combined lesion GT
        - ``expert_masks``: dict[str, Tensor] — per-pathology GT (optional)
        - ``lung_mask``: [1, 256, 256] — lung mask (optional)

    Falls back to synthetic data for smoke tests.
    """

    def __init__(self, cache_dir: Path | None = None, num_synthetic: int = 200) -> None:
        self.cache_dir = cache_dir
        if cache_dir is not None and cache_dir.exists():
            self.samples = sorted(cache_dir.glob("*.pt"))
        else:
            self.samples = list(range(num_synthetic))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if isinstance(self.samples[idx], Path):
            return torch.load(self.samples[idx], weights_only=False)

        # Synthetic fallback
        emb = torch.randn(256, 64, 64)
        mask = torch.zeros(1, 256, 256)
        cx, cy = torch.randint(64, 192, (2,)).tolist()
        rx, ry = torch.randint(10, 40, (2,)).tolist()
        yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
        blob = ((xx - cx).float() / rx) ** 2 + ((yy - cy).float() / ry) ** 2
        mask[0] = (blob < 1.0).float()
        return {"image_emb": emb, "lesion_mask": mask}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_joint(config: dict[str, Any], *, cache_dir: Path | None = None) -> Path:
    moe_cfg = config.get("moe", {})
    train_cfg = config.get("moe_training", {}).get("joint", {})

    epochs = int(train_cfg.get("epochs", 15))
    batch_size = int(train_cfg.get("batch_size", 4))
    lr_gate = float(train_cfg.get("lr_gate", 5e-4))
    lr_experts = float(train_cfg.get("lr_experts", 1e-4))
    lr_fusion = float(train_cfg.get("lr_fusion", 1e-4))
    lb_weight = float(train_cfg.get("load_balance_weight", 0.1))
    save_dir = Path(config.get("moe_training", {}).get("save_dir", "outputs/moe_runs"))
    save_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(config.get("moe_training", {}).get("device"))
    seed_everything(1337)

    num_experts = int(moe_cfg.get("num_experts", 4))

    # Build models
    routing_gate = Component3RoutingGate(
        RoutingGateConfig(
            num_experts=num_experts,
            temperature=float(moe_cfg.get("routing_temperature", 1.0)),
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

    # Load pretrained expert weights if available
    for i, name in enumerate(EXPERT_NAMES[:num_experts]):
        pretrained = save_dir / f"expert_{name}_best.pt"
        if pretrained.is_file():
            expert_bank.experts[i].load_state_dict(
                torch.load(pretrained, map_location=device, weights_only=False)
            )
            print(f"  Loaded pretrained expert: {name} from {pretrained}")

    # Optimiser with per-module LR
    optimizer = torch.optim.AdamW(
        [
            {"params": routing_gate.parameters(), "lr": lr_gate},
            {"params": expert_bank.parameters(), "lr": lr_experts},
            {"params": fusion.parameters(), "lr": lr_fusion},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = JointMoEDataset(cache_dir=cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config.get("moe_training", {}).get("num_workers", 2)),
        pin_memory=True,
        drop_last=True,
    )

    use_amp = bool(config.get("moe_training", {}).get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    print(f"[Joint MoE] {epochs} epochs, bs={batch_size}")
    print(f"  LR: gate={lr_gate}, experts={lr_experts}, fusion={lr_fusion}")

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    save_every = int(config.get("moe_training", {}).get("save_every", 2))

    for epoch in range(epochs):
        routing_gate.train()
        expert_bank.train()
        fusion.train()

        running = {"mask_loss": 0.0, "lb_loss": 0.0, "total": 0.0}
        n_batches = 0

        for batch in loader:
            emb = batch["image_emb"].to(device)
            target = batch["lesion_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                # C3 → C5 → C6
                routing_weights = routing_gate(emb)
                expert_logits = expert_bank(emb)
                fusion_out = fusion(expert_logits, routing_weights)

                # Mask loss: fused mask vs ground truth
                mask_loss = bce_dice_loss(fusion_out.mask_fused_logits, target)

                # Load-balance auxiliary loss
                lb_loss = routing_load_balance_loss(routing_weights, num_experts=num_experts)

                total_loss = mask_loss + lb_weight * lb_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running["mask_loss"] += mask_loss.item()
            running["lb_loss"] += lb_loss.item()
            running["total"] += total_loss.item()
            n_batches += 1

        scheduler.step()

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        record = {"epoch": epoch, **avg}
        history.append(record)
        print(f"  epoch {epoch}: total={avg['total']:.4f}  mask={avg['mask_loss']:.4f}  lb={avg['lb_loss']:.4f}")

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            _save_moe_checkpoint(save_dir / "moe_best.pt", routing_gate, expert_bank, fusion, epoch)

        if (epoch + 1) % save_every == 0:
            _save_moe_checkpoint(save_dir / f"moe_epoch{epoch}.pt", routing_gate, expert_bank, fusion, epoch)

    # Save final
    final_path = _save_moe_checkpoint(save_dir / "moe_checkpoint.pt", routing_gate, expert_bank, fusion, epochs - 1)

    hist_path = save_dir / "moe_joint_history.jsonl"
    with hist_path.open("w") as f:
        for rec in history:
            f.write(json.dumps(rec) + "\n")

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint MoE training (Phase 2).")
    p.add_argument("--config", default="configs/moe.yaml")
    p.add_argument("--cache-dir", default=None, help="Cached embedding directory.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

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
