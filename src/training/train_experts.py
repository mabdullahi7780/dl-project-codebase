"""Expert pretraining script (Phase 1 of MoE training).

Each expert decoder is pretrained independently on its specialised
pathology using ground-truth masks from TBX11K (which has bounding-box
annotations) and synthetic targets from the TXV Grad-CAM maps for
the relevant pathology class.

Usage::

    python -m src.training.train_experts \
        --config configs/moe.yaml \
        --paths configs/paths.yaml \
        --expert consolidation \
        --epochs 10

    # Or pretrain all experts sequentially:
    python -m src.training.train_experts --config configs/moe.yaml --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from src.components.component1_encoder import (
    Component1EncoderConfig,
    build_component1_encoder,
    load_trainable_state_dict,
)
from src.components.component1_dann import Component1DANNModel, DANNHead
from src.components.component5_experts import (
    ExpertBank,
    ExpertDecoderConfig,
    EXPERT_NAMES,
    LightweightExpertDecoder,
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
    """Combined BCE + Dice loss for mask prediction.

    Args:
        pred_logits: [B, 1, H, W] raw logits.
        target:      [B, 1, H, W] binary ground truth in {0, 1}.
        bce_weight:  weight for BCE term (Dice gets 1 - bce_weight).
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)

    pred_prob = torch.sigmoid(pred_logits)
    intersection = (pred_prob * target).sum(dim=(2, 3))
    union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
    dice = dice.mean()

    return bce_weight * bce + (1.0 - bce_weight) * dice


# ---------------------------------------------------------------------------
# Synthetic target dataset
# ---------------------------------------------------------------------------

class ExpertPretrainDataset(Dataset):
    """Generates (image_embedding, target_mask) pairs for expert pretraining.

    For now this is a placeholder that generates synthetic targets.  In a
    real run you would:

    1. Pre-compute and cache Component 1 image embeddings for the training
       set (frozen MedSAM ViT-B forward pass).
    2. Generate per-pathology pseudo-ground-truth masks from TBX11K bboxes
       or from TXV Grad-CAM maps thresholded per class.
    3. Load those cached embeddings and targets here.

    The ``--cache-dir`` flag points to the directory of pre-computed
    embeddings.
    """

    def __init__(
        self,
        expert_name: str,
        *,
        cache_dir: Path | None = None,
        num_synthetic: int = 200,
    ) -> None:
        self.expert_name = expert_name
        self.cache_dir = cache_dir

        if cache_dir is not None and cache_dir.exists():
            self.samples = sorted(cache_dir.glob("*.pt"))
        else:
            # Synthetic fallback for smoke testing
            self.samples = list(range(num_synthetic))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if isinstance(self.samples[idx], Path):
            data = torch.load(self.samples[idx], weights_only=False)
            return {
                "image_emb": data["image_emb"],       # [256, 64, 64]
                "target_mask": data["target_mask"],     # [1, 256, 256]
            }

        # Synthetic: random embedding + random blob mask
        emb = torch.randn(256, 64, 64)
        mask = torch.zeros(1, 256, 256)
        # Place a random elliptical blob
        cx, cy = torch.randint(64, 192, (2,)).tolist()
        rx, ry = torch.randint(10, 40, (2,)).tolist()
        yy, xx = torch.meshgrid(
            torch.arange(256), torch.arange(256), indexing="ij"
        )
        blob = ((xx - cx).float() / rx) ** 2 + ((yy - cy).float() / ry) ** 2
        mask[0] = (blob < 1.0).float()
        return {"image_emb": emb, "target_mask": mask}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_single_expert(
    expert_name: str,
    config: dict[str, Any],
    *,
    cache_dir: Path | None = None,
    device: torch.device | None = None,
) -> Path:
    """Pretrain a single expert decoder.

    Returns the path to the saved expert checkpoint.
    """
    moe_cfg = config.get("moe", {})
    train_cfg = config.get("moe_training", {}).get("pretrain", {})

    epochs = int(train_cfg.get("epochs", 10))
    batch_size = int(train_cfg.get("batch_size", 4))
    lr = float(train_cfg.get("lr", 1e-4))
    save_dir = Path(config.get("moe_training", {}).get("save_dir", "outputs/moe_runs"))
    save_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = pick_device(config.get("moe_training", {}).get("device"))

    expert = LightweightExpertDecoder(
        expert_name=expert_name,
        in_channels=int(moe_cfg.get("expert_in_channels", 256)),
        mid_channels=int(moe_cfg.get("expert_mid_channels", 128)),
        dropout=float(moe_cfg.get("expert_dropout", 0.1)),
    ).to(device)

    dataset = ExpertPretrainDataset(expert_name, cache_dir=cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config.get("moe_training", {}).get("num_workers", 2)),
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(expert.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = bool(config.get("moe_training", {}).get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    print(f"[Expert pretrain] {expert_name}: {epochs} epochs, bs={batch_size}, lr={lr}")

    history: list[dict[str, Any]] = []
    best_loss = float("inf")

    for epoch in range(epochs):
        expert.train()
        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            emb = batch["image_emb"].to(device)
            target = batch["target_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                pred_logits = expert(emb)
                loss = bce_dice_loss(pred_logits, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(n_batches, 1)
        record = {"epoch": epoch, "expert": expert_name, "loss": avg_loss}
        history.append(record)
        print(f"  epoch {epoch}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = save_dir / f"expert_{expert_name}_best.pt"
            torch.save(expert.state_dict(), ckpt_path)

    # Save final + history
    final_path = save_dir / f"expert_{expert_name}_final.pt"
    torch.save(expert.state_dict(), final_path)

    hist_path = save_dir / f"expert_{expert_name}_history.jsonl"
    with hist_path.open("w") as f:
        for rec in history:
            f.write(json.dumps(rec) + "\n")

    print(f"  Expert {expert_name} saved to {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain MoE expert decoders (Phase 1).")
    p.add_argument("--config", default="configs/moe.yaml")
    p.add_argument("--expert", choices=list(EXPERT_NAMES), default=None,
                   help="Which expert to train (omit for --all).")
    p.add_argument("--all", action="store_true", help="Train all experts sequentially.")
    p.add_argument("--cache-dir", default=None, help="Pre-computed embedding cache dir.")
    p.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if args.epochs is not None:
        config.setdefault("moe_training", {}).setdefault("pretrain", {})["epochs"] = args.epochs

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    device = pick_device(config.get("moe_training", {}).get("device"))

    experts_to_train = list(EXPERT_NAMES) if args.all else ([args.expert] if args.expert else list(EXPERT_NAMES))

    if args.dry_run:
        print(f"[Dry run] Would train experts: {experts_to_train}")
        return

    for name in experts_to_train:
        train_single_expert(name, config, cache_dir=cache_dir, device=device)


if __name__ == "__main__":
    main()
