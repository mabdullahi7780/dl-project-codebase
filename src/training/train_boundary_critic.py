"""Boundary Critic training (Phase 3 of MoE training).

Trains the ResNet18 boundary critic from Component 7 to predict whether
a given mask boundary is anatomically plausible.  Positive examples come
from ground-truth lung/lesion boundaries; negative examples are
deliberately corrupted versions (random shifts, dilations, holes).

Usage::

    python -m src.training.train_boundary_critic \
        --config configs/moe.yaml \
        --cache-dir outputs/embedding_cache \
        --epochs 5
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

from src.components.component7_verification import (
    BoundaryCriticConfig,
    Component7BoundaryCritic,
)
from src.core.device import pick_device
from src.core.seed import seed_everything


def _corrupt_mask(mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Generate a deliberately bad boundary for negative samples."""
    m = mask.clone()
    if mode == "shift":
        # Random translation creating boundary misalignment
        dx, dy = torch.randint(-12, 13, (2,)).tolist()
        m = torch.roll(m, shifts=(dy, dx), dims=(-2, -1))
    elif mode == "dilate":
        # Over-dilate to spill outside lung
        for _ in range(5):
            m = F.max_pool2d(m, 3, stride=1, padding=1)
    elif mode == "holes":
        # Punch random holes
        n = 5
        h, w = m.shape[-2:]
        for _ in range(n):
            cy, cx = torch.randint(0, h, (1,)).item(), torch.randint(0, w, (1,)).item()
            r = 8
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            hole = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
            m = m * (~hole).float().unsqueeze(0).unsqueeze(0)
    return m.clamp(0, 1)


class BoundaryCriticDataset(Dataset):
    """Generates (image, mask) pairs with binary plausibility labels.

    Positive (label=1): real lesion mask + image.
    Negative (label=0): corrupted mask + image.
    """

    def __init__(self, cache_dir: Path | None = None, num_synthetic: int = 200) -> None:
        self.cache_dir = cache_dir
        self.corruption_modes = ["shift", "dilate", "holes"]
        if cache_dir is not None and cache_dir.exists():
            self.samples = sorted(cache_dir.glob("*.pt"))
        else:
            self.samples = list(range(num_synthetic))

    def __len__(self) -> int:
        return len(self.samples) * 2  # double for pos+neg

    def __getitem__(self, idx: int) -> dict[str, Any]:
        is_negative = idx % 2 == 1
        sample_idx = idx // 2

        if isinstance(self.samples[sample_idx], Path):
            data = torch.load(self.samples[sample_idx], weights_only=False)
            mask = data["lesion_mask"]
            image = data.get("image_1024", torch.zeros(1, 1024, 1024))
            lung = data.get("lung_mask", torch.ones(1, 256, 256))
        else:
            # Synthetic
            mask = torch.zeros(1, 256, 256)
            cx, cy = torch.randint(64, 192, (2,)).tolist()
            r = torch.randint(15, 40, (1,)).item()
            yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
            mask[0] = (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2).float()
            image = torch.rand(1, 1024, 1024)
            lung = torch.ones(1, 256, 256)

        if is_negative:
            mode = self.corruption_modes[idx % len(self.corruption_modes)]
            mask = _corrupt_mask(mask, mode)

        # Build the 3-channel critic input
        crop = Component7BoundaryCritic.prepare_crop(image, mask, lung)

        return {
            "crop": crop.squeeze(0),  # [3, 224, 224]
            "label": torch.tensor(0.0 if is_negative else 1.0),
        }


def train_boundary_critic(config: dict[str, Any], *, cache_dir: Path | None = None) -> Path:
    train_cfg = config.get("moe_training", {}).get("boundary_critic", {})
    epochs = int(train_cfg.get("epochs", 5))
    batch_size = int(train_cfg.get("batch_size", 16))
    lr = float(train_cfg.get("lr", 1e-3))
    save_dir = Path(config.get("moe_training", {}).get("save_dir", "outputs/moe_runs"))
    save_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(config.get("moe_training", {}).get("device"))
    seed_everything(1337)

    critic = Component7BoundaryCritic(
        BoundaryCriticConfig(
            pretrained=True,
            freeze_blocks_1_to_3=bool(train_cfg.get("freeze_blocks_1_to_3", True)),
        )
    ).to(device)

    dataset = BoundaryCriticDataset(cache_dir=cache_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    trainable = [p for p in critic.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    bce = nn.BCELoss()

    print(f"[Boundary critic] {epochs} epochs, bs={batch_size}, lr={lr}")
    history: list[dict[str, Any]] = []

    for epoch in range(epochs):
        critic.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for batch in loader:
            crops = batch["crop"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            scores = critic(crops)
            loss = bce(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (scores > 0.5).float()
            running_acc += (preds == labels).float().mean().item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        avg_acc = running_acc / max(n_batches, 1)
        record = {"epoch": epoch, "loss": avg_loss, "accuracy": avg_acc}
        history.append(record)
        print(f"  epoch {epoch}: loss={avg_loss:.4f} acc={avg_acc:.4f}")

    final_path = save_dir / "boundary_critic.pt"
    torch.save(critic.state_dict(), final_path)
    hist_path = save_dir / "boundary_critic_history.jsonl"
    with hist_path.open("w") as f:
        for rec in history:
            f.write(json.dumps(rec) + "\n")

    print(f"Boundary critic saved to {final_path}")
    return final_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Component 7 ResNet18 boundary critic.")
    p.add_argument("--config", default="configs/moe.yaml")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if args.epochs is not None:
        config.setdefault("moe_training", {}).setdefault("boundary_critic", {})["epochs"] = args.epochs

    if args.dry_run:
        print("[Dry run] boundary critic config:")
        print(json.dumps(config.get("moe_training", {}).get("boundary_critic", {}), indent=2))
        return

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    train_boundary_critic(config, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
