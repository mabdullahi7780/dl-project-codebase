from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from src.components.component4_lung import Component4MedSAM, bce_dice_loss
from src.core.device import pick_device
from src.core.seed import seed_everything
from src.training.train_component1_dann import (
    load_yaml_config,
)

def dice_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> float:
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return float((2.0 * intersection + smooth) / (union + smooth))

def collate_component4_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x_3ch": torch.stack([item["x_3ch"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "dataset_id": [item["dataset_id"] for item in batch],
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Component 4 Lung Mask.")
    parser.add_argument("--config", default="configs/component4_lung.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)["component4_lung"]
    seed_everything(int(config["training"]["seed"]))

    if args.dry_run:
        print("Dry run OK for Component 4.")
        return

    device = pick_device(config["training"].get("device"))
    model = Component4MedSAM(backend=config["model"].get("backend", "mock")).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(config["training"]["lr"]))
    
    print(f"Model parameters: {sum(p.numel() for p in trainable_params)} trainable.")

if __name__ == "__main__":
    main()
