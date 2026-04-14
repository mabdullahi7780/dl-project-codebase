"""Fine-tune the Component 4 lung mask decoder (MedSAM ViT-B).

The image encoder and prompt encoder are frozen. Only the mask decoder
is optimised. Loss is BCE + Dice on the 256x256 low-res head.

Best checkpoint is selected by **highest validation Dice** (not lowest loss):
Dice is the downstream metric we actually care about for lung segmentation,
and BCE+Dice loss can drift while Dice still improves because BCE dominates
on easy background pixels. Best-by-Dice also matches the evaluation script.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.components.component4_lung import Component4MedSAM, bce_dice_loss
from src.core.device import describe_device, pick_device
from src.core.seed import seed_everything
from src.data.component4_lung_dataset import (
    Component4LungDataset,
    collate_component4_batch,
    parse_manifest,
)
from src.training.train_component1_dann import load_yaml_config


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    dice: float
    iou: float

    def as_dict(self) -> dict[str, float]:
        return {"loss": self.loss, "dice": self.dice, "iou": self.iou}


def _binary_dice_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float,
    smooth: float = 1e-5,
) -> tuple[float, float]:
    preds = (torch.sigmoid(logits) > threshold).to(dtype=targets.dtype)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    preds_sum = preds.sum(dim=(1, 2, 3))
    targets_sum = targets.sum(dim=(1, 2, 3))
    union = preds_sum + targets_sum
    dice = (2.0 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (preds_sum + targets_sum - intersection + smooth)
    return float(dice.mean().item()), float(iou.mean().item())


def _maybe_autocast(device: torch.device, amp_enabled: bool):
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def _build_loader(
    manifest_path: str,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    apply_clahe: bool | None,
    max_samples: int | None,
    repo_root: Path,
) -> DataLoader:
    records = parse_manifest(manifest_path, repo_root=repo_root, split=split)
    if max_samples is not None and max_samples > 0:
        records = records[: int(max_samples)]
    dataset = Component4LungDataset(records, apply_clahe=apply_clahe)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_component4_batch,
        pin_memory=False,
        drop_last=False,
    )


def _train_one_epoch(
    model: Component4MedSAM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    threshold: float,
    log_every: int,
    epoch: int,
) -> EpochMetrics:
    model.train()
    # Keep frozen branches deterministic (BatchNorm etc.).
    model.encoder.eval()
    if hasattr(model, "prompt_encoder"):
        model.prompt_encoder.eval()

    running_loss, running_dice, running_iou = 0.0, 0.0, 0.0
    n_batches = 0

    for step, batch in enumerate(loader, start=1):
        x_3ch = batch["x_3ch"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _maybe_autocast(device, amp_enabled):
            logits = model.forward_logits(x_3ch)
            loss = bce_dice_loss(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            dice, iou = _binary_dice_iou(logits.detach().float(), targets, threshold=threshold)

        running_loss += float(loss.item())
        running_dice += dice
        running_iou += iou
        n_batches += 1

        if log_every and step % log_every == 0:
            print(
                f"[epoch {epoch} step {step}/{len(loader)}] "
                f"loss={loss.item():.4f} dice={dice:.4f} iou={iou:.4f}"
            )

    denom = max(n_batches, 1)
    return EpochMetrics(
        loss=running_loss / denom,
        dice=running_dice / denom,
        iou=running_iou / denom,
    )


@torch.no_grad()
def _validate(
    model: Component4MedSAM,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float,
) -> EpochMetrics:
    model.eval()
    running_loss, running_dice, running_iou = 0.0, 0.0, 0.0
    n_batches = 0
    for batch in loader:
        x_3ch = batch["x_3ch"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        logits = model.forward_logits(x_3ch)
        loss = bce_dice_loss(logits, targets)
        dice, iou = _binary_dice_iou(logits.float(), targets, threshold=threshold)
        running_loss += float(loss.item())
        running_dice += dice
        running_iou += iou
        n_batches += 1
    denom = max(n_batches, 1)
    return EpochMetrics(
        loss=running_loss / denom,
        dice=running_dice / denom,
        iou=running_iou / denom,
    )


def _save_checkpoint(
    path: Path,
    *,
    model: Component4MedSAM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "decoder_state_dict": model.decoder_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_dice": float(best_dice),
        "config": config,
        "model_type": model.model_type,
        "mask_threshold": model.mask_threshold,
    }
    torch.save(payload, path)


def _maybe_resume(
    resume_path: str | None,
    *,
    model: Component4MedSAM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    if not resume_path:
        return 0, -1.0
    checkpoint_file = Path(resume_path).expanduser()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_file}")
    payload = torch.load(checkpoint_file, map_location=device)
    model.load_decoder_state_dict(payload["decoder_state_dict"])
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    start_epoch = int(payload.get("epoch", 0)) + 1
    best_dice = float(payload.get("best_dice", -1.0))
    print(f"Resumed from {checkpoint_file} at epoch {start_epoch}, best_dice={best_dice:.4f}")
    return start_epoch, best_dice


def _dry_run(
    model: Component4MedSAM,
    loader: DataLoader,
    *,
    device: torch.device,
) -> None:
    print(f"Dry run: loaded {len(loader.dataset)} records from manifest.")
    batch = next(iter(loader))
    x_3ch = batch["x_3ch"].to(device)
    targets = batch["mask"].to(device)
    print(
        f"Dry run shapes: x_3ch={tuple(x_3ch.shape)} dtype={x_3ch.dtype} "
        f"range=[{float(x_3ch.min()):.3f}, {float(x_3ch.max()):.3f}]; "
        f"mask={tuple(targets.shape)} sum={int(targets.sum().item())}"
    )
    model.eval()
    with torch.no_grad():
        logits = model.forward_logits(x_3ch)
    print(f"Dry run forward OK: logits={tuple(logits.shape)} device={logits.device}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Component 4 Lung Mask Decoder.")
    parser.add_argument("--config", default="configs/component4_lung.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Load one batch, forward once, exit.")
    parser.add_argument("--resume", default=None, help="Override resume checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_root = load_yaml_config(args.config)
    cfg = cfg_root["component4_lung"]
    training_cfg = cfg["training"]
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    seed_everything(int(training_cfg.get("seed", 1337)))
    device = pick_device(training_cfg.get("device"))
    print(f"Device: {describe_device(device)}")

    amp_enabled = bool(training_cfg.get("amp", False)) and device.type == "cuda"
    if bool(training_cfg.get("amp", False)) and device.type != "cuda":
        print(f"AMP requested but device is {device.type}; AMP disabled.")

    repo_root = Path(args.config).expanduser().resolve().parents[1]
    train_manifest = data_cfg["train_manifest"]
    val_manifest = data_cfg.get("val_manifest", train_manifest)

    apply_clahe = data_cfg.get("apply_clahe")
    max_train_samples = training_cfg.get("max_train_samples")
    max_val_samples = training_cfg.get("max_val_samples")

    train_loader = _build_loader(
        train_manifest,
        "train",
        batch_size=int(training_cfg.get("batch_size", 2)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        shuffle=True,
        apply_clahe=apply_clahe,
        max_samples=max_train_samples,
        repo_root=repo_root,
    )
    val_loader = _build_loader(
        val_manifest,
        "val",
        batch_size=int(training_cfg.get("batch_size", 2)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        shuffle=False,
        apply_clahe=apply_clahe,
        max_samples=max_val_samples,
        repo_root=repo_root,
    )

    model = Component4MedSAM(
        backend=model_cfg.get("backend", "auto"),
        checkpoint_path=model_cfg.get("checkpoint_path"),
        model_type=model_cfg.get("model_type", "vit_b"),
        mask_threshold=float(model_cfg.get("mask_threshold", 0.5)),
    ).to(device)
    print(f"Component 4 backend: {model.active_backend}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params)}")

    if args.dry_run:
        _dry_run(model, train_loader, device=device)
        return

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(training_cfg.get("lr", 5e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

    resume_path = args.resume or training_cfg.get("resume_checkpoint")
    start_epoch, best_dice = _maybe_resume(resume_path, model=model, optimizer=optimizer, device=device)

    save_dir = Path(training_cfg["save_dir"]).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    best_name = training_cfg.get("save_name", "component4_mask_decoder.pt")
    best_path = save_dir / best_name
    last_path = save_dir / f"last_{best_name}"
    history_path = save_dir / "training_history.jsonl"

    threshold = float(model_cfg.get("mask_threshold", 0.5))
    epochs = int(training_cfg.get("epochs", 50))
    log_every = int(training_cfg.get("log_every", 10))
    save_every = int(training_cfg.get("save_every", 1))
    patience = int(training_cfg.get("patience", 10))

    epochs_since_improvement = 0
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            amp_enabled=amp_enabled,
            scaler=scaler,
            threshold=threshold,
            log_every=log_every,
            epoch=epoch,
        )
        val_metrics = _validate(model, val_loader, device=device, threshold=threshold)
        elapsed = time.time() - t0
        print(
            f"[epoch {epoch}] train loss={train_metrics.loss:.4f} dice={train_metrics.dice:.4f} "
            f"iou={train_metrics.iou:.4f} | val loss={val_metrics.loss:.4f} "
            f"dice={val_metrics.dice:.4f} iou={val_metrics.iou:.4f} | {elapsed:.1f}s"
        )

        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train": train_metrics.as_dict(),
                        "val": val_metrics.as_dict(),
                    }
                )
                + "\n"
            )

        improved = val_metrics.dice > best_dice
        if improved:
            best_dice = val_metrics.dice
            epochs_since_improvement = 0
            _save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_dice=best_dice,
                config=cfg,
            )
            print(f"  -> new best val dice={best_dice:.4f}, saved {best_path}")
        else:
            epochs_since_improvement += 1

        if save_every and ((epoch + 1) % save_every == 0):
            _save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_dice=best_dice,
                config=cfg,
            )

        if patience and epochs_since_improvement >= patience:
            print(f"Early stop: no val Dice improvement for {patience} epochs.")
            break

    print(f"Training complete. Best val Dice={best_dice:.4f}. Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
