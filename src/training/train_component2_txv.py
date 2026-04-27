from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.components.component2_txv import Component2SoftDomainContext, supervised_contrastive_loss
from src.core.device import pick_device
from src.core.seed import seed_everything
from src.training.train_component1_dann import (
    Component1DomainDataset,
    DomainSampleRef,
    build_component1_manifest,
    load_yaml_config,
    maybe_limit_manifest,
)
from src.components.component0_qc import harmonise_sample


def _extract_tb_label(sample: DomainSampleRef) -> int | None:
    """Extract binary TB label (1=TB, 0=normal) from the sample reference.

    Montgomery / Shenzhen: filename stem ends with ``_0`` (normal) or ``_1`` (TB).
    TBX11K: image path contains ``imgs/tb/`` (TB) or ``imgs/health/`` / ``sick_non_covid/`` (normal).
    NIH CXR-14: no TB labels — returns None so the sample is excluded from TB loss.
    """
    from pathlib import Path as _Path

    if sample.dataset_id in ("montgomery", "shenzhen"):
        source = _Path(sample.source if sample.image_path is None else sample.image_path)
        parts = source.stem.rsplit("_", 1)
        if len(parts) == 2:
            suffix = parts[-1]
            if suffix == "1":
                return 1
            if suffix == "0":
                return 0
        return None

    if sample.dataset_id == "tbx11k":
        path_str = sample.image_path or sample.source or ""
        path_parts = _Path(path_str).parts
        for i, part in enumerate(path_parts):
            if part == "imgs" and i + 1 < len(path_parts):
                sub = path_parts[i + 1]
                if sub == "tb":
                    return 1
                if sub in ("health", "sick_non_covid"):
                    return 0
        return None

    return None  # nih_cxr14 has no TB labels


class Component2DomainDataset(Component1DomainDataset):
    """Dataset providing the TXV-normalised [1, 224, 224] branch instead of [3, 1024, 1024].

    Each item also exposes a ``tb_label`` key:  0 or 1 when a label is available,
    -1 when the dataset carries no TB annotation (NIH, or unlabelled TBX11K subset).
    """

    def __init__(self, samples: list[DomainSampleRef]) -> None:
        # Component 2 doesn't use the DANN augmentations (gamma/poisson)
        super().__init__(samples, apply_augmentation=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self._load_image(sample)

        harmonised = harmonise_sample({
            "image": image,
            "dataset_id": sample.dataset_id,
            "source": sample.source
        })

        raw_label = _extract_tb_label(sample)
        tb_label = -1 if raw_label is None else int(raw_label)

        return {
            "x_224": harmonised.x_224,
            "domain_id": torch.tensor(sample.domain_id, dtype=torch.long),
            "tb_label": torch.tensor(tb_label, dtype=torch.long),
            "dataset_id": sample.dataset_id,
            "source": sample.source
        }


def collate_component2_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x_224": torch.stack([item["x_224"] for item in batch], dim=0),
        "domain_id": torch.stack([item["domain_id"] for item in batch], dim=0),
        "tb_label": torch.stack([item["tb_label"] for item in batch], dim=0),
        "dataset_id": [item["dataset_id"] for item in batch],
        "source": [item["source"] for item in batch],
    }


def stratified_split(samples: list[DomainSampleRef], val_ratio: float = 0.2, seed: int = 1337) -> tuple[list[DomainSampleRef], list[DomainSampleRef]]:
    """Split the dataset preserving the domain distribution."""
    import random
    from collections import defaultdict

    grouped: dict[int, list[DomainSampleRef]] = defaultdict(list)
    for s in samples:
        grouped[s.domain_id].append(s)

    rng = random.Random(seed)
    train_samples = []
    val_samples = []

    for _, items in grouped.items():
        rng.shuffle(items)
        split_idx = int(len(items) * (1.0 - val_ratio))
        train_samples.extend(items[:split_idx])
        val_samples.extend(items[split_idx:])

    return train_samples, val_samples


class EarlyStopping:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False

    def step(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return True  # Improvement detected
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return False


def build_optimizer(model: Component2SoftDomainContext, config: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(config["training"]["lr"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr)


def tb_bce_loss(tb_logits: torch.Tensor, tb_labels: torch.Tensor) -> torch.Tensor:
    """BCE loss computed only on samples that have a valid TB label (label != -1)."""
    valid = tb_labels != -1
    if not valid.any():
        return torch.tensor(0.0, device=tb_logits.device, requires_grad=False)
    logits_valid = tb_logits[valid].float().squeeze(1)
    labels_valid = tb_labels[valid].float()
    return torch.nn.functional.binary_cross_entropy_with_logits(logits_valid, labels_valid)


def save_routing_head(model: Component2SoftDomainContext, config: dict[str, Any]) -> Path:
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / str(config["training"]["save_name"])
    payload = {
        **{
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith("tb_head")
        },
        "tb_head": model.tb_head_state_dict(),
    }
    torch.save(payload, save_path)
    return save_path


def train_one_epoch(
    model: Component2SoftDomainContext,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    tb_head_weight: float = 1.0,
) -> float:
    model.train()
    running_loss = 0.0
    running_items = 0

    for batch in loader:
        x_224 = batch["x_224"].to(device)
        domain_targets = batch["domain_id"].to(device)
        tb_labels = batch["tb_label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        domain_ctx, _ = model(x_224)
        contrastive = supervised_contrastive_loss(domain_ctx, domain_targets, temperature=temperature)

        # Fix 1: add TB binary classification loss on top of contrastive objective
        tb_logits = model.forward_tb_logit(x_224)
        tb_loss = tb_bce_loss(tb_logits, tb_labels)
        loss = contrastive + tb_head_weight * tb_loss

        loss.backward()
        optimizer.step()

        batch_size = domain_targets.shape[0]
        running_loss += float(loss.item()) * batch_size
        running_items += batch_size

    return running_loss / max(running_items, 1)


@torch.no_grad()
def validate_one_epoch(
    model: Component2SoftDomainContext,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    temperature: float,
    tb_head_weight: float = 1.0,
) -> float:
    model.eval()
    running_loss = 0.0
    running_items = 0

    for batch in loader:
        x_224 = batch["x_224"].to(device)
        domain_targets = batch["domain_id"].to(device)
        tb_labels = batch["tb_label"].to(device)

        domain_ctx, _ = model(x_224)
        contrastive = supervised_contrastive_loss(domain_ctx, domain_targets, temperature=temperature)
        tb_logits = model.forward_tb_logit(x_224)
        tb_loss = tb_bce_loss(tb_logits, tb_labels)
        loss = contrastive + tb_head_weight * tb_loss

        batch_size = domain_targets.shape[0]
        running_loss += float(loss.item()) * batch_size
        running_items += batch_size

    return running_loss / max(running_items, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Component 2 TXV Routing Head.")
    parser.add_argument("--config", default="configs/component2_txv.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--component1_config", default="configs/component1_dann.yaml", help="To reuse the dataset split pointers.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)["component2_txv"]
    seed = int(config["training"]["seed"])
    seed_everything(seed)

    samples = build_component1_manifest(paths_config=args.paths, component1_config=args.component1_config)
    excluded_datasets = {
        str(dataset_id).lower()
        for dataset_id in config["data"].get("exclude_datasets", [])
    }
    if excluded_datasets:
        samples = [sample for sample in samples if sample.dataset_id not in excluded_datasets]
    samples = maybe_limit_manifest(samples, config["training"].get("limit_per_domain"))
    
    train_samples, val_samples = stratified_split(samples, val_ratio=config["data"]["val_split"], seed=seed)

    if args.dry_run:
        from collections import Counter
        print(json.dumps({
            "train_total": len(train_samples),
            "val_total": len(val_samples),
            "train_domains": dict(Counter(s.domain_id for s in train_samples)),
            "val_domains": dict(Counter(s.domain_id for s in val_samples)),
            "backend": config["model"]["backend"]
        }, indent=2))
        return

    device = pick_device(config["training"].get("device"))
    model = Component2SoftDomainContext(
        backend=config["model"]["backend"],
        weights=config["model"]["weights"]
    ).to(device)

    optimizer = build_optimizer(model, config)
    
    train_loader = DataLoader(
        Component2DomainDataset(train_samples),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_component2_batch,
    )
    val_loader = DataLoader(
        Component2DomainDataset(val_samples),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_component2_batch,
    )

    epochs = int(config["training"]["epochs"])
    temperature = float(config["training"]["temperature"])
    tb_head_weight = float(config["training"].get("tb_head_weight", 1.0))
    early_stopper = EarlyStopping(patience=int(config["training"]["patience"]))

    best_model_path = config["training"]["save_dir"]
    best_path = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, temperature, tb_head_weight)
        val_loss = validate_one_epoch(model, val_loader, device, temperature, tb_head_weight)
        
        improved = early_stopper.step(val_loss)
        
        status = " (improved --> saving routing head)" if improved else f" (patience: {early_stopper.counter}/{early_stopper.patience})"
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{status}")

        if improved:
            best_path = save_routing_head(model, config)

        if early_stopper.stop:
            print("Early stopping triggered due to no improvement on validation loss.")
            break

    print(f"Training complete. Best routing head saved natively to {best_path}")


if __name__ == "__main__":
    main()
