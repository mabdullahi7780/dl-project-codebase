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


class Component2DomainDataset(Component1DomainDataset):
    """Dataset providing the TXV-normalised [1, 224, 224] branch instead of [3, 1024, 1024]."""

    def __init__(self, samples: list[DomainSampleRef]) -> None:
        # Component 2 doesn't use the DANN augmentations (gamma/poisson)
        super().__init__(samples, apply_augmentation=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self._load_image(sample)
        
        # QC & Normalization via Component 0 logic
        harmonised = harmonise_sample({
            "image": image,
            "dataset_id": sample.dataset_id,
            "source": sample.source
        })

        return {
            "x_224": harmonised.x_224,
            "domain_id": torch.tensor(sample.domain_id, dtype=torch.long),
            "dataset_id": sample.dataset_id,
            "source": sample.source
        }


def collate_component2_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x_224": torch.stack([item["x_224"] for item in batch], dim=0),
        "domain_id": torch.stack([item["domain_id"] for item in batch], dim=0),
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


def save_routing_head(model: Component2SoftDomainContext, config: dict[str, Any]) -> Path:
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / str(config["training"]["save_name"])
    
    payload = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(payload, save_path)
    return save_path


def train_one_epoch(
    model: Component2SoftDomainContext,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float
) -> float:
    model.train()
    running_loss = 0.0
    running_items = 0

    for batch in loader:
        x_224 = batch["x_224"].to(device)
        domain_targets = batch["domain_id"].to(device)

        optimizer.zero_grad(set_to_none=True)
        domain_ctx, _ = model(x_224)
        
        loss = supervised_contrastive_loss(domain_ctx, domain_targets, temperature=temperature)
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
    temperature: float
) -> float:
    model.eval()
    running_loss = 0.0
    running_items = 0

    for batch in loader:
        x_224 = batch["x_224"].to(device)
        domain_targets = batch["domain_id"].to(device)

        domain_ctx, _ = model(x_224)
        loss = supervised_contrastive_loss(domain_ctx, domain_targets, temperature=temperature)
        
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
    early_stopper = EarlyStopping(patience=int(config["training"]["patience"]))

    best_model_path = config["training"]["save_dir"]
    best_path = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, temperature)
        val_loss = validate_one_epoch(model, val_loader, device, temperature)
        
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
