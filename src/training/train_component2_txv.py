from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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
    """Supports both min-mode (loss) and max-mode (AUROC)."""

    def __init__(self, patience: int = 3, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.stop = False

    def step(self, value: float) -> bool:
        improved = (self.mode == "max" and value > self.best) or (
            self.mode == "min" and value < self.best
        )
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return False


def build_optimizer(model: Component2SoftDomainContext, config: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(config["training"]["lr"])
    weight_decay = float(config["training"].get("weight_decay", 1e-5))
    tb_lr_factor = float(config["training"].get("tb_head_lr_factor", 5.0))
    routing_params = [p for p in model.domain_routing_head.parameters() if p.requires_grad]
    tb_params = [p for p in model.tb_head.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": routing_params, "lr": lr, "weight_decay": weight_decay},
            {"params": tb_params, "lr": lr * tb_lr_factor, "weight_decay": weight_decay},
        ]
    )


def tb_bce_loss(
    tb_logits: torch.Tensor,
    tb_labels: torch.Tensor,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """BCE loss computed only on samples that have a valid TB label (label != -1)."""
    valid = tb_labels != -1
    if not valid.any():
        return torch.tensor(0.0, device=tb_logits.device, requires_grad=False)
    logits_valid = tb_logits[valid].float().squeeze(1)
    labels_valid = tb_labels[valid].float()
    pw = (
        torch.tensor([pos_weight], device=tb_logits.device, dtype=torch.float32)
        if pos_weight is not None
        else None
    )
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits_valid, labels_valid, pos_weight=pw
    )


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


def _forward_both_heads(
    model: Component2SoftDomainContext,
    x_224: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run backbone once under no_grad, then compute both heads from the same pooled features.

    Backbone is frozen — running it twice per batch (old approach) wastes GPU memory and compute
    because autograd tracks activations through the frozen layers for no benefit.
    """
    with torch.no_grad():
        features = model.txv_model.features(x_224)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
    domain_ctx_raw = model.domain_routing_head(pooled)
    domain_ctx = F.normalize(domain_ctx_raw, p=2, dim=1)
    tb_logits = model.tb_head(pooled)
    return domain_ctx, tb_logits


def train_one_epoch(
    model: Component2SoftDomainContext,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    tb_head_weight: float = 1.0,
    pos_weight: float | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    running_items = 0

    for batch in loader:
        x_224 = batch["x_224"].to(device)
        domain_targets = batch["domain_id"].to(device)
        tb_labels = batch["tb_label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        domain_ctx, tb_logits = _forward_both_heads(model, x_224)

        contrastive = supervised_contrastive_loss(domain_ctx, domain_targets, temperature=temperature)
        tb_loss = tb_bce_loss(tb_logits, tb_labels, pos_weight=pos_weight)
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
    pos_weight: float | None = None,
) -> tuple[float, float]:
    """Returns (val_loss, val_tb_auroc). AUROC is nan if no valid TB labels found."""
    model.eval()
    running_loss = 0.0
    running_items = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        x_224 = batch["x_224"].to(device)
        domain_targets = batch["domain_id"].to(device)
        tb_labels = batch["tb_label"].to(device)

        domain_ctx, tb_logits = _forward_both_heads(model, x_224)
        contrastive = supervised_contrastive_loss(domain_ctx, domain_targets, temperature=temperature)
        tb_loss = tb_bce_loss(tb_logits, tb_labels, pos_weight=pos_weight)
        loss = contrastive + tb_head_weight * tb_loss

        batch_size = domain_targets.shape[0]
        running_loss += float(loss.item()) * batch_size
        running_items += batch_size

        all_logits.append(tb_logits.squeeze(1).cpu())
        all_labels.append(tb_labels.cpu())

    val_loss = running_loss / max(running_items, 1)

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    valid_mask = labels_cat != -1
    auroc = float("nan")
    if valid_mask.any() and labels_cat[valid_mask].unique().numel() >= 2:
        try:
            from sklearn.metrics import roc_auc_score
            probs = torch.sigmoid(logits_cat[valid_mask]).numpy()
            auroc = float(roc_auc_score(labels_cat[valid_mask].numpy(), probs))
        except Exception:
            pass

    return val_loss, auroc


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

    if model.active_backend != "xrv":
        raise RuntimeError(
            "Component 2 training requires torchxrayvision (active_backend='xrv'). "
            "Install it with: pip install torchxrayvision"
        )
    print(f"Backend: {model.active_backend} | Device: {device}")

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

    # Compute pos_weight from training labels to handle TB/normal class imbalance.
    from collections import Counter as _Counter
    label_counts = _Counter(
        _extract_tb_label(s) for s in train_samples if _extract_tb_label(s) is not None
    )
    n_pos = label_counts.get(1, 1)
    n_neg = label_counts.get(0, 1)
    pos_weight = float(n_neg / n_pos)
    print(f"TB label counts — pos: {n_pos}, neg: {n_neg} | pos_weight: {pos_weight:.2f}")

    # Early-stop on val TB AUROC (higher is better). Fall back to loss if AUROC is nan.
    early_stopper = EarlyStopping(patience=int(config["training"]["patience"]), mode="max")
    best_path = None

    print(f"\n{'Ep':>4} {'TrainLoss':>10} {'ValLoss':>9} {'ValAUROC':>9} {'':>20}")
    print("-" * 58)

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, temperature, tb_head_weight, pos_weight
        )
        val_loss, val_auroc = validate_one_epoch(
            model, val_loader, device, temperature, tb_head_weight, pos_weight
        )

        # Primary metric: AUROC; if still nan (first few epochs), use negative val_loss so
        # EarlyStopping in "max" mode still works (higher −loss = lower loss = better).
        monitor = val_auroc if not (val_auroc != val_auroc) else -val_loss  # nan check
        improved = early_stopper.step(monitor)

        auroc_str = f"{val_auroc:.4f}" if val_auroc == val_auroc else "  nan "
        status = " <-- saved" if improved else f" (patience {early_stopper.counter}/{early_stopper.patience})"
        print(f"{epoch + 1:>4} {train_loss:>10.4f} {val_loss:>9.4f} {auroc_str:>9}{status}")

        if improved:
            best_path = save_routing_head(model, config)

        if early_stopper.stop:
            print("\nEarly stopping triggered.")
            break

    print(f"\nTraining complete. Best checkpoint saved to {best_path}")
    print(f"Best val AUROC: {early_stopper.best:.4f}")


if __name__ == "__main__":
    main()
