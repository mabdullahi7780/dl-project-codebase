from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.components.component0_qc import canonicalise_dataset_id
from src.components.component1_dann import (
    Component1DANNModel,
    DANNHead,
    DANNHeadConfig,
    compute_dann_lambda,
    domain_classification_loss,
)
from src.components.component1_encoder import (
    Component1EncoderConfig,
    LoRAConfig,
    build_component1_encoder,
    save_trainable_state_dict,
)
from src.core.device import pick_device
from src.core.seed import seed_everything
from src.data.harmonise import harmonise_sample


DOMAIN_TO_ID: dict[str, int] = {
    "montgomery": 0,
    "shenzhen": 1,
    "tbx11k": 2,
    "nih_cxr14": 3,
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]+))?\}")


@dataclass(slots=True)
class DomainSampleRef:
    dataset_id: str
    domain_id: int
    source: str
    image_path: str | None = None
    archive_path: str | None = None
    member_name: str | None = None


def apply_domain_adaptation_augment(
    x_3ch: torch.Tensor,
    *,
    gamma_range: tuple[float, float] = (0.7, 1.4),
    poisson_peaks: tuple[float, ...] = (32.0, 64.0, 128.0),
) -> torch.Tensor:
    """Apply the slide-specified gamma jitter + Poisson noise to grayscale triplets."""

    if x_3ch.shape[0] != 3:
        raise ValueError(f"Expected 3-channel tensor, got {tuple(x_3ch.shape)}.")

    single = x_3ch[:1].clamp(0.0, 1.0)
    gamma = float(torch.empty(1).uniform_(gamma_range[0], gamma_range[1]).item())
    peak_idx = int(torch.randint(len(poisson_peaks), (1,)).item())
    peak = float(poisson_peaks[peak_idx])

    gamma_adjusted = single.pow(gamma)
    noisy = torch.poisson((gamma_adjusted * peak).clamp_min(0.0)) / peak
    return noisy.clamp(0.0, 1.0).repeat(3, 1, 1)


class Component1DomainDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: list[DomainSampleRef],
        *,
        apply_augmentation: bool = False,
        augmentation_datasets: tuple[str, ...] = ("montgomery", "shenzhen"),
        gamma_range: tuple[float, float] = (0.7, 1.4),
        poisson_peaks: tuple[float, ...] = (32.0, 64.0, 128.0),
    ) -> None:
        self.samples = samples
        self._tar_cache: dict[str, tarfile.TarFile] = {}
        self.apply_augmentation = apply_augmentation
        self.augmentation_datasets = augmentation_datasets
        self.gamma_range = gamma_range
        self.poisson_peaks = poisson_peaks

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self._load_image(sample)
        harmonised = harmonise_sample(
            {
                "image": image,
                "dataset_id": sample.dataset_id,
                "source": sample.source,
            }
        )
        x_3ch = harmonised.x_3ch
        if self.apply_augmentation and sample.dataset_id in self.augmentation_datasets:
            x_3ch = apply_domain_adaptation_augment(
                x_3ch,
                gamma_range=self.gamma_range,
                poisson_peaks=self.poisson_peaks,
            )
        return {
            "x_3ch": x_3ch,
            "domain_id": torch.tensor(sample.domain_id, dtype=torch.long),
            "dataset_id": sample.dataset_id,
            "source": sample.source,
        }

    def _load_image(self, sample: DomainSampleRef) -> np.ndarray:
        if sample.image_path is not None:
            return _read_image_file(Path(sample.image_path))

        if sample.archive_path is None or sample.member_name is None:
            raise ValueError(f"Incomplete sample reference: {sample!r}")

        archive = self._tar_cache.get(sample.archive_path)
        if archive is None:
            archive = tarfile.open(sample.archive_path, mode="r:gz")
            self._tar_cache[sample.archive_path] = archive

        member = archive.extractfile(sample.member_name)
        if member is None:
            raise FileNotFoundError(f"Missing tar member {sample.member_name!r} in {sample.archive_path}.")
        with member:
            with Image.open(io.BytesIO(member.read())) as image:
                return np.asarray(image.convert("L"))


def collate_component1_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x_3ch": torch.stack([item["x_3ch"] for item in batch], dim=0),
        "domain_id": torch.stack([item["domain_id"] for item in batch], dim=0),
        "dataset_id": [item["dataset_id"] for item in batch],
        "source": [item["source"] for item in batch],
    }


def _read_image_file(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def _resolve_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(key, default)

        return ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {key: _resolve_env_vars(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(inner) for inner in value]
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return _resolve_env_vars(yaml.safe_load(handle))


def _iter_image_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []

    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        paths.append(path)
    return paths


def _build_generic_image_samples(dataset_id: str, root: Path) -> list[DomainSampleRef]:
    canonical = canonicalise_dataset_id(dataset_id)
    return [
        DomainSampleRef(
            dataset_id=canonical,
            domain_id=DOMAIN_TO_ID[canonical],
            source=str(path),
            image_path=str(path),
        )
        for path in _iter_image_paths(root)
    ]


def _load_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_nih_metadata_image_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "Image Index" not in (reader.fieldnames or []):
            raise ValueError(f"Expected `Image Index` column in NIH metadata CSV: {path}")
        return [row["Image Index"].strip() for row in reader if row.get("Image Index")]


def _build_tbx11k_samples(root: Path, list_name: str) -> list[DomainSampleRef]:
    list_path = root / "lists" / list_name
    if list_path.exists():
        rel_paths = _load_lines(list_path)
        samples: list[DomainSampleRef] = []
        for rel in rel_paths:
            image_path = root / "imgs" / rel
            if not image_path.exists():
                continue
            samples.append(
                DomainSampleRef(
                    dataset_id="tbx11k",
                    domain_id=DOMAIN_TO_ID["tbx11k"],
                    source=rel,
                    image_path=str(image_path),
                )
            )
        if samples:
            return samples
    return _build_generic_image_samples("tbx11k", root / "imgs")


def _normalise_nih_cache(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    normalised: dict[str, dict[str, str]] = {}
    for image_name, value in payload.items():
        if isinstance(value, str):
            normalised[image_name] = {
                "archive_path": value,
                "member_name": image_name,
            }
        else:
            normalised[image_name] = {
                "archive_path": str(value["archive_path"]),
                "member_name": str(value["member_name"]),
            }
    return normalised


def _build_nih_archive_index(images_root: Path, cache_path: Path | None = None) -> dict[str, dict[str, str]]:
    if cache_path is not None and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return _normalise_nih_cache(json.load(handle))

    archives = sorted(path for path in images_root.glob("*.tar.gz") if path.is_file())
    archive_index: dict[str, dict[str, str]] = {}
    for archive_path in archives:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                member_name = Path(member.name).name
                if member_name.startswith("._"):
                    continue
                if Path(member_name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                archive_index[member_name] = {
                    "archive_path": str(archive_path),
                    "member_name": member.name,
                }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(archive_index, handle, indent=2, sort_keys=True)
    return archive_index


def _build_nih_samples(
    root: Path,
    split_name: str | None,
    *,
    cache_path: Path | None = None,
    metadata_csv: str | None = None,
) -> list[DomainSampleRef]:
    archive_index = _build_nih_archive_index(root / "images", cache_path=cache_path)
    if split_name:
        split_path = root / split_name
        if not split_path.exists():
            raise FileNotFoundError(f"Expected NIH split file at {split_path}.")
        image_names = _load_lines(split_path)
    else:
        metadata_name = metadata_csv or "Data_Entry_2017_v2020.csv"
        metadata_path = root / metadata_name
        if not metadata_path.exists():
            raise FileNotFoundError(f"Expected NIH metadata CSV at {metadata_path}.")
        image_names = _load_nih_metadata_image_ids(metadata_path)

    samples: list[DomainSampleRef] = []
    for image_name in image_names:
        archive_ref = archive_index.get(image_name)
        if archive_ref is None:
            continue
        samples.append(
            DomainSampleRef(
                dataset_id="nih_cxr14",
                domain_id=DOMAIN_TO_ID["nih_cxr14"],
                source=image_name,
                archive_path=archive_ref["archive_path"],
                member_name=archive_ref["member_name"],
            )
        )
    return samples


def build_component1_manifest(
    *,
    paths_config: str | Path = "configs/paths.yaml",
    component1_config: str | Path = "configs/component1_dann.yaml",
) -> list[DomainSampleRef]:
    paths_yaml = load_yaml_config(paths_config)
    component1_yaml = load_yaml_config(component1_config)["component1_dann"]
    dataset_roots = paths_yaml["datasets"]
    cache_path = Path(component1_yaml["data"]["manifest_cache"])
    tbx_list = component1_yaml["data"]["tbx_list"]
    nih_split = component1_yaml["data"].get("nih_split")
    nih_metadata_csv = component1_yaml["data"].get("nih_metadata_csv")

    samples: list[DomainSampleRef] = []
    samples.extend(_build_generic_image_samples("montgomery", Path(dataset_roots["montgomery"])))
    samples.extend(_build_generic_image_samples("shenzhen", Path(dataset_roots["shenzhen"])))
    samples.extend(_build_tbx11k_samples(Path(dataset_roots["tbx11k"]), tbx_list))
    samples.extend(
        _build_nih_samples(
            Path(dataset_roots["nih_cxr14"]),
            nih_split,
            cache_path=cache_path,
            metadata_csv=nih_metadata_csv,
        )
    )
    return samples


def maybe_limit_manifest(samples: list[DomainSampleRef], limit_per_domain: int | None) -> list[DomainSampleRef]:
    if limit_per_domain is None:
        return samples

    grouped: dict[str, list[DomainSampleRef]] = {dataset_id: [] for dataset_id in DOMAIN_TO_ID}
    for sample in samples:
        grouped[sample.dataset_id].append(sample)

    limited: list[DomainSampleRef] = []
    for dataset_id, group in grouped.items():
        limited.extend(group[:limit_per_domain])
    return limited


def build_weighted_sampler(
    samples: list[DomainSampleRef],
    domain_weights: dict[str, float],
) -> WeightedRandomSampler:
    weights = [
        float(domain_weights.get(sample.dataset_id, 1.0))
        for sample in samples
    ]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(samples),
        replacement=True,
    )


def build_model(config: dict[str, Any]) -> Component1DANNModel:
    encoder_cfg = config["encoder"]
    lora_cfg = config["lora"]
    dann_cfg = config["dann_head"]

    encoder = build_component1_encoder(
        Component1EncoderConfig(
            backend=str(encoder_cfg["backend"]),
            checkpoint_path=encoder_cfg.get("checkpoint_path"),
            freeze_backbone=bool(encoder_cfg["freeze_backbone"]),
            input_size=int(encoder_cfg["input_size"]),
            patch_size=int(encoder_cfg["patch_size"]),
            embed_dim=int(encoder_cfg["embed_dim"]),
            depth=int(encoder_cfg["depth"]),
            num_heads=int(encoder_cfg["num_heads"]),
            mlp_ratio=float(encoder_cfg["mlp_ratio"]),
            lora=LoRAConfig(
                rank=int(lora_cfg["rank"]),
                alpha=float(lora_cfg["alpha"]),
                dropout=float(lora_cfg["dropout"]),
                target_modules=tuple(lora_cfg["target_modules"]),
            ),
        )
    )
    head = DANNHead(
        DANNHeadConfig(
            input_dim=int(dann_cfg["input_dim"]),
            hidden_dim=int(dann_cfg["hidden_dim"]),
            num_domains=int(dann_cfg["num_domains"]),
            dropout=float(dann_cfg["dropout"]),
        )
    )
    return Component1DANNModel(encoder=encoder, head=head)


def build_optimizer(model: Component1DANNModel, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    lora_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            lora_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(train_cfg["lr_lora"])},
            {"params": head_params, "lr": float(train_cfg["lr_dann"])},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )


def train_one_epoch(
    model: Component1DANNModel,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    device: torch.device,
    ramp_epochs: int,
    max_lambda: float,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_items = 0
    lambda_ = compute_dann_lambda(epoch, ramp_epochs=ramp_epochs, max_lambda=max_lambda)

    for batch in loader:
        x_3ch = batch["x_3ch"].to(device)
        domain_targets = batch["domain_id"].to(device)

        optimizer.zero_grad(set_to_none=True)
        _, dom_logits = model(x_3ch, lambda_=lambda_)
        loss = domain_classification_loss(dom_logits, domain_targets)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * int(domain_targets.shape[0])
        predictions = dom_logits.argmax(dim=1)
        running_correct += int((predictions == domain_targets).sum().item())
        running_items += int(domain_targets.shape[0])

    average_loss = running_loss / max(running_items, 1)
    accuracy = running_correct / max(running_items, 1)
    return {"loss": average_loss, "accuracy": accuracy, "lambda": lambda_}


def save_checkpoint(
    model: Component1DANNModel,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    config: dict[str, Any],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        save_path,
    )


def save_component1_artifacts(
    model: Component1DANNModel,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> Path:
    save_dir = Path(config["training"]["save_dir"])
    adapter_name = str(config["training"].get("adapter_save_name", "component1_adapters.safetensors"))
    adapter_path = save_trainable_state_dict(model, save_dir / adapter_name)

    if config["training"].get("save_full_checkpoint", False):
        save_name = str(config["training"]["save_name"])
        save_checkpoint(model, optimizer, metrics, config, save_dir / save_name)

    return adapter_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Component 1 DANN on the 4-domain CXR mix.")
    parser.add_argument("--config", default="configs/component1_dann.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)["component1_dann"]
    seed_everything(int(config["training"]["seed"]))

    samples = build_component1_manifest(paths_config=args.paths, component1_config=args.config)
    samples = maybe_limit_manifest(samples, config["training"].get("limit_per_domain"))
    if not samples:
        raise RuntimeError("Component 1 manifest is empty. Check HDD dataset paths in configs/paths.yaml.")

    if args.dry_run:
        counts = {
            dataset_id: sum(1 for sample in samples if sample.dataset_id == dataset_id)
            for dataset_id in DOMAIN_TO_ID
        }
        print(json.dumps({"counts": counts, "total": len(samples)}, indent=2))
        return

    device = pick_device(config["training"].get("device"))
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    sampler = build_weighted_sampler(samples, config["data"]["domain_sampling_weights"])
    loader = DataLoader(
        Component1DomainDataset(
            samples,
            apply_augmentation=bool(config["data"].get("apply_augmentation", True)),
            augmentation_datasets=tuple(config["data"].get("augmentation_datasets", ["montgomery", "shenzhen"])),
            gamma_range=tuple(config["data"].get("gamma_range", [0.7, 1.4])),
            poisson_peaks=tuple(config["data"].get("poisson_peaks", [32.0, 64.0, 128.0])),
        ),
        batch_size=int(config["training"]["batch_size"]),
        sampler=sampler,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_component1_batch,
    )

    epochs = int(config["training"]["epochs"])
    metrics: dict[str, float] = {}
    for epoch in range(epochs):
        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            epoch=epoch,
            device=device,
            ramp_epochs=int(config["training"]["grl_ramp_epochs"]),
            max_lambda=float(config["training"]["max_lambda"]),
        )
        print(
            f"epoch={epoch + 1}/{epochs} "
            f"loss={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"lambda={metrics['lambda']:.4f}"
        )

    adapter_path = save_component1_artifacts(model, optimizer, metrics, config)
    print(f"saved trainable adapters to {adapter_path}")


if __name__ == "__main__":
    main()
