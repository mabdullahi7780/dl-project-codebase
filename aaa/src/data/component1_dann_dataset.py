"""Manifest-driven dataset for Component 1 (DANN) fine-tuning.

Component 1 only needs an image and a domain label. There are no masks.
This module mirrors the Component 4 manifest pattern so Kaggle / Colab
runs can use a single CSV instead of walking dataset roots by hand.

Manifest CSV (header row required):

    sample_id, image_path, dataset, split

- ``image_path``: absolute or repo-relative path to a grayscale CXR.
- ``dataset``:    ``montgomery`` | ``shenzhen`` | ``tbx11k`` | ``nih_cxr14``.
- ``split``:      ``train`` | ``val``.

Each item from the dataset is a dict
``{"x_3ch": [3, 1024, 1024], "domain_id": LongTensor, "dataset_id": str}``
ready to feed straight into ``Component1DANNModel``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.components.component0_qc import canonicalise_dataset_id
from src.data.harmonise import harmonise_sample


SUPPORTED_SPLITS = {"train", "val"}
REQUIRED_COLUMNS = ("sample_id", "image_path", "dataset", "split")

DOMAIN_TO_ID: dict[str, int] = {
    "montgomery": 0,
    "shenzhen": 1,
    "tbx11k": 2,
    "nih_cxr14": 3,
}


@dataclass(slots=True)
class Component1Record:
    sample_id: str
    image_path: Path
    dataset: str
    domain_id: int
    split: str


def _resolve_path(raw: str, *, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def parse_component1_manifest(
    manifest_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    split: str | None = None,
) -> list[Component1Record]:
    """Parse a Component 1 manifest CSV into typed records.

    Raises ``FileNotFoundError`` if the manifest or any image file is missing.
    Raises ``ValueError`` for missing columns, unknown datasets, or empty splits.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Component 1 manifest not found: {manifest}")

    root = Path(repo_root).expanduser().resolve() if repo_root else manifest.parent
    records: list[Component1Record] = []

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"Manifest {manifest} is missing required columns: {missing}. "
                f"Expected header: {REQUIRED_COLUMNS}."
            )

        for row_index, row in enumerate(reader, start=2):
            row_split = (row["split"] or "").strip().lower()
            if row_split not in SUPPORTED_SPLITS:
                raise ValueError(
                    f"{manifest}:{row_index} has unsupported split {row_split!r}. "
                    f"Expected one of {sorted(SUPPORTED_SPLITS)}."
                )
            if split is not None and row_split != split:
                continue

            dataset_raw = (row["dataset"] or "").strip().lower()
            try:
                dataset_id = canonicalise_dataset_id(dataset_raw)
            except ValueError as exc:
                raise ValueError(f"{manifest}:{row_index} {exc}") from exc
            if dataset_id not in DOMAIN_TO_ID:
                raise ValueError(
                    f"{manifest}:{row_index} dataset {dataset_id!r} has no domain id. "
                    f"Expected one of {sorted(DOMAIN_TO_ID)}."
                )

            image_path = _resolve_path(row["image_path"], repo_root=root)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{manifest}:{row_index} image_path does not exist: {image_path}"
                )

            records.append(
                Component1Record(
                    sample_id=row["sample_id"].strip() or f"row{row_index}",
                    image_path=image_path,
                    dataset=dataset_id,
                    domain_id=DOMAIN_TO_ID[dataset_id],
                    split=row_split,
                )
            )

    if split is not None and not records:
        raise ValueError(f"Manifest {manifest} yielded zero records for split={split!r}.")
    return records


def load_grayscale_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def _gamma_poisson_augment(
    x_3ch: torch.Tensor,
    *,
    gamma_range: tuple[float, float],
    poisson_peaks: tuple[float, ...],
) -> torch.Tensor:
    """Slide-spec gamma jitter + Poisson noise on the grayscale triplet."""

    single = x_3ch[:1].clamp(0.0, 1.0)
    gamma = float(torch.empty(1).uniform_(gamma_range[0], gamma_range[1]).item())
    peak_idx = int(torch.randint(len(poisson_peaks), (1,)).item())
    peak = float(poisson_peaks[peak_idx])

    gamma_adjusted = single.pow(gamma)
    noisy = torch.poisson((gamma_adjusted * peak).clamp_min(0.0)) / peak
    return noisy.clamp(0.0, 1.0).repeat(3, 1, 1)


class Component1ManifestDataset(Dataset[dict[str, Any]]):
    """Returns ``{x_3ch [3,1024,1024], domain_id, dataset_id, sample_id}``.

    Apply Component 0 harmonisation, optionally followed by gamma/Poisson
    augmentation on the configured domains (Montgomery + Shenzhen by default).
    """

    def __init__(
        self,
        records: list[Component1Record],
        *,
        apply_augmentation: bool = False,
        augmentation_datasets: tuple[str, ...] = ("montgomery", "shenzhen"),
        gamma_range: tuple[float, float] = (0.7, 1.4),
        poisson_peaks: tuple[float, ...] = (32.0, 64.0, 128.0),
    ) -> None:
        if not records:
            raise ValueError("Component1ManifestDataset received zero records.")
        self.records = records
        self.apply_augmentation = bool(apply_augmentation)
        self.augmentation_datasets = tuple(augmentation_datasets)
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))
        self.poisson_peaks = tuple(float(peak) for peak in poisson_peaks)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_np = load_grayscale_image(record.image_path)
        harmonised = harmonise_sample(
            {
                "image": image_np,
                "dataset_id": record.dataset,
                "source": str(record.image_path),
            }
        )
        x_3ch = harmonised.x_3ch
        if self.apply_augmentation and record.dataset in self.augmentation_datasets:
            x_3ch = _gamma_poisson_augment(
                x_3ch,
                gamma_range=self.gamma_range,
                poisson_peaks=self.poisson_peaks,
            )
        return {
            "x_3ch": x_3ch,
            "domain_id": torch.tensor(record.domain_id, dtype=torch.long),
            "dataset_id": record.dataset,
            "sample_id": record.sample_id,
        }


def collate_component1_manifest_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x_3ch": torch.stack([item["x_3ch"] for item in batch], dim=0),
        "domain_id": torch.stack([item["domain_id"] for item in batch], dim=0),
        "dataset_id": [item["dataset_id"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
    }
