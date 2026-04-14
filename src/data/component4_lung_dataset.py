"""Manifest-driven dataset for Component 4 (lung mask) fine-tuning.

Manifest CSV columns (header row required):
    sample_id, image_path, mask_path, dataset, split

- ``image_path``: absolute or repo-relative path to a grayscale CXR.
- ``mask_path``: either
    * a single binary lung mask (any non-zero pixel = lung), OR
    * two paths separated by ``|`` to merge on-the-fly (left|right) — useful
      for Montgomery if ``scripts/merge_montgomery_lung_masks.py`` was not
      run ahead of time.
- ``dataset``: canonical dataset id (montgomery, shenzhen, tbx11k, nih_cxr14).
- ``split``: train | val | test.

Masks are resized to 256x256 (MedSAM low-res head) and binarised. Images are
harmonised via Component 0 to produce ``x_3ch`` at 1024x1024 in [0, 1].
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src.components.component0_qc import harmonise_sample


LOW_RES_MASK_SIZE = 256
SUPPORTED_SPLITS = {"train", "val", "test"}
REQUIRED_COLUMNS = ("sample_id", "image_path", "mask_path", "dataset", "split")


@dataclass(slots=True)
class Component4Record:
    sample_id: str
    image_path: Path
    mask_paths: tuple[Path, ...]
    dataset: str
    split: str


def _resolve_path(raw: str, *, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def parse_manifest(
    manifest_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    split: str | None = None,
) -> list[Component4Record]:
    """Parse a Component 4 manifest CSV into records.

    If ``split`` is given, only records with that split are returned.
    Raises ``ValueError`` if the header is wrong, a path is missing, or
    multiple mask paths are mixed with a single-file path.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Component 4 manifest not found: {manifest}")

    root = Path(repo_root).expanduser().resolve() if repo_root else manifest.parent
    records: list[Component4Record] = []
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

            image_path = _resolve_path(row["image_path"], repo_root=root)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{manifest}:{row_index} image_path does not exist: {image_path}"
                )

            raw_mask = (row["mask_path"] or "").strip()
            if not raw_mask:
                raise ValueError(f"{manifest}:{row_index} has empty mask_path.")
            mask_parts = [_resolve_path(p, repo_root=root) for p in raw_mask.split("|") if p.strip()]
            for mp in mask_parts:
                if not mp.is_file():
                    raise FileNotFoundError(
                        f"{manifest}:{row_index} mask_path component does not exist: {mp}"
                    )

            records.append(
                Component4Record(
                    sample_id=row["sample_id"].strip() or f"row{row_index}",
                    image_path=image_path,
                    mask_paths=tuple(mask_parts),
                    dataset=(row["dataset"] or "").strip().lower(),
                    split=row_split,
                )
            )

    if split is not None and not records:
        raise ValueError(f"Manifest {manifest} yielded zero records for split={split!r}.")
    return records


def load_grayscale_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def load_and_merge_binary_mask(mask_paths: tuple[Path, ...]) -> np.ndarray:
    """Union one or more PNG masks into a single {0,1} uint8 array.

    Raises ``ValueError`` if the resulting mask is empty or components have
    mismatched shapes.
    """

    if not mask_paths:
        raise ValueError("load_and_merge_binary_mask requires at least one mask path.")

    accumulator: np.ndarray | None = None
    for path in mask_paths:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("L"))
        binary = (arr > 0).astype(np.uint8)
        if accumulator is None:
            accumulator = binary
        else:
            if binary.shape != accumulator.shape:
                raise ValueError(
                    f"Mask shape mismatch when merging {mask_paths}: {binary.shape} vs {accumulator.shape}."
                )
            accumulator = np.maximum(accumulator, binary)

    assert accumulator is not None
    if int(accumulator.sum()) == 0:
        raise ValueError(f"Merged lung mask is empty for paths: {mask_paths}")
    return accumulator


def resize_mask_to(mask_uint8: np.ndarray, size: int) -> torch.Tensor:
    tensor = torch.from_numpy(mask_uint8).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(size, size), mode="nearest")
    return (resized.squeeze(0) > 0.5).to(dtype=torch.float32)  # [1, size, size]


class Component4LungDataset(Dataset[dict[str, Any]]):
    """Returns dicts: {x_3ch [3,1024,1024], mask [1,256,256], sample_id, dataset}."""

    def __init__(
        self,
        records: list[Component4Record],
        *,
        low_res_mask_size: int = LOW_RES_MASK_SIZE,
        apply_clahe: bool | None = None,
    ) -> None:
        if not records:
            raise ValueError("Component4LungDataset received zero records.")
        self.records = records
        self.low_res_mask_size = int(low_res_mask_size)
        self.apply_clahe = apply_clahe

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_np = load_grayscale_image(record.image_path)
        harmonised = harmonise_sample(
            {"image": image_np, "dataset_id": record.dataset},
            apply_clahe=self.apply_clahe,
        )
        mask_np = load_and_merge_binary_mask(record.mask_paths)
        mask_lowres = resize_mask_to(mask_np, self.low_res_mask_size)
        return {
            "x_3ch": harmonised.x_3ch,
            "mask": mask_lowres,
            "sample_id": record.sample_id,
            "dataset": record.dataset,
        }


def collate_component4_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x_3ch": torch.stack([item["x_3ch"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
    }
