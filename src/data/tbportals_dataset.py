"""PyTorch Dataset for the TB Portals manifest contract.

Reads the fixed manifest columns (see :mod:`src.data.tbportals`) and yields
samples for ALP regression + cavity classification + (optional) DANN country
adversary. Augmentation matches Kantipudi et al.: +/-15 deg rotation,
horizontal flip, up to 10% zoom; ImageNet normalization; 224x224 input.

Lung cropping is optional and decoupled: if ``crops_dir`` is given and a cached
crop exists for an image, it is used; otherwise the full image is resized. This
lets you validate the data pipeline end-to-end *before* wiring up MedSAM lung
crops (see ``scripts/cache_lung_crops.py``).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Separator used by tbportals.py to encode zip-resident DICOMs in the manifest.
_ZIP_SEP = "!DICOM!"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224


def _decode_dicom_array(ds) -> Image.Image:  # type: ignore[no-untyped-def]
    """Convert a pydicom Dataset to an RGB PIL Image (all three CXR variants)."""
    pi = getattr(ds, "PhotometricInterpretation", "MONOCHROME2").strip()
    arr = ds.pixel_array  # (H, W) or (H, W, 3)

    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        if pi == "MONOCHROME1":
            arr = arr.max() - arr  # invert: bright bg -> dark
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / max(hi - lo, 1.0) * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")

    # Color XC camera images
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / max(hi - lo, 1.0) * 255.0).clip(0, 255).astype(np.uint8)
    if pi.startswith("YBR"):
        return Image.fromarray(arr, mode="YCbCr").convert("RGB")
    return Image.fromarray(arr).convert("RGB")


def _load_image(path: str) -> Image.Image:
    """Load an image — supports plain files, DICOMs, and zip-resident DICOMs.

    TB Portals DICOMs come in three variants:
    - Grayscale MONOCHROME2 (most CXRs): uint8 or uint16
    - Grayscale MONOCHROME1 (inverted): inverted before normalisation
    - Color XC (camera-digitised): RGB or YBR; converted via PIL

    Zip-resident paths are encoded as ``"zip_path!DICOM!entry_name"`` by
    :func:`src.data.tbportals.build_dicom_index` when the DICOMs have not been
    extracted yet.  After running ``scripts/extract_cxr_dcm.ps1`` all paths
    become plain file paths and this branch is unused.
    """
    try:
        import pydicom  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pydicom is required for DICOM images. "
            "Install with:  pip install pydicom 'pylibjpeg[all]'"
        ) from exc

    # ── Zip-resident DICOM ────────────────────────────────────────────────────
    if _ZIP_SEP in path:
        sep_idx = path.index(_ZIP_SEP)
        zip_path = path[:sep_idx]
        entry_name = path[sep_idx + len(_ZIP_SEP):]
        with zipfile.ZipFile(zip_path, "r") as z:
            data = z.read(entry_name)
        if entry_name.lower().endswith(".dcm"):
            ds = pydicom.dcmread(io.BytesIO(data))
            return _decode_dicom_array(ds)
        return Image.open(io.BytesIO(data)).convert("RGB")

    # ── Standard image (PNG, JPEG, …) ─────────────────────────────────────────
    if Path(path).suffix.lower() != ".dcm":
        return Image.open(path).convert("RGB")

    # ── Extracted DICOM file ──────────────────────────────────────────────────
    ds = pydicom.dcmread(path)
    return _decode_dicom_array(ds)


def build_country_index(train_df: pd.DataFrame) -> dict[str, int]:
    """Map each training country to a contiguous index for the DANN head."""
    countries = sorted(train_df["country"].unique())
    return {c: i for i, c in enumerate(countries)}


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TBPortalsDataset(Dataset):
    """ALP + cavity dataset over a manifest split.

    Each item is a dict with:
        image:       FloatTensor [3, 224, 224]
        alp:         FloatTensor []  in [0, 1]   (alp_0_100 / 100)
        cavity:      FloatTensor []  in {0, 1}
        country_idx: LongTensor  []  (-1 if country not in country_to_idx)
        image_id, patient_id: str
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        train: bool,
        country_to_idx: dict[str, int] | None = None,
        crops_dir: str | Path | None = None,
        use_lung_crop: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.train = train
        self.country_to_idx = country_to_idx or {}
        self.crops_dir = Path(crops_dir) if crops_dir is not None else None
        self.use_lung_crop = use_lung_crop and self.crops_dir is not None
        self.tf = build_transforms(train)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> str:
        if self.use_lung_crop and self.crops_dir is not None:
            crop = self.crops_dir / f"{row['image_id']}.png"
            if crop.is_file():
                return str(crop)
        return str(row["image_path"])

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        img = _load_image(self._resolve_image_path(row))
        image = self.tf(img)

        country_idx = self.country_to_idx.get(str(row["country"]), -1)
        return {
            "image": image,
            "alp": torch.tensor(float(row["alp_0_100"]) / 100.0, dtype=torch.float32),
            "cavity": torch.tensor(float(row["cavity"]), dtype=torch.float32),
            "country_idx": torch.tensor(int(country_idx), dtype=torch.long),
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"]),
        }
