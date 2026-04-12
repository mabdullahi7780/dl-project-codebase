from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from src.core.constants import (
    ASPECT_RATIO_RANGE,
    DATASET_BIT_DEPTH_DIVISORS,
    DATASET_ID_ALIASES,
    DEFAULT_CLAHE_BY_DATASET,
    MANDATORY_CLAHE_DATASETS,
    MIN_IMAGE_EDGE_PX,
    VALID_PA_VIEWS,
    X1024_SIZE,
    X224_SIZE,
)
from src.core.types import HarmonisedCXR
from src.data.transforms_qc import apply_clahe_2d, center_crop_square, resize_chw, to_grayscale_tensor


def canonicalise_dataset_id(dataset_id: str) -> str:
    """Map dataset aliases to the canonical repository identifier."""

    key = dataset_id.strip().lower()
    try:
        return DATASET_ID_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_BIT_DEPTH_DIVISORS))
        raise ValueError(f"Unsupported dataset_id={dataset_id!r}. Expected one of: {valid}.") from exc


def validate_view(view: str | None) -> str | None:
    """Accept only PA projections when view metadata is available."""

    if view is None:
        return None

    normalised = str(view).strip().upper().replace("-", "").replace("_", "")
    if normalised in VALID_PA_VIEWS:
        return "PA"

    raise ValueError(
        f"Component 0 only accepts PA studies when projection metadata exists. Got view={view!r}."
    )


def validate_image_shape(
    image: torch.Tensor,
    *,
    min_edge: int = MIN_IMAGE_EDGE_PX,
    aspect_ratio_range: tuple[float, float] = ASPECT_RATIO_RANGE,
) -> None:
    """Reject clearly invalid images before any resizing happens."""

    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale [H, W] tensor after conversion, got shape={tuple(image.shape)}.")

    height, width = int(image.shape[0]), int(image.shape[1])
    if min(height, width) < int(min_edge):
        raise ValueError(
            f"Rejected image smaller than the required minimum edge {min_edge}px. Got {height}x{width}."
        )

    aspect_ratio = width / height
    lower, upper = aspect_ratio_range
    if not (lower <= aspect_ratio <= upper):
        raise ValueError(
            f"Rejected image with aspect ratio {aspect_ratio:.3f}; expected within [{lower}, {upper}]."
        )

    if not bool(torch.isfinite(image).all()):
        raise ValueError("Rejected image because it contains non-finite values.")

    if torch.max(image) <= torch.min(image):
        raise ValueError("Rejected image because it has no intensity variation.")


def normalise_by_dataset(image: torch.Tensor | Any, dataset_id: str) -> torch.Tensor:
    """Normalize raw grayscale pixels into the [0, 1] range using dataset-specific bit depth."""

    dataset_key = canonicalise_dataset_id(dataset_id)
    grayscale = to_grayscale_tensor(image)

    if bool(torch.isfinite(grayscale).all()) and float(grayscale.min()) >= 0.0 and float(grayscale.max()) <= 1.0:
        return grayscale.clamp(0.0, 1.0)

    divisor = DATASET_BIT_DEPTH_DIVISORS[dataset_key]
    return (grayscale / divisor).clamp(0.0, 1.0)


def make_x1024(image: torch.Tensor | Any) -> torch.Tensor:
    """Create the high-resolution harmonised tensor with shape [1, 1024, 1024]."""

    grayscale = to_grayscale_tensor(image)
    return resize_chw(grayscale, size=(X1024_SIZE, X1024_SIZE))


def make_x224_txv(image: torch.Tensor | Any) -> torch.Tensor:
    """Approximate the TorchXRayVision crop-then-resize preprocessing path."""

    grayscale = to_grayscale_tensor(image)
    cropped = center_crop_square(grayscale)
    return resize_chw(cropped, size=(X224_SIZE, X224_SIZE))


def apply_clahe_x1024(
    x_1024: torch.Tensor,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> torch.Tensor:
    """Apply CLAHE only to the 1024-resolution branch."""

    if x_1024.shape != (1, X1024_SIZE, X1024_SIZE):
        raise ValueError(
            f"CLAHE expects x_1024 with shape (1, {X1024_SIZE}, {X1024_SIZE}); got {tuple(x_1024.shape)}."
        )

    return apply_clahe_2d(
        x_1024,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )


def resolve_clahe_setting(dataset_id: str, apply_clahe: bool | None) -> bool:
    """Enforce the baseline CLAHE defaults and Montgomery requirement."""

    if dataset_id in MANDATORY_CLAHE_DATASETS:
        if apply_clahe is False:
            raise ValueError(f"CLAHE is mandatory for dataset_id={dataset_id!r}.")
        return True

    if apply_clahe is None:
        return DEFAULT_CLAHE_BY_DATASET.get(dataset_id, False)

    return bool(apply_clahe)


def build_meta(sample: Mapping[str, Any], *, dataset_id: str, view: str | None, clahe_applied: bool) -> dict[str, Any]:
    """Build the metadata payload for HarmonisedCXR without copying raw image pixels."""

    meta = {key: value for key, value in sample.items() if key != "image"}
    meta["dataset_id"] = dataset_id
    meta["scanner_domain"] = sample.get("scanner_domain", dataset_id)
    meta["clahe_applied"] = clahe_applied
    meta["qc_passed"] = True
    if view is not None:
        meta["view"] = view
    return meta


def harmonise_sample(
    sample: Mapping[str, Any],
    *,
    apply_clahe: bool | None = None,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
) -> HarmonisedCXR:
    """Run baseline Component 0 QC and normalization on one dataset sample."""

    if "image" not in sample:
        raise KeyError("Missing required sample key: 'image'.")
    if "dataset_id" not in sample:
        raise KeyError("Missing required sample key: 'dataset_id'.")

    dataset_id = canonicalise_dataset_id(str(sample["dataset_id"]))
    validated_view = validate_view(sample.get("view"))

    grayscale = to_grayscale_tensor(sample["image"])
    validate_image_shape(grayscale)

    normalized = normalise_by_dataset(grayscale, dataset_id)

    x_1024_base = make_x1024(normalized)
    x_224 = make_x224_txv(normalized)
    x_3ch = x_1024_base.repeat(3, 1, 1)

    clahe_enabled = resolve_clahe_setting(dataset_id, apply_clahe)
    x_1024 = x_1024_base
    if clahe_enabled:
        x_1024 = apply_clahe_x1024(
            x_1024_base,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )

    meta = build_meta(
        sample,
        dataset_id=dataset_id,
        view=validated_view,
        clahe_applied=clahe_enabled,
    )

    return HarmonisedCXR(
        x_1024=x_1024.to(dtype=torch.float32),
        x_224=x_224.to(dtype=torch.float32),
        x_3ch=x_3ch.to(dtype=torch.float32),
        meta=meta,
    )
