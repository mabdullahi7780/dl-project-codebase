from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.morphology import binary_erode, connected_component_stats


@dataclass(slots=True)
class BoundaryScoreBreakdown:
    boundary_score: float
    lesion_fraction: float
    spill_fraction: float
    n_components: int
    compactness: float
    edge_alignment: float


def _to_numpy_mask(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    return (mask > 0.5).astype(np.uint8)


def _resize_x1024_to_256(x_1024: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x_1024, np.ndarray):
        tensor = torch.from_numpy(x_1024).float()
    else:
        tensor = x_1024.detach().cpu().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.unsqueeze(0)
    else:
        raise ValueError(f"Expected x_1024 [1, 1024, 1024] or [1024, 1024], got {tuple(tensor.shape)}.")
    resized = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


def score_boundary_quality(
    lesion_mask_256: torch.Tensor | np.ndarray,
    lung_mask_256: torch.Tensor | np.ndarray,
    x_1024: torch.Tensor | np.ndarray,
) -> BoundaryScoreBreakdown:
    lesion = _to_numpy_mask(lesion_mask_256)
    lung = _to_numpy_mask(lung_mask_256)
    image = _resize_x1024_to_256(x_1024)

    lung_area = int(lung.sum())
    lesion_in_lung = lesion & lung
    lesion_area = int(lesion_in_lung.sum())
    lesion_fraction = 0.0 if lung_area == 0 else lesion_area / float(lung_area)

    spill = lesion & (~lung.astype(bool))
    spill_fraction = 0.0 if lesion.sum() == 0 else float(spill.sum()) / float(max(int(lesion.sum()), 1))

    component_sizes = connected_component_stats(lesion_in_lung.astype(bool))
    n_components = len(component_sizes)

    boundary = lesion_in_lung.astype(bool) ^ binary_erode(lesion_in_lung.astype(bool))
    grad_y, grad_x = np.gradient(image)
    grad_mag = np.sqrt((grad_x ** 2) + (grad_y ** 2))
    boundary_grad = float(grad_mag[boundary].mean()) if boundary.any() else 0.0
    lung_grad = float(grad_mag[lung.astype(bool)].mean()) if lung.any() else 1e-6
    edge_alignment = float(np.clip(boundary_grad / (lung_grad + 1e-6), 0.0, 2.0) / 2.0)

    perimeter = int(boundary.sum())
    if perimeter == 0 or lesion_area == 0:
        compactness = 0.0
    else:
        compactness = float(np.clip((4.0 * np.pi * lesion_area) / ((perimeter ** 2) + 1e-6), 0.0, 1.0))

    component_score = 1.0 if n_components <= 3 else max(0.0, 1.0 - ((n_components - 3) * 0.15))
    area_score = 1.0 if 0.001 <= lesion_fraction <= 0.6 else 0.35
    spill_score = max(0.0, 1.0 - spill_fraction)

    boundary_score = float(
        np.clip(
            (0.25 * edge_alignment) + (0.25 * compactness) + (0.2 * component_score) + (0.2 * spill_score) + (0.1 * area_score),
            0.0,
            1.0,
        )
    )

    return BoundaryScoreBreakdown(
        boundary_score=boundary_score,
        lesion_fraction=float(lesion_fraction),
        spill_fraction=float(spill_fraction),
        n_components=n_components,
        compactness=float(compactness),
        edge_alignment=float(edge_alignment),
    )
