from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.components.baseline_lesion_proposer import SUSPICIOUS_CLASS_SET
from src.components.component2_txv import TXV_CLASS_NAMES
from src.utils.morphology import connected_component_stats


@dataclass(slots=True)
class FPAuditBreakdown:
    fp_probability: float
    lesion_fraction: float
    suspicious_probability: float
    n_components: int
    spill_fraction: float


def estimate_fp_probability(
    lesion_mask_256: torch.Tensor,
    lung_mask_256: torch.Tensor,
    pathology_logits: torch.Tensor | None = None,
    *,
    class_names: tuple[str, ...] = TXV_CLASS_NAMES,
) -> FPAuditBreakdown:
    lesion = (lesion_mask_256.detach().cpu().numpy() > 0.5).astype(np.uint8)
    lung = (lung_mask_256.detach().cpu().numpy() > 0.5).astype(np.uint8)

    if lesion.ndim == 3 and lesion.shape[0] == 1:
        lesion = lesion[0]
    if lung.ndim == 3 and lung.shape[0] == 1:
        lung = lung[0]

    lesion_in_lung = lesion & lung
    lung_area = int(lung.sum())
    lesion_fraction = 0.0 if lung_area == 0 else float(lesion_in_lung.sum()) / float(lung_area)

    spill = lesion & (~lung.astype(bool))
    spill_fraction = 0.0 if lesion.sum() == 0 else float(spill.sum()) / float(max(int(lesion.sum()), 1))

    component_sizes = connected_component_stats(lesion_in_lung.astype(bool))
    n_components = len(component_sizes)

    suspicious_probability = 0.5
    if pathology_logits is not None:
        probs = torch.sigmoid(pathology_logits.detach().cpu())
        suspicious_indices = [i for i, name in enumerate(class_names) if name in SUSPICIOUS_CLASS_SET]
        if suspicious_indices:
            suspicious_probability = float(probs[suspicious_indices].mean().item())

    tiny_penalty = 1.0 if 0.0 < lesion_fraction < 0.002 else 0.0
    noisy_components_penalty = min(max(n_components - 3, 0) / 5.0, 1.0)
    unsupported_area_penalty = 0.8 if lesion_fraction > 0.7 else 0.0

    fp_probability = float(
        np.clip(
            (0.45 * (1.0 - suspicious_probability))
            + (0.2 * tiny_penalty)
            + (0.15 * noisy_components_penalty)
            + (0.1 * spill_fraction)
            + (0.1 * unsupported_area_penalty),
            0.0,
            1.0,
        )
    )

    return FPAuditBreakdown(
        fp_probability=fp_probability,
        lesion_fraction=float(lesion_fraction),
        suspicious_probability=float(suspicious_probability),
        n_components=n_components,
        spill_fraction=float(spill_fraction),
    )
