from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.components.component8_metrics import compute_timika_score


@dataclass(slots=True)
class BaselineTimikaResult:
    ALP: float
    cavity_flag: int
    timika_score: float
    severity: str
    cavitation_confidence: str


def compute_baseline_timika(
    lesion_mask_refined_1024: torch.Tensor,
    lung_mask_1024: torch.Tensor,
) -> BaselineTimikaResult:
    metrics = compute_timika_score(
        lesion_mask_refined_1024,
        lung_mask_1024,
        None,
    )
    return BaselineTimikaResult(
        ALP=float(metrics["ALP"]),
        cavity_flag=0,
        timika_score=float(metrics["timika_score"]),
        severity=str(metrics["severity"]),
        cavitation_confidence="not-assessed-baseline",
    )


def compute_moe_timika(
    lesion_mask_refined_1024: torch.Tensor,
    lung_mask_1024: torch.Tensor,
    cavity_mask_256: torch.Tensor | None = None,
) -> BaselineTimikaResult:
    """Upgraded Timika computation for the MoE path.

    When ``cavity_mask_256`` is provided (from Expert 2, the cavity
    decoder), it is passed to ``compute_timika_score`` to derive a real
    cavity flag instead of defaulting to 0.

    Args:
        lesion_mask_refined_1024: [1, 1024, 1024] refined lesion mask.
        lung_mask_1024:           [1, 1024, 1024] lung mask.
        cavity_mask_256:          [1, 256, 256] Expert 2 cavity probability
                                  map (optional — ``None`` falls back to
                                  baseline behaviour).

    Returns:
        BaselineTimikaResult with real cavity detection when available.
    """
    if cavity_mask_256 is not None:
        # Expert 2 produces [1, 256, 256]; compute_timika_score expects
        # [256, 256] or [1, 256, 256] numpy/tensor.
        if cavity_mask_256.ndim == 4:
            cavity_mask_256 = cavity_mask_256.squeeze(0)
        metrics = compute_timika_score(
            lesion_mask_refined_1024,
            lung_mask_1024,
            cavity_mask_256,
        )
        confidence = "expert2-radiographic"
    else:
        metrics = compute_timika_score(
            lesion_mask_refined_1024,
            lung_mask_1024,
            None,
        )
        confidence = "not-assessed-baseline"

    return BaselineTimikaResult(
        ALP=float(metrics["ALP"]),
        cavity_flag=int(metrics["cavity_flag"]),
        timika_score=float(metrics["timika_score"]),
        severity=str(metrics["severity"]),
        cavitation_confidence=confidence,
    )
