from __future__ import annotations

from dataclasses import dataclass

import torch

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
