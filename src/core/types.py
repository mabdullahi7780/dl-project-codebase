from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class HarmonisedCXR:
    """Standardized Component 0 output contract."""

    x_1024: torch.Tensor
    x_224: torch.Tensor
    x_3ch: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BaselineInferenceBundle:
    """Shared end-to-end inference state for later baseline components."""

    harmonised: HarmonisedCXR
    img_emb: torch.Tensor | None = None
    dom_logits: torch.Tensor | None = None
    domain_ctx: torch.Tensor | None = None
    pathology_logits: torch.Tensor | None = None
    lung_mask_256: torch.Tensor | None = None
    lung_mask_1024: torch.Tensor | None = None
    lesion_mask_coarse_256: torch.Tensor | None = None
    lesion_mask_refined_256: torch.Tensor | None = None
    lesion_mask_refined_1024: torch.Tensor | None = None
    boundary_score: float | None = None
    fp_prob: float | None = None
    alp: float | None = None
    cavity_flag: int | None = None
    timika_score: float | None = None
    severity: str | None = None
    evidence_json: dict[str, Any] | None = None
    report_text: str | None = None

    # --- MoE-specific fields (None when running the baseline path) ---
    pipeline_mode: str = "baseline"  # "baseline" | "moe"
    routing_weights: torch.Tensor | None = None      # [1, K] expert weights from C3
    expert_masks_256: list[torch.Tensor] | None = None  # K × [1, 1, 256, 256] per-expert probs
    mask_fused_256: torch.Tensor | None = None        # [1, 1, 256, 256] fused mask from C6
    mask_variance_256: torch.Tensor | None = None     # [1, 1, 256, 256] inter-expert variance
    cavity_mask_256: torch.Tensor | None = None       # [1, 1, 256, 256] Expert 2 cavity mask
