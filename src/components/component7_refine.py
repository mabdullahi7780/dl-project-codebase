from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.utils.morphology import fill_binary_holes, postprocess_binary_mask


@dataclass(slots=True)
class BaselineRefineConfig:
    min_area_px: int = 48
    opening_iters: int = 1
    closing_iters: int = 1
    weak_boundary_threshold: float = 0.45
    suppress_fp_threshold: float = 0.85
    caution_fp_threshold: float = 0.65


def refine_mask(
    lesion_mask_coarse_256: torch.Tensor,
    lung_mask_256: torch.Tensor,
    x_1024: torch.Tensor,
    boundary_score: float,
    fp_prob: float,
    config: BaselineRefineConfig | None = None,
) -> torch.Tensor:
    _ = x_1024
    cfg = config or BaselineRefineConfig()
    lesion = (lesion_mask_coarse_256.detach().cpu().numpy() > 0.5)
    lung = (lung_mask_256.detach().cpu().numpy() > 0.5)

    if lesion.ndim == 3 and lesion.shape[0] == 1:
        lesion = lesion[0]
    if lung.ndim == 3 and lung.shape[0] == 1:
        lung = lung[0]

    refined = lesion & lung
    refined = postprocess_binary_mask(
        refined,
        min_area=cfg.min_area_px,
        opening_iters=cfg.opening_iters,
        closing_iters=cfg.closing_iters,
    )
    refined = fill_binary_holes(refined)

    if boundary_score < cfg.weak_boundary_threshold:
        refined = postprocess_binary_mask(
            refined,
            min_area=max(cfg.min_area_px, 96),
            opening_iters=cfg.opening_iters + 1,
            closing_iters=cfg.closing_iters + 1,
        )

    if fp_prob >= cfg.suppress_fp_threshold:
        refined = np.zeros_like(refined, dtype=bool)
    elif fp_prob >= cfg.caution_fp_threshold:
        refined = postprocess_binary_mask(
            refined,
            min_area=max(cfg.min_area_px, 128),
            opening_iters=cfg.opening_iters + 1,
            closing_iters=cfg.closing_iters,
        )

    return torch.from_numpy(refined.astype(np.float32)).unsqueeze(0).to(lesion_mask_coarse_256.device)
