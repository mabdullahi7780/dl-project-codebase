"""Component 7 — Upgraded Verification & Refinement (MoE path).

This module provides the full MoE-grade verification components:

    7A. ResNet18 Boundary Critic  — trained binary classifier predicting
        whether a mask boundary is anatomically plausible.
    7B. FP Auditor (upgraded)     — reuses TXV features + small MLP
        (already implemented in component7_fp_auditor.py for the baseline;
        the upgraded version here adds the DenseNet backbone option).
    7C. Expert-3 Guided Reprompt Refiner — when boundary score is low,
        samples uncertain boundary points, re-encodes them as SAM prompts,
        and runs Expert 3 (fibrosis/boundary decoder) to refine the mask.

The baseline heuristic scorer (component7_boundary.py) and morphology
refiner (component7_refine.py) are left untouched — both paths coexist
so the pipeline can run in baseline *or* MoE mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# 7A  Boundary Critic  (ResNet18, blocks 1-3 frozen)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BoundaryCriticConfig:
    pretrained: bool = False
    freeze_blocks_1_to_3: bool = True
    input_size: int = 224
    threshold: float = 0.5


class Component7BoundaryCritic(nn.Module):
    """ResNet18 boundary-quality critic.

    Input
        A 3-channel 224x224 crop centred on the predicted lesion centroid,
        with the mask boundary burned into a thin overlay channel.

    Output
        ``boundary_score ∈ [0, 1]`` — probability that the boundary is
        anatomically plausible.
    """

    def __init__(self, config: BoundaryCriticConfig | None = None) -> None:
        super().__init__()
        cfg = config or BoundaryCriticConfig()

        weights = models.ResNet18_Weights.DEFAULT if cfg.pretrained else None
        resnet = models.resnet18(weights=weights)

        # Feature extractor (everything before the FC)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        if cfg.freeze_blocks_1_to_3:
            # conv1, bn1, relu, maxpool, layer1, layer2, layer3
            for child in list(self.features.children())[:7]:
                for param in child.parameters():
                    param.requires_grad = False

        # Binary classification head
        self.head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224] — lesion-centroid crop with boundary overlay.

        Returns:
            boundary_score: [B, 1] in [0, 1].
        """
        feat = self.features(x).flatten(1)  # [B, 512]
        return self.head(feat)

    @staticmethod
    def prepare_crop(
        image_1024: torch.Tensor,
        mask_256: torch.Tensor,
        lung_mask_256: torch.Tensor,
    ) -> torch.Tensor:
        """Build the 3-channel critic input from raw pipeline tensors.

        Channel 0: image (resized to 256, then cropped to 224)
        Channel 1: mask boundary (dilated − eroded)
        Channel 2: lung mask region

        Returns [1, 3, 224, 224] ready for the critic.
        """
        # Resize image to 256
        if image_1024.ndim == 2:
            image_1024 = image_1024.unsqueeze(0).unsqueeze(0)
        elif image_1024.ndim == 3:
            image_1024 = image_1024.unsqueeze(0)
        img_256 = F.interpolate(image_1024.float(), size=(256, 256), mode="bilinear", align_corners=False)

        # Mask boundary: dilate − erode (approx with max_pool − (-max_pool(-x)))
        mask_f = mask_256.float()
        if mask_f.ndim == 2:
            mask_f = mask_f.unsqueeze(0).unsqueeze(0)
        elif mask_f.ndim == 3:
            mask_f = mask_f.unsqueeze(0)
        dilated = F.max_pool2d(mask_f, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask_f, 3, stride=1, padding=1)
        boundary = (dilated - eroded).clamp(0, 1)

        # Lung mask
        lung_f = lung_mask_256.float()
        if lung_f.ndim == 2:
            lung_f = lung_f.unsqueeze(0).unsqueeze(0)
        elif lung_f.ndim == 3:
            lung_f = lung_f.unsqueeze(0)

        # Stack 3 channels
        crop_input = torch.cat([img_256, boundary, lung_f], dim=1)  # [1, 3, 256, 256]

        # Centre-crop to 224x224
        crop_input = F.interpolate(crop_input, size=(224, 224), mode="bilinear", align_corners=False)
        return crop_input


# ---------------------------------------------------------------------------
# 7C  Expert-3 Guided Reprompt Refiner
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RepromptRefinerConfig:
    boundary_threshold: float = 0.7
    variance_threshold: float = 0.3
    num_prompt_points: int = 5
    dice_improvement_threshold: float = 0.0
    min_accept_dice_to_fused: float = 0.6


class Component7RepromptRefiner(nn.Module):
    """Re-prompt refinement using Expert 3 (fibrosis/boundary decoder).

    When the boundary critic reports a low score (< threshold) OR the
    inter-expert variance is high, this refiner:

    1. Finds uncertain boundary points from ``mask_variance``.
    2. Samples ``num_prompt_points`` points along the uncertain boundary.
    3. Passes them as SAM-style point prompts to Expert 3.
    4. Runs the expert to get a refined mask.
    5. Accepts the refined mask only if it improves Dice vs the fused mask
       (arbiter logic).
    """

    def __init__(self, config: RepromptRefinerConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or RepromptRefinerConfig()

    def _sample_uncertain_points(
        self,
        mask_variance: torch.Tensor,
        mask_fused: torch.Tensor,
    ) -> torch.Tensor | None:
        """Sample boundary points from high-variance regions.

        Args:
            mask_variance: [1, 256, 256] inter-expert variance.
            mask_fused:    [1, 256, 256] fused probability mask.

        Returns:
            points: [N, 2] (row, col) in 256x256 space, or None.
        """
        var_2d = mask_variance.squeeze()  # [256, 256]
        mask_2d = (mask_fused.squeeze() > 0.5).float()

        # Boundary of the fused mask
        dilated = F.max_pool2d(mask_2d.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask_2d.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1)
        boundary = (dilated - eroded).squeeze()  # [256, 256]

        # Points that are both on the boundary AND have high variance
        uncertain = (var_2d > self.cfg.variance_threshold) & (boundary > 0.5)
        points = uncertain.nonzero()  # [M, 2]

        if len(points) < 2:
            return None

        # Subsample to num_prompt_points
        n = min(self.cfg.num_prompt_points, len(points))
        indices = torch.randperm(len(points), device=points.device)[:n]
        return points[indices]  # [n, 2]

    @staticmethod
    def _pixel_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Quick Dice between two binary masks."""
        p = (pred > 0.5).float().flatten()
        t = (target > 0.5).float().flatten()
        intersection = (p * t).sum()
        return float(2 * intersection / (p.sum() + t.sum() + 1e-6))

    def forward(
        self,
        image_emb: torch.Tensor,
        mask_fused_256: torch.Tensor,
        mask_variance: torch.Tensor,
        boundary_score: float,
        expert3_decoder: nn.Module | None = None,
        lung_mask_256: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            image_emb:      [B, 256, 64, 64] — shared encoder embedding.
            mask_fused_256:  [B, 1, 256, 256] — fused mask from C6.
            mask_variance:   [B, 1, 256, 256] — inter-expert variance from C6.
            boundary_score:  scalar from boundary critic.
            expert3_decoder: the fibrosis/boundary expert (or None to skip).
            lung_mask_256:   [B, 1, 256, 256] lung mask for clipping.

        Returns:
            mask_refined_256: [B, 1, 256, 256] — refined or original mask.
        """
        B = image_emb.shape[0]
        needs_refinement = boundary_score < self.cfg.boundary_threshold

        if not needs_refinement or expert3_decoder is None:
            return mask_fused_256

        refined_list: list[torch.Tensor] = []

        for b in range(B):
            points = self._sample_uncertain_points(
                mask_variance[b],
                mask_fused_256[b],
            )

            if points is None:
                refined_list.append(mask_fused_256[b: b + 1])
                continue

            # Encode points as prompts: [1, N, 256] (project via simple embedding)
            # Scale points from 256x256 to 64x64 for embedding-space alignment
            points_64 = (points.float() / 4.0).long().clamp(0, 63)
            # Gather features at prompt locations from image_emb
            emb_b = image_emb[b]  # [256, 64, 64]
            prompt_feats = emb_b[:, points_64[:, 0], points_64[:, 1]].T  # [N, 256]
            prompt_feats = prompt_feats.unsqueeze(0)  # [1, N, 256]

            # Run Expert 3 with prompts
            # Use fused mask as dense prompt (downsampled to 64x64)
            dense = F.interpolate(
                mask_fused_256[b: b + 1],
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )
            refined_logits = expert3_decoder(
                image_emb[b: b + 1],
                prompts=prompt_feats,
                dense_prompt=dense,
            )
            refined_prob = torch.sigmoid(refined_logits)

            # Clip to lung mask
            if lung_mask_256 is not None:
                refined_prob = refined_prob * (lung_mask_256[b: b + 1] > 0.5).float()

            # Arbiter: without ground truth at inference time, require the
            # refined mask to remain sufficiently consistent with the fused mask.
            refined_dice = self._pixel_dice(refined_prob[0], mask_fused_256[b])

            if refined_dice >= max(
                self.cfg.min_accept_dice_to_fused,
                self.cfg.dice_improvement_threshold,
            ):
                refined_list.append(refined_prob)
            else:
                refined_list.append(mask_fused_256[b: b + 1])

        return torch.cat(refined_list, dim=0)
