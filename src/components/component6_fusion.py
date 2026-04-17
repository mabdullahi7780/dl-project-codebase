"""Component 6 — Expert Fusion.

Combines mask logits from all active experts (C5) using the routing
weights from C3 into a single fused lesion mask.

The fusion operates in **logit space** (before sigmoid) so that expert
confidences compose linearly.  This is more principled than averaging
probabilities, which suppresses high-confidence predictions.

Shape contract
--------------
    expert_logits:   list of K × [B, 1, 256, 256]  — raw logits from experts
    routing_weights: [B, K]                         — from Component 3
    output:          FusionOutput dataclass with:
                       mask_fused_logits  [B, 1, 256, 256]
                       mask_fused_256     [B, 1, 256, 256]  (after sigmoid)
                       mask_variance      [B, 1, 256, 256]  (inter-expert)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(slots=True)
class FusionOutput:
    """Outputs from expert fusion."""

    mask_fused_logits: torch.Tensor   # [B, 1, 256, 256] raw fused logits
    mask_fused_256: torch.Tensor      # [B, 1, 256, 256] sigmoid prob map
    mask_variance: torch.Tensor       # [B, 1, 256, 256] inter-expert disagreement
    expert_masks_256: list[torch.Tensor]  # per-expert sigmoid masks (for C7/C8)


@dataclass(slots=True)
class FusionConfig:
    num_experts: int = 4
    fusion_mode: str = "weighted_logit"   # "weighted_logit" | "weighted_prob"
    learnable_bias: bool = False


class Component6ExpertFusion(nn.Module):
    """Logit-space weighted fusion of expert mask outputs.

    Given K expert logit maps and K routing weights, the fused logits are::

        fused_logit[b, :, h, w] = sum_k( w[b,k] * logit_k[b, :, h, w] )

    where ``w`` are the softmax routing weights from Component 3.

    The module also computes **mask variance**, a pixel-wise measure of
    expert disagreement.  Component 7 uses this to decide whether
    boundary refinement is needed: high variance → uncertain boundary.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        super().__init__()
        cfg = config or FusionConfig()
        self.num_experts = cfg.num_experts
        self.fusion_mode = cfg.fusion_mode

        if cfg.learnable_bias:
            self.expert_bias = nn.Parameter(torch.zeros(cfg.num_experts))
        else:
            self.expert_bias = None

    def forward(
        self,
        expert_logits: list[torch.Tensor],
        routing_weights: torch.Tensor,
    ) -> FusionOutput:
        """
        Args:
            expert_logits:   list of K tensors, each [B, 1, 256, 256].
            routing_weights: [B, K] from Component 3 routing gate.

        Returns:
            FusionOutput with fused mask and inter-expert variance.
        """
        K = len(expert_logits)
        if K == 0:
            raise ValueError("expert_logits must contain at least one tensor.")

        B = expert_logits[0].shape[0]
        device = expert_logits[0].device

        if routing_weights.shape != (B, K):
            raise ValueError(
                f"routing_weights shape {tuple(routing_weights.shape)} does not match "
                f"batch={B}, num_experts={K}."
            )

        # Stack: [B, K, 1, 256, 256]
        stacked_logits = torch.stack(expert_logits, dim=1)

        # Add optional learnable per-expert bias
        if self.expert_bias is not None:
            bias = self.expert_bias[:K].view(1, K, 1, 1, 1)
            stacked_logits = stacked_logits + bias

        # Routing weights: [B, K] → [B, K, 1, 1, 1] for broadcasting
        w = routing_weights.view(B, K, 1, 1, 1)

        if self.fusion_mode == "weighted_logit":
            fused_logits = (w * stacked_logits).sum(dim=1)  # [B, 1, 256, 256]
        elif self.fusion_mode == "weighted_prob":
            probs = torch.sigmoid(stacked_logits)
            fused_prob = (w * probs).sum(dim=1)
            fused_logits = torch.logit(fused_prob.clamp(1e-6, 1 - 1e-6))
        else:
            raise ValueError(f"Unknown fusion_mode {self.fusion_mode!r}")

        fused_mask = torch.sigmoid(fused_logits)

        # Inter-expert variance (computed on probabilities for interpretability)
        expert_probs = torch.sigmoid(stacked_logits)  # [B, K, 1, 256, 256]
        expert_masks_list = [expert_probs[:, k] for k in range(K)]

        # Weighted variance: Var = E[X^2] - E[X]^2
        mean_prob = (w * expert_probs).sum(dim=1)       # [B, 1, 256, 256]
        mean_sq = (w * expert_probs ** 2).sum(dim=1)    # [B, 1, 256, 256]
        variance = (mean_sq - mean_prob ** 2).clamp_min(0.0)

        return FusionOutput(
            mask_fused_logits=fused_logits,
            mask_fused_256=fused_mask,
            mask_variance=variance,
            expert_masks_256=expert_masks_list,
        )
