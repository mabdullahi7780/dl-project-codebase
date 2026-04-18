"""Component 3 — Routing Gate.

Produces soft expert weights from the shared image embedding so the
fusion module (C6) knows how much to trust each expert's mask for a
given image.

Shape contract
--------------
    input:  img_emb        [B, 256, 64, 64]  — from Component 1
    output: expert_weights  [B, K]            — softmax weights, K = num_experts

The gate is deliberately small (~50 K params) because the embedding
already carries rich spatial information — the gate only needs to
learn *which pathology pattern dominates* in this image.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class RoutingGateConfig:
    in_channels: int = 256
    hidden_dim: int = 128
    num_experts: int = 4
    use_domain_ctx: bool = False
    domain_ctx_dim: int = 256
    dropout: float = 0.1
    temperature: float = 1.0
    top_k: int | None = None  # None = use all experts


class Component3RoutingGate(nn.Module):
    """Soft routing gate: GAP → MLP → softmax expert weights.

    The gate supports two inference modes:

    * **Dense** (default, ``top_k=None``): all experts contribute,
      weighted by softmax.  This is standard soft-MoE.
    * **Sparse** (``top_k=K``): only the top-K experts have non-zero
      weight; the rest are zeroed.  Saves compute at inference time
      and encourages expert specialisation during training.
    """

    def __init__(self, config: RoutingGateConfig | None = None) -> None:
        super().__init__()
        cfg = config or RoutingGateConfig()
        self.num_experts = cfg.num_experts
        self.use_domain_ctx = cfg.use_domain_ctx
        self.temperature = cfg.temperature
        self.top_k = cfg.top_k

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.domain_proj = (
            nn.Linear(cfg.domain_ctx_dim, cfg.in_channels)
            if cfg.use_domain_ctx
            else None
        )

        self.gate = nn.Sequential(
            nn.Linear(cfg.in_channels, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_experts),
        )

    def forward(
        self,
        img_emb: torch.Tensor,
        domain_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img_emb: [B, 256, 64, 64] from Component 1 encoder.

        Returns:
            expert_weights: [B, K] — sum-to-one soft weights.
        """
        if img_emb.ndim != 4:
            raise ValueError(
                f"RoutingGate expects img_emb [B, C, H, W], got ndim={img_emb.ndim}."
            )

        pooled = self.pool(img_emb).flatten(1)  # [B, 256]
        if self.domain_proj is not None:
            if domain_ctx is None:
                raise ValueError("RoutingGate configured with use_domain_ctx=True but no domain_ctx was provided.")
            if domain_ctx.ndim != 2:
                raise ValueError(
                    f"RoutingGate expects domain_ctx [B, D], got ndim={domain_ctx.ndim}."
                )
            if domain_ctx.shape[0] != pooled.shape[0]:
                raise ValueError(
                    f"RoutingGate batch mismatch: pooled batch={pooled.shape[0]} vs domain_ctx batch={domain_ctx.shape[0]}."
                )
            pooled = pooled + self.domain_proj(domain_ctx)
        logits = self.gate(pooled)               # [B, K]

        # Temperature-scaled softmax
        weights = F.softmax(logits / self.temperature, dim=-1)  # [B, K]

        # Optional top-k sparsity
        if self.top_k is not None and self.top_k < self.num_experts:
            topk_vals, topk_idx = weights.topk(self.top_k, dim=-1)
            sparse = torch.zeros_like(weights)
            sparse.scatter_(1, topk_idx, topk_vals)
            # Re-normalise so weights still sum to 1
            weights = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)

        return weights


# ---------------------------------------------------------------------------
# Load-balancing auxiliary loss (for training only)
# ---------------------------------------------------------------------------

def routing_load_balance_loss(
    expert_weights: torch.Tensor,
    *,
    num_experts: int | None = None,
) -> torch.Tensor:
    """Encourage the gate to distribute work evenly across experts.

    This is the standard Switch Transformer auxiliary loss: the product
    of the fraction of tokens routed to each expert and the mean gate
    probability for that expert, summed over experts and multiplied by
    ``num_experts`` so the ideal uniform value equals 1.

    Args:
        expert_weights: [B, K] softmax weights from the gate.
        num_experts:    K (inferred from tensor if omitted).

    Returns:
        Scalar loss ≥ 1 (1 when perfectly balanced).
    """
    K = num_experts or expert_weights.shape[-1]
    # f_i = fraction of batch where expert i has the highest weight
    assignments = expert_weights.argmax(dim=-1)  # [B]
    f = torch.zeros(K, device=expert_weights.device)
    for i in range(K):
        f[i] = (assignments == i).float().mean()
    # p_i = mean probability assigned to expert i across the batch
    p = expert_weights.mean(dim=0)  # [K]
    return (K * (f * p).sum())
