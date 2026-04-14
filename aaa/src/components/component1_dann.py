from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.component1_encoder import Component1Encoder


class _GradientReverse(torch.autograd.Function):
    """GRL: forward is identity, backward multiplies grad by -lambda."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GradientReverse.apply(x, lambda_)


def compute_dann_lambda(epoch: int, ramp_epochs: int = 10, max_lambda: float = 1.0) -> float:
    if ramp_epochs <= 0:
        return float(max_lambda)
    progress = min(max(epoch, 0), ramp_epochs) / float(ramp_epochs)
    return float(max_lambda) * progress


@dataclass(slots=True)
class DANNHeadConfig:
    input_dim: int = 256
    hidden_dim: int = 128
    num_domains: int = 4
    dropout: float = 0.3


class DANNHead(nn.Module):
    """4-way domain classifier with a gradient reversal layer."""

    def __init__(self, config: DANNHeadConfig | None = None) -> None:
        super().__init__()
        cfg = config or DANNHeadConfig()
        self.config = cfg
        self.norm = nn.LayerNorm(cfg.input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_domains),
        )

    def forward(self, pooled: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        if pooled.ndim != 2:
            raise ValueError(
                f"DANNHead expects pooled features [B, C]; got shape {tuple(pooled.shape)}."
            )
        reversed_ = gradient_reverse(pooled, lambda_)
        return self.mlp(self.norm(reversed_))


def pool_image_embedding(img_emb: torch.Tensor) -> torch.Tensor:
    if img_emb.ndim != 4:
        raise ValueError(
            f"Expected SAM image embedding [B, C, H, W]; got shape {tuple(img_emb.shape)}."
        )
    return img_emb.mean(dim=(2, 3))


class Component1DANNModel(nn.Module):
    """Shared image encoder + DANN domain head."""

    def __init__(self, encoder: Component1Encoder, head: DANNHead | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head or DANNHead()

    def forward(self, x_3ch: torch.Tensor, *, lambda_: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        img_emb = self.encoder(x_3ch)
        dom_logits = self.head(pool_image_embedding(img_emb), lambda_=lambda_)
        return img_emb, dom_logits


def domain_classification_loss(dom_logits: torch.Tensor, domain_targets: torch.Tensor) -> torch.Tensor:
    if dom_logits.ndim != 2:
        raise ValueError(f"Expected dom_logits [B, num_domains], got {tuple(dom_logits.shape)}.")
    if domain_targets.ndim != 1:
        raise ValueError(f"Expected domain_targets [B], got {tuple(domain_targets.shape)}.")
    if dom_logits.shape[0] != domain_targets.shape[0]:
        raise ValueError(
            "Batch mismatch between dom_logits and domain_targets: "
            f"{dom_logits.shape[0]} vs {domain_targets.shape[0]}."
        )
    return F.cross_entropy(dom_logits, domain_targets)
