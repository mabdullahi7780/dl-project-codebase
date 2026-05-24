"""DenseNet121 + Mixture-of-Experts ALP regressor (+ cavity + DANN heads).

Single unified model that covers the whole ablation ladder via flags:

- ``num_experts=1, use_dann=False`` -> Kantipudi A2 reproduction (Day 1 gate).
- ``num_experts=K``               -> MoE regression head (Day 2 contribution 1).
- ``use_dann=True``               -> country-adversarial features (Day 2 contribution 2).

ALP is a soft mixture of per-expert sigmoid predictions, so the output is always
in [0, 1] (multiply by 100 for the radiologist scale). Cavity is a shared-backbone
binary head. The DANN head reuses the project's gradient-reversal layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121

from src.components.component1_dann import gradient_reverse

FEATURE_DIM = 1024  # DenseNet121 pooled feature width


@dataclass(slots=True)
class ALPMoEConfig:
    num_experts: int = 1
    expert_hidden: int = 256
    gate_temperature: float = 1.0
    cavity_hidden: int = 256
    dropout: float = 0.2
    use_dann: bool = False
    num_countries: int = 5
    dann_hidden: int = 256
    pretrained: bool = True
    freeze_backbone: bool = False


@dataclass(slots=True)
class ALPMoEOutput:
    alp: torch.Tensor  # [B] in [0, 1]
    cavity_logit: torch.Tensor  # [B]
    features: torch.Tensor  # [B, 1024]
    gate_weights: torch.Tensor | None = None  # [B, K]
    expert_alps: torch.Tensor | None = None  # [B, K] in [0, 1]
    country_logit: torch.Tensor | None = None  # [B, num_countries]


class DenseNet121Backbone(nn.Module):
    """ImageNet-pretrained DenseNet121 feature extractor -> pooled [B, 1024]."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = densenet121(weights=weights)
        self.features = net.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
        return f


class ALPMoEHead(nn.Module):
    """K expert regressors + softmax gate. K=1 collapses to a single head."""

    def __init__(self, cfg: ALPMoEConfig) -> None:
        super().__init__()
        self.num_experts = cfg.num_experts
        self.gate_temperature = cfg.gate_temperature
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(FEATURE_DIM, cfg.expert_hidden),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.expert_hidden, 1),
            )
            for _ in range(self.num_experts)
        )
        self.gate = nn.Linear(FEATURE_DIM, self.num_experts) if self.num_experts > 1 else None

    def forward(
        self, f: torch.Tensor, *, uniform_gate: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        expert_logits = torch.cat([e(f) for e in self.experts], dim=1)  # [B, K]
        expert_alps = torch.sigmoid(expert_logits)  # [B, K] in [0, 1]

        if self.num_experts == 1:
            return expert_alps[:, 0], None, expert_alps

        if uniform_gate or self.gate is None:
            weights = torch.full_like(expert_alps, 1.0 / self.num_experts)
        else:
            weights = F.softmax(self.gate(f) / self.gate_temperature, dim=1)

        alp = (weights * expert_alps).sum(dim=1)  # [B]
        return alp, weights, expert_alps


class DANNCountryHead(nn.Module):
    """Country classifier behind a gradient reversal layer."""

    def __init__(self, cfg: ALPMoEConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(FEATURE_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM, cfg.dann_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dann_hidden, cfg.num_countries),
        )

    def forward(self, f: torch.Tensor, lambda_: float) -> torch.Tensor:
        return self.mlp(self.norm(gradient_reverse(f, lambda_)))


class ALPMoEModel(nn.Module):
    """Backbone + ALP MoE head + cavity head + optional DANN head."""

    def __init__(self, config: ALPMoEConfig | None = None) -> None:
        super().__init__()
        cfg = config or ALPMoEConfig()
        self.config = cfg

        self.backbone = DenseNet121Backbone(pretrained=cfg.pretrained)
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.alp_head = ALPMoEHead(cfg)
        self.cavity_head = nn.Sequential(
            nn.Linear(FEATURE_DIM, cfg.cavity_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.cavity_hidden, 1),
        )
        self.dann_head = DANNCountryHead(cfg) if cfg.use_dann else None

    def set_gate_trainable(self, trainable: bool) -> None:
        """Phase-1 (warm experts) vs Phase-2 (train gate) control for the MoE."""
        if self.alp_head.gate is not None:
            for p in self.alp_head.gate.parameters():
                p.requires_grad = trainable

    def forward(
        self,
        x: torch.Tensor,
        *,
        dann_lambda: float = 0.0,
        uniform_gate: bool = False,
    ) -> ALPMoEOutput:
        f = self.backbone(x)
        alp, gate_weights, expert_alps = self.alp_head(f, uniform_gate=uniform_gate)
        cavity_logit = self.cavity_head(f).squeeze(1)
        country_logit = (
            self.dann_head(f, dann_lambda) if self.dann_head is not None else None
        )
        return ALPMoEOutput(
            alp=alp,
            cavity_logit=cavity_logit,
            features=f,
            gate_weights=gate_weights,
            expert_alps=expert_alps,
            country_logit=country_logit,
        )


def load_balancing_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """Encourage the gate to use all experts (minimized at the uniform routing).

    Returns ``K * sum_k mean_b(w_bk)^2`` which equals 1.0 at perfectly uniform
    routing and grows as the gate collapses onto fewer experts.
    """
    importance = gate_weights.mean(dim=0)  # [K]
    return gate_weights.shape[1] * (importance * importance).sum()
