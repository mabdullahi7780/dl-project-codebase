"""Paper-faithful Kantipudi et al. (JIIM 2024) approach A2 models.

A2 uses TWO fully independent DenseNet121 networks (NOT a shared-backbone
multi-task model like ``component_alp_moe``):

  - ``ALPRegressor``     : DenseNet121 + a single neuron + sigmoid, output in
                           [0, 1] = fraction of lung affected (paper p.2176).
                           Trained with MSE loss.
  - ``CavityClassifier`` : DenseNet121 + a fully connected layer with two
                           outputs (paper p.2176). Trained with cross-entropy.

Both are initialised from ImageNet weights and consume lung-cropped 224x224
inputs. Keeping them separate is the whole point of strict replication: it
avoids the competing-gradient problem of a shared backbone and matches the
paper's training protocol exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121

FEATURE_DIM = 1024  # DenseNet121 pooled feature width


class _DenseNet121Features(nn.Module):
    """ImageNet DenseNet121 trunk -> global-average-pooled [B, 1024]."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = densenet121(weights=weights).features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = F.relu(f, inplace=True)
        return F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)


class ALPRegressor(nn.Module):
    """DenseNet121 + 1-neuron sigmoid head. Output: ALP in [0, 1] (x100 = %)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = _DenseNet121Features(pretrained)
        self.head = nn.Linear(FEATURE_DIM, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.backbone(x))).squeeze(1)  # [B] in [0,1]


class CavityClassifier(nn.Module):
    """DenseNet121 + 2-output FC head. Returns logits for cross-entropy."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = _DenseNet121Features(pretrained)
        self.head = nn.Linear(FEATURE_DIM, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))  # [B, 2] logits


class TimikaRegressor(nn.Module):
    """DenseNet121 + 1-neuron sigmoid head (Kantipudi A3: direct Timika).

    The paper passes the output through a sigmoid and multiplies by 140 to span
    the Timika range [0, 140]. To mirror ``ALPRegressor`` and keep training
    numerically stable, ``forward`` returns the sigmoid fraction in [0, 1]; the
    caller multiplies by 140 at prediction time and trains MSE against
    ``timika_true / 140``.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = _DenseNet121Features(pretrained)
        self.head = nn.Linear(FEATURE_DIM, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.backbone(x))).squeeze(1)  # [B] in [0,1]; x140 = Timika
