"""Light heads trained on cached frozen-backbone features (linear-probe style).

For Rung 1 the "model" is just one of these MLP heads on top of cached RAD-DINO
embeddings. Later rungs replace ``RegressionHead`` with the severity-MoE + retrieval
+ critic, but keep the same feature-in / Timika-out contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """feature -> sigmoid scalar in [0, 1] (x100 = ALP%, or x140 = Timika)."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(1)  # [B] in [0,1]


class ClassifierHead(nn.Module):
    """feature -> 2-logit cavity classifier."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, 2] logits
