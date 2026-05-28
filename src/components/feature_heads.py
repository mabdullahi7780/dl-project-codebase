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


# ── Rung 5: severity-specialised MoE + uncertainty critic ─────────────────────
# K regression experts, softly gated. Unlike the failed DA-MoE there is NO shared
# trainable trunk (features are frozen/cached) and NO gradient-reversal — so there
# is nothing to collapse. The critic predicts per-sample reliability (regresses the
# absolute residual), used later by conformal/ensemble weighting.


class MoERegressionHead(nn.Module):
    """feature -> sigmoid scalar in [0,1] via K softly-gated experts."""

    def __init__(self, in_dim: int, hidden: int = 256, n_experts: int = 4,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            for _ in range(n_experts)
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x: torch.Tensor, *, return_gate: bool = False):
        ex = torch.cat([e(x) for e in self.experts], dim=1)   # [B, K] raw expert outputs
        w = torch.softmax(self.gate(x), dim=1)                # [B, K] gate weights
        mixed = torch.sigmoid((ex * w).sum(dim=1))            # [B] in [0,1]
        if return_gate:
            return mixed, w
        return mixed


class CriticHead(nn.Module):
    """feature -> predicted reliability in [0,1] (1 = trustworthy, 0 = unreliable).

    Trained to regress ``1 - |error|`` of the main head, so high output means the
    main prediction is expected to be accurate. Used to weight ensemble members
    and flag out-of-distribution test samples.
    """

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(1)  # [B] in [0,1]


class SpatialCavityHead(nn.Module):
    """Per-patch cavity scoring + attention pooling over a frozen patch grid.

    Cavities are *localized* lesions occupying a small fraction of the lung field.
    A global CLS embedding averages over the whole image and washes the signal out
    (RAD-DINO Romania cavity AUC = 0.675 with the global head). This head scores
    each patch independently then pools the patch-level evidence via a learned
    attention weight, so a single high-score patch can dominate the prediction —
    matching the clinical fact that finding *one* cavity suffices to label the
    image cavity-positive.

    Input  : [B, P, D] patch tokens (e.g. RAD-DINO 7x7 grid pooled).
    Output : [B, 2] cavity-vs-no-cavity logits (drop-in for ClassifierHead).
    """

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 2),                                # per-patch logits
        )
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),                            # per-patch attention logit
        )

    def forward(self, x: torch.Tensor, *, return_attn: bool = False):
        # x: [B, P, D]
        per_patch = self.score(x)                                 # [B, P, 2]
        attn = torch.softmax(self.attn(x).squeeze(-1), dim=1)     # [B, P]
        pooled = (per_patch * attn.unsqueeze(-1)).sum(dim=1)      # [B, 2]
        if return_attn:
            return pooled, attn
        return pooled


class SpatialALPHead(nn.Module):
    """A1-adapted: per-zone involvement over a patch grid -> ALP in [0,1].

    Input is a pooled patch grid ``[B, P, D]`` (P = grid cells, e.g. 49 for 7x7).
    A shared scorer predicts each cell's lesion involvement in [0,1]; ALP is the
    mean over cells (a differentiable stand-in for Kantipudi's lesion-area /
    lung-area ratio, using frozen foundation features instead of YOLO detections).
    """

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, *, return_map: bool = False):
        # x: [B, P, D]
        cell = torch.sigmoid(self.scorer(x)).squeeze(-1)   # [B, P] per-cell involvement
        alp = cell.mean(dim=1)                             # [B] in [0,1]
        if return_map:
            return alp, cell
        return alp
