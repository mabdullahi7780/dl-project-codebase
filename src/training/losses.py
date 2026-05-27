"""Balanced-regression losses for imbalanced Timika regression.

The cross-country failure is a LABEL-shift problem: the held-out country (Moldova)
is sicker, sitting in a sparsely-sampled high-ALP region, so a plain-MSE regressor
collapses toward the training mean. Balanced MSE (Ren et al., CVPR 2022) corrects
the implicit training-label prior so sparse regions are not down-weighted.

``BMCLoss`` is the Batch-based Monte-Carlo variant — it needs NO prior knowledge
of the label distribution (it treats each target as a class over the batch's
targets, with a learnable noise scale). Add ``BMCLoss.parameters()`` to the
optimiser so ``noise_sigma`` is learned.

``lds_weights`` implements Label Distribution Smoothing (Yang et al., ICML 2021)
for an optional weighted-MSE variant in later rungs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BMCLoss(nn.Module):
    """Balanced MSE via Batch-based Monte-Carlo (Ren et al. 2022), 1-D targets."""

    def __init__(self, init_noise_sigma: float = 1.0) -> None:
        super().__init__()
        self.noise_sigma = nn.Parameter(torch.tensor(float(init_noise_sigma)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(-1, 1)
        target = target.reshape(-1, 1)
        noise_var = self.noise_sigma ** 2
        logits = -0.5 * (pred - target.T) ** 2 / noise_var          # [B, B]
        labels = torch.arange(pred.size(0), device=pred.device)
        loss = F.cross_entropy(logits, labels)
        return loss * (2 * noise_var).detach()                       # keep MSE scale


def lds_weights(targets: np.ndarray, *, n_bins: int = 20, sigma: float = 2.0,
                lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Per-sample weights = 1 / smoothed-label-density (LDS). Returns [N], mean 1."""
    from scipy.ndimage import gaussian_filter1d
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(targets, edges) - 1, 0, n_bins - 1)
    hist = np.bincount(idx, minlength=n_bins).astype(np.float64)
    smooth = gaussian_filter1d(hist, sigma=sigma, mode="constant")
    dens = np.maximum(smooth[idx], 1e-6)
    w = 1.0 / dens
    return (w / w.mean()).astype(np.float32)


def make_reg_loss(kind: str):
    """Return (loss_module_or_none, callable(pred,target)->loss)."""
    if kind == "mse":
        return None, lambda p, t: F.mse_loss(p, t)
    if kind == "bmc":
        mod = BMCLoss()
        return mod, mod
    raise ValueError(f"unknown reg loss {kind!r} (use mse | bmc)")
