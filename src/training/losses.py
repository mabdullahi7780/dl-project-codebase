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


# ── Rung 4: cavity-head upgrade ───────────────────────────────────────────────
# Romania's Timika MAE is dominated by cavity errors (cavAUC ~0.68; each
# misclassification costs 40 Timika points). Class-balanced focal loss
# (Lin et al. 2017 focal + Cui et al. 2019 effective-number reweighting)
# down-weights easy negatives and re-balances the (often skewed) cavity prior.


def class_balanced_weights(n_per_class: list[int] | np.ndarray, *, beta: float = 0.999) -> torch.Tensor:
    """Effective-number class weights (Cui et al. 2019), normalised to mean 1."""
    n = np.asarray(n_per_class, dtype=np.float64)
    eff = 1.0 - np.power(beta, np.maximum(n, 1.0))
    w = (1.0 - beta) / np.maximum(eff, 1e-12)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def focal_ce(logits: torch.Tensor, target: torch.Tensor, *, gamma: float = 2.0,
             weight: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-class focal cross-entropy. ``logits`` [B, C], ``target`` [B] long."""
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    logp_t = logp.gather(1, target[:, None]).squeeze(1)
    p_t = p.gather(1, target[:, None]).squeeze(1)
    loss = -((1.0 - p_t) ** gamma) * logp_t
    if weight is not None:
        loss = loss * weight.to(logits.device)[target]
    return loss.mean()
