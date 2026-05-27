"""Rung 3: retrieval-augmented label calibration (the agentic novelty core).

A plain regressor trained on the source countries collapses toward the *training*
label mean on the sicker held-out country (Moldova) — this is label-distribution
shift, not covariate shift, so domain-invariance hurts. Retrieval calibration
attacks it directly and non-parametrically: for each test image, find its k
nearest neighbours among the TRAIN pool in frozen RAD-DINO feature space and read
off their labels. If a Moldova image's neighbours are high-severity training
images, the retrieved estimate is high — pulling the prediction off the mean.

The final prediction blends the parametric head and the retrieved estimate::

    y = (1 - alpha) * head_pred + alpha * knn_label_mean

``alpha`` is selected on the validation split (held-out from the train pool), so
the blend is honest — never tuned on the test country. ``alpha = 0`` recovers the
pure head (the ablation), making the retrieval contribution directly measurable.

This is a transductive, training-free calibrator over a foundation feature space;
applying it to cross-country TB severity (Timika) regression is, to our knowledge,
novel.
"""

from __future__ import annotations

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


class RetrievalCalibrator:
    """kNN label estimate in feature space, blended with a parametric prediction.

    Targets are kept on the model's working scale (e.g. ALP/100 in [0,1]); the
    caller rescales afterwards. Cosine similarity with a softmax-over-distance
    kernel weights neighbours (closer = more weight).
    """

    def __init__(self, k: int = 20, *, temperature: float = 0.1) -> None:
        self.k = k
        self.temperature = temperature
        self._feats: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def fit(self, feats: np.ndarray, labels: np.ndarray) -> "RetrievalCalibrator":
        self._feats = _l2_normalize(np.asarray(feats, dtype=np.float32))
        self._labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        return self

    def retrieve(self, feats: np.ndarray) -> np.ndarray:
        """Return the kNN label estimate for each query (no blending)."""
        if self._feats is None:
            raise RuntimeError("RetrievalCalibrator.fit must be called first")
        q = _l2_normalize(np.asarray(feats, dtype=np.float32))
        sims = q @ self._feats.T                              # [Nq, Ntrain] cosine
        k = min(self.k, sims.shape[1])
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]    # top-k per row
        rows = np.arange(sims.shape[0])[:, None]
        top_sims = sims[rows, idx]                            # [Nq, k]
        w = np.exp(top_sims / self.temperature)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
        return (w * self._labels[idx]).sum(axis=1)            # [Nq]

    def calibrate(self, feats: np.ndarray, head_pred: np.ndarray, alpha: float) -> np.ndarray:
        """Blend: (1-alpha)*head + alpha*retrieved. alpha=0 -> pure head."""
        if alpha <= 0.0:
            return np.asarray(head_pred, dtype=np.float32)
        knn = self.retrieve(feats)
        return ((1.0 - alpha) * np.asarray(head_pred, dtype=np.float32)
                + alpha * knn).astype(np.float32)

    def select_alpha(self, val_feats: np.ndarray, val_head_pred: np.ndarray,
                     val_true: np.ndarray, *,
                     grid: np.ndarray | None = None) -> tuple[float, float]:
        """Pick alpha minimising val MAE over a grid. Returns (alpha, val_mae)."""
        if grid is None:
            grid = np.linspace(0.0, 1.0, 21)
        vt = np.asarray(val_true, dtype=np.float32)
        best_a, best_mae = 0.0, float("inf")
        for a in grid:
            blended = self.calibrate(val_feats, val_head_pred, float(a))
            mae = float(np.mean(np.abs(blended - vt)))
            if mae < best_mae:
                best_mae, best_a = mae, float(a)
        return best_a, best_mae
