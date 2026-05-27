"""Rung 2: test-time feature adaptation (the covariate-shift ablation).

Two cheap, standard domain-generalisation moves on frozen features:

  * ``FeatureStandardizer`` — z-score features. Fit on the train pool (inductive)
    or re-fit on the held-out country at test (transductive, a.k.a. test-time BN
    adaptation): the head sees zero-mean/unit-var features regardless of any
    per-country intensity/contrast offset.
  * ``coral_align`` — CORAL (Sun & Saenko 2016): whiten source features and
    re-colour them with the target covariance, removing 2nd-order domain shift.

HONEST NOTE: we diagnosed Moldova as *label* shift, not covariate shift, so these
are expected to help little. They are included as the ablation reviewers expect —
to *demonstrate* that aligning feature statistics does not close the gap, which is
itself evidence for the label-shift diagnosis.
"""

from __future__ import annotations

import numpy as np


class FeatureStandardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, feats: np.ndarray) -> "FeatureStandardizer":
        f = np.asarray(feats, dtype=np.float32)
        self.mean_ = f.mean(axis=0, keepdims=True)
        self.std_ = f.std(axis=0, keepdims=True) + 1e-6
        return self

    def transform(self, feats: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("FeatureStandardizer.fit must be called first")
        return ((np.asarray(feats, dtype=np.float32) - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, feats: np.ndarray) -> np.ndarray:
        return self.fit(feats).transform(feats)


def coral_align(source: np.ndarray, target: np.ndarray, *, eps: float = 1e-5) -> np.ndarray:
    """Re-colour ``source`` features to match ``target`` covariance (CORAL).

    Returns adapted source features with the same shape. Used to align the test
    country's features toward the train-pool statistics (or vice versa).
    """
    s = np.asarray(source, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    d = s.shape[1]
    cs = np.cov(s, rowvar=False) + eps * np.eye(d)
    ct = np.cov(t, rowvar=False) + eps * np.eye(d)
    s_white = s @ _inv_sqrt(cs)
    s_recolor = s_white @ _sqrtm(ct)
    return s_recolor.astype(np.float32)


def _sqrtm(mat: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.T


def _inv_sqrt(mat: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, 1e-12, None)
    return (v * (1.0 / np.sqrt(w))) @ v.T
