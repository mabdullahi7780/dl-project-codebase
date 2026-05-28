"""Probabilistic calibration metrics for cavity (the only binary head in the pipeline).

Brier score and Expected Calibration Error (ECE) quantify *how well-calibrated* the
cavity probability outputs are — a low AUC head can still be well-calibrated; a high
AUC head can still be over/under-confident. Both matter clinically because Timika
adds 40 points if the cavity *flag* fires, so threshold + confidence interact.

Reliability bins return per-bin (mean confidence, fraction positive, count) — drop
straight into a reliability diagram.
"""

from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between binary label and predicted probability."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, *,
                     n_bins: int = 10) -> dict[str, np.ndarray]:
    """Return per-bin (mean_conf, fraction_pos, count) for a reliability diagram."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    mean_conf = np.zeros(n_bins); frac_pos = np.zeros(n_bins); count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        count[b] = c
        if c == 0:
            continue
        mean_conf[b] = float(y_prob[mask].mean())
        frac_pos[b] = float(y_true[mask].mean())
    return {"mean_conf": mean_conf, "frac_pos": frac_pos, "count": count, "edges": edges}


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *,
                               n_bins: int = 10) -> float:
    """ECE = sum_b (n_b/N) * |mean_conf_b - frac_pos_b|. Lower is better, 0 = perfect."""
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins)
    N = bins["count"].sum()
    if N == 0:
        return float("nan")
    return float(np.sum(bins["count"] / N
                        * np.abs(bins["mean_conf"] - bins["frac_pos"])))


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *,
                              n_bins: int = 10) -> float:
    """MCE = max_b |mean_conf_b - frac_pos_b| over non-empty bins."""
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins)
    nz = bins["count"] > 0
    if not nz.any():
        return float("nan")
    return float(np.max(np.abs(bins["mean_conf"][nz] - bins["frac_pos"][nz])))
