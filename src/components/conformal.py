"""Rung 6: split-conformal prediction intervals + deep ensemble.

Two contributions on top of the point predictor:

  * ``ensemble_mean`` — average M independently-seeded heads. Deep ensembles give
    a small, reliable accuracy bump and a free epistemic-uncertainty estimate
    (inter-member std), at M x the (already tiny) training cost.

  * ``split_conformal`` — distribution-free Timika prediction intervals with a
    finite-sample marginal coverage guarantee (Vovk; Lei et al. 2018). We
    calibrate the residual quantile on a held-out split, then report empirical
    coverage and mean interval width on the test country. Under domain shift the
    marginal guarantee can break, so we also report *country-conditional*
    coverage — quantifying exactly how miscalibrated severity estimates become
    when the test domain shifts, which is the clinically honest thing to report.
"""

from __future__ import annotations

import numpy as np


def ensemble_mean(preds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and inter-member std of a list of [N] prediction vectors."""
    stack = np.stack([np.asarray(p, dtype=np.float32) for p in preds], axis=0)  # [M, N]
    return stack.mean(axis=0), stack.std(axis=0)


def conformal_quantile(cal_true: np.ndarray, cal_pred: np.ndarray, *,
                       alpha: float = 0.1) -> float:
    """Finite-sample-corrected (1-alpha) quantile of absolute residuals."""
    res = np.abs(np.asarray(cal_pred, dtype=np.float64) - np.asarray(cal_true, dtype=np.float64))
    n = len(res)
    if n == 0:
        return float("nan")
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(res, level, method="higher"))


def split_conformal(cal_true: np.ndarray, cal_pred: np.ndarray,
                    test_pred: np.ndarray, test_true: np.ndarray, *,
                    alpha: float = 0.1) -> dict[str, float]:
    """Calibrate on (cal_true, cal_pred); report interval stats on the test set."""
    q = conformal_quantile(cal_true, cal_pred, alpha=alpha)
    lo = np.asarray(test_pred, dtype=np.float64) - q
    hi = np.asarray(test_pred, dtype=np.float64) + q
    tt = np.asarray(test_true, dtype=np.float64)
    covered = ((tt >= lo) & (tt <= hi)).mean()
    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "q": q,
        "coverage": float(covered),
        "mean_width": float(2.0 * q),
    }
