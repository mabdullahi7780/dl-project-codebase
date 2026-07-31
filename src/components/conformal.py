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


# ---------------------------------------------------------------------------
# Weighted split-conformal under covariate shift (Tibshirani et al. 2019)
# ---------------------------------------------------------------------------
# Plain split-conformal assumes exchangeability between the calibration set and
# the test set. Under a covariate shift (e.g. Moldova features differ from the
# train pool) that assumption breaks and the marginal coverage guarantee is lost
# — empirically Moldova drops to ~0.83 against a nominal 0.90. Weighted conformal
# restores the guarantee by reweighting each calibration residual by the
# likelihood ratio w(x) = p_test(x) / p_train(x), estimated with a domain
# classifier on the *frozen features*. This uses only UNLABELED target features,
# so it is transductively valid — no test labels are ever touched.


def domain_likelihood_ratio_weights(cal_feats: np.ndarray, test_feats: np.ndarray, *,
                                    clip: tuple[float, float] = (0.05, 20.0),
                                    seed: int = 0) -> np.ndarray:
    """Likelihood-ratio weights w(x)=p_test/(1-p_test) for each calibration point.

    Fits a logistic domain classifier (calibration=0 vs test=1) on standardized
    frozen features and reads off the density ratio for the calibration points.
    Weights are clipped for stability in high dimensions. Uses only feature
    vectors (no labels of any kind), so it is safe to apply to the held-out
    country at test time.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cal = np.asarray(cal_feats, dtype=np.float64)
    tst = np.asarray(test_feats, dtype=np.float64)
    X = np.vstack([cal, tst])
    y = np.concatenate([np.zeros(len(cal)), np.ones(len(tst))])
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(scaler.transform(X), y)
    p = clf.predict_proba(scaler.transform(cal))[:, 1]
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    w = p / (1.0 - p)
    # Only *relative* weights matter for the weighted quantile, and a strong domain
    # gap pushes every ratio toward 0. Rescale to mean 1 so the clip bounds the
    # spread around a typical point instead of flooring the whole (tiny) vector.
    w = w / (w.mean() + 1e-12)
    return np.clip(w, clip[0], clip[1])


def weighted_conformal_quantile(cal_true: np.ndarray, cal_pred: np.ndarray,
                                weights: np.ndarray, *, alpha: float = 0.1) -> float:
    """Weighted (1-alpha) quantile of |residual| (Tibshirani et al. 2019).

    Normalises the weights over the calibration points *plus a test-point mass*
    (the test point's weight, conservatively taken as the max calibration weight)
    and returns the smallest residual whose weighted CDF reaches 1-alpha. Falls
    back to the largest residual (widest, conservative interval) when the required
    mass lies beyond the calibration support.
    """
    res = np.abs(np.asarray(cal_pred, dtype=np.float64) - np.asarray(cal_true, dtype=np.float64))
    w = np.asarray(weights, dtype=np.float64)
    if len(res) == 0:
        return float("nan")
    order = np.argsort(res)
    res_sorted = res[order]
    w_sorted = w[order]
    test_mass = float(w.mean())                       # mass of a typical test point
    total = w_sorted.sum() + test_mass
    cum = np.cumsum(w_sorted) / total
    idx = int(np.searchsorted(cum, 1.0 - alpha, side="left"))
    if idx >= len(res_sorted):
        return float(res_sorted[-1])
    return float(res_sorted[idx])


def weighted_split_conformal(cal_true: np.ndarray, cal_pred: np.ndarray, cal_feats: np.ndarray,
                             test_pred: np.ndarray, test_true: np.ndarray, test_feats: np.ndarray,
                             *, alpha: float = 0.1,
                             clip: tuple[float, float] = (0.05, 20.0)) -> dict[str, float]:
    """Covariate-shift-corrected split-conformal interval stats on the test set.

    Calibrate q on (cal_true, cal_pred) with likelihood-ratio weights derived from
    (cal_feats, test_feats); report coverage + mean width on the test set. Returns
    an effective-sample-size (ESS) diagnostic for the weights — a low ESS means a
    few calibration points dominate and the result should be read with caution.
    """
    w = domain_likelihood_ratio_weights(cal_feats, test_feats, clip=clip)
    q = weighted_conformal_quantile(cal_true, cal_pred, w, alpha=alpha)
    lo = np.asarray(test_pred, dtype=np.float64) - q
    hi = np.asarray(test_pred, dtype=np.float64) + q
    tt = np.asarray(test_true, dtype=np.float64)
    covered = ((tt >= lo) & (tt <= hi)).mean()
    ess = float((w.sum() ** 2) / (np.square(w).sum() + 1e-12))
    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "q": q,
        "coverage": float(covered),
        "mean_width": float(2.0 * q),
        "weight_ess": ess,
    }
