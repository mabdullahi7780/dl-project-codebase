"""Unit tests for weighted split-conformal (WS-B Moldova fix).

Weighted conformal must (a) reduce to ordinary conformal when calibration and test
share a distribution, and (b) inflate the interval — raising coverage — when the
test set is covariate-shifted and residuals grow with the shifted feature. All
tests use synthetic data and the pure functions only (no GPU, no repo state).
"""
from __future__ import annotations

import numpy as np

from src.components.conformal import (
    conformal_quantile,
    domain_likelihood_ratio_weights,
    split_conformal,
    weighted_conformal_quantile,
    weighted_split_conformal,
)

ALPHA = 0.1


def _coverage(test_true, test_pred, q):
    lo, hi = test_pred - q, test_pred + q
    return float(((test_true >= lo) & (test_true <= hi)).mean())


def test_no_shift_reduces_to_ordinary_conformal():
    rng = np.random.default_rng(0)
    d = 8
    cal_feats = rng.normal(0, 1, (800, d))
    test_feats = rng.normal(0, 1, (400, d))          # same distribution
    cal_true = np.zeros(800)
    cal_pred = rng.normal(0, 10, 800)                # residuals iid, no feature link
    test_true = np.zeros(400)
    test_pred = rng.normal(0, 10, 400)

    w = domain_likelihood_ratio_weights(cal_feats, test_feats)
    # classifier cannot separate identical distributions -> weights near 1
    assert 0.5 < float(w.mean()) < 2.0

    std = split_conformal(cal_true, cal_pred, test_pred, test_true, alpha=ALPHA)
    wt = weighted_split_conformal(cal_true, cal_pred, cal_feats,
                                  test_pred, test_true, test_feats, alpha=ALPHA)
    # weighted coverage should track standard coverage closely under no shift
    assert abs(wt["coverage"] - std["coverage"]) < 0.10
    assert wt["coverage"] >= 0.80


def test_target_shift_widens_and_lifts_coverage():
    # Moderate covariate shift (representative of Moldova): standard split-conformal
    # under-covers; weighted conformal widens the interval and lifts coverage toward
    # nominal. The recovery is real but modest -- it cannot fully fix a shift the
    # calibration set has limited overlap with (that is the honest-null regime).
    rng = np.random.default_rng(1)
    d = 8
    # calibration: feature ~ N(0,1); residual grows with feature 0
    cal_f = rng.normal(0.0, 1.0, (1200, d))
    cal_res = np.abs(cal_f[:, 0] * 6.0 + rng.normal(0, 4, 1200))
    cal_true = np.zeros(1200)
    cal_pred = cal_res                                # |pred-true| = cal_res

    # test: feature 0 shifted up -> test residuals larger, exchangeability broken
    test_f = rng.normal(1.0, 1.0, (500, d))
    test_res = np.abs(test_f[:, 0] * 6.0 + rng.normal(0, 4, 500))
    test_true = np.zeros(500)
    test_pred = test_res

    q_std = conformal_quantile(cal_true, cal_pred, alpha=ALPHA)
    w = domain_likelihood_ratio_weights(cal_f, test_f)
    q_wt = weighted_conformal_quantile(cal_true, cal_pred, w, alpha=ALPHA)

    # weighting toward test-like (high-feature) calibration points widens the interval
    assert q_wt > q_std
    # and that lifts test coverage meaningfully toward nominal (0.90)
    cov_std = _coverage(test_true, test_pred, q_std)
    cov_wt = _coverage(test_true, test_pred, q_wt)
    assert cov_wt > cov_std
    assert cov_wt - cov_std >= 0.02
    assert cov_wt <= 1.0


def test_weights_clipped_and_ess_valid():
    rng = np.random.default_rng(2)
    d = 8
    cal_f = rng.normal(0, 1, (600, d))
    test_f = rng.normal(3.0, 1, (300, d))            # strong shift -> extreme ratios
    w = domain_likelihood_ratio_weights(cal_f, test_f, clip=(0.05, 20.0))
    assert w.min() >= 0.05 - 1e-9
    assert w.max() <= 20.0 + 1e-9

    out = weighted_split_conformal(np.zeros(600), rng.normal(0, 5, 600), cal_f,
                                   rng.normal(0, 5, 300), np.zeros(300), test_f, alpha=ALPHA)
    assert 0.0 < out["weight_ess"] <= 600.0
    assert out["mean_width"] == 2.0 * out["q"]
