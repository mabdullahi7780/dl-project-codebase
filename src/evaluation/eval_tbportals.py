"""Evaluation for TB Portals Timika scoring, matched to Kantipudi et al.

Metrics on the radiologist scale:
- ALP / Timika regression: MAE, MAE% (MAE/140*100), RMSE, Pearson r, Spearman, R^2
- Cavity classification: AUC, F1, precision, recall
- Bootstrap 95% CIs (paired resampling over images)

``KANTIPUDI_A2`` holds their reported best-model numbers so the comparison table
prints ours-vs-theirs per held-out country.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

CAVITY_BONUS = 40.0
TIMIKA_MAX = 140.0

# Kantipudi et al. (JIIM 2024), best model A2, country-segregated test.
KANTIPUDI_A2: dict[str, dict[str, float]] = {
    "Romania": {"timika_mae": 18.70, "timika_mae_pct": 13.36, "timika_pearson": 0.70,
                "alp_mae": 11.86, "cavity_auc": 0.80, "cavity_f1": 0.81},
    "Moldova": {"timika_mae": 18.85, "timika_mae_pct": 13.46, "timika_pearson": 0.84,
                "alp_mae": 16.24, "cavity_auc": 0.88, "cavity_f1": 0.71},
    "Kazakhstan": {"timika_mae": 19.62, "timika_mae_pct": 14.01, "timika_pearson": 0.70,
                   "alp_mae": 12.16, "cavity_auc": 0.85, "cavity_f1": 0.72},
}

# Kantipudi et al. (JIIM 2024), approach A3 (direct Timika regression), Table 7.
KANTIPUDI_A3: dict[str, dict[str, float]] = {
    "Romania":    {"timika_mae": 19.67, "timika_mae_pct": 14.05, "timika_pearson": 0.70},
    "Moldova":    {"timika_mae": 18.98, "timika_mae_pct": 13.56, "timika_pearson": 0.85},
    "Kazakhstan": {"timika_mae": 22.12, "timika_mae_pct": 15.80, "timika_pearson": 0.74},
}

# Kantipudi et al. (JIIM 2024), approach A1 (lung-seg + YOLOv5 lesion-det + cavity),
# Table 7. Cavity AUC/F1 are shared with A2 (same classifier). A1 is their WORST
# approach; ALP comes from detection geometry, not regression.
KANTIPUDI_A1: dict[str, dict[str, float]] = {
    "Romania":    {"timika_mae": 23.83, "timika_mae_pct": 17.02, "timika_pearson": 0.59,
                   "cavity_auc": 0.80, "cavity_f1": 0.81},
    "Moldova":    {"timika_mae": 24.44, "timika_mae_pct": 17.46, "timika_pearson": 0.80,
                   "cavity_auc": 0.88, "cavity_f1": 0.71},
    "Kazakhstan": {"timika_mae": 22.13, "timika_mae_pct": 15.81, "timika_pearson": 0.68,
                   "cavity_auc": 0.85, "cavity_f1": 0.72},
}


# OUR locked single-network baselines (3-seed means, baseline_runs/BASELINE_COMPARISON.md).
# These are the HONEST target the agentic model must beat (Kantipudi above is aspirational).
LOCKED_BASELINE: dict[str, dict[str, dict[str, float]]] = {
    "a2": {"Romania": {"timika_mae": 20.11, "timika_pearson": 0.684},
           "Moldova": {"timika_mae": 30.68, "timika_pearson": 0.697},
           "Kazakhstan": {"timika_mae": 21.35, "timika_pearson": 0.648}},
    "a3": {"Romania": {"timika_mae": 20.26, "timika_pearson": 0.699},
           "Moldova": {"timika_mae": 26.16, "timika_pearson": 0.759},
           "Kazakhstan": {"timika_mae": 21.90, "timika_pearson": 0.692}},
    "a1": {"Romania": {"timika_mae": 26.84, "timika_pearson": 0.570},
           "Moldova": {"timika_mae": 32.76, "timika_pearson": 0.643},
           "Kazakhstan": {"timika_mae": 21.87, "timika_pearson": 0.651}},
}


def paired_bootstrap_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn,
    *,
    n_boot: int = 2000,
    seed: int = 1337,
) -> tuple[float, float, float]:
    """95% CI of metric(a) - metric(b) via PAIRED resampling on the same images.

    Use when both models' per-image predictions exist (e.g. agentic vs a re-run
    baseline). Returns (delta_point, ci_lo, ci_hi). If the CI excludes 0 the
    difference is significant at ~p<0.05.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
        if not np.isnan(d):
            deltas.append(d)
    if not deltas:
        return (point, float("nan"), float("nan"))
    return (point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


@dataclass(slots=True)
class Predictions:
    alp_true_100: np.ndarray
    alp_pred_100: np.ndarray
    cavity_true: np.ndarray
    cavity_prob: np.ndarray


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Predictions:
    model.eval()
    alp_t, alp_p, cav_t, cav_p = [], [], [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        alp_p.append(out.alp.detach().cpu().numpy() * 100.0)
        cav_p.append(torch.sigmoid(out.cavity_logit).detach().cpu().numpy())
        alp_t.append(batch["alp"].numpy() * 100.0)
        cav_t.append(batch["cavity"].numpy())
    return Predictions(
        alp_true_100=np.concatenate(alp_t),
        alp_pred_100=np.concatenate(alp_p),
        cavity_true=np.concatenate(cav_t),
        cavity_prob=np.concatenate(cav_p),
    )


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(pearsonr(a, b)[0])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(spearmanr(a, b)[0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else float("nan")
    return {
        "mae": mae,
        "mae_pct": mae / TIMIKA_MAX * 100.0,
        "rmse": rmse,
        "pearson": _safe_pearson(y_true, y_pred),
        "spearman": _safe_spearman(y_true, y_pred),
        "r2": r2,
    }


def cavity_metrics(y_true: np.ndarray, prob: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob > threshold).astype(int)
    yt = y_true.astype(int)
    if len(np.unique(yt)) < 2:
        auc = float("nan")  # AUC undefined when only one class present
    else:
        auc = float(roc_auc_score(yt, prob))
    return {
        "auc": auc,
        "f1": float(f1_score(yt, pred, zero_division=0)),
        "precision": float(precision_score(yt, pred, zero_division=0)),
        "recall": float(recall_score(yt, pred, zero_division=0)),
        "positives": int(yt.sum()),
        "n": int(len(yt)),
    }


def assemble_timika(alp_100: np.ndarray, cavity_flag: np.ndarray) -> np.ndarray:
    return alp_100 + CAVITY_BONUS * cavity_flag


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    *,
    n_boot: int = 1000,
    seed: int = 1337,
) -> tuple[float, float]:
    """Percentile 95% CI of ``metric_fn(y_true, y_pred)`` via paired resampling."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = metric_fn(y_true[idx], y_pred[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def evaluate_split(preds: Predictions, *, cavity_threshold: float = 0.5) -> dict[str, dict]:
    """Full metric bundle (ALP, cavity, Timika) with bootstrap CIs."""
    cavity_flag_pred = (preds.cavity_prob > cavity_threshold).astype(float)
    timika_true = assemble_timika(preds.alp_true_100, preds.cavity_true)
    timika_pred = assemble_timika(preds.alp_pred_100, cavity_flag_pred)

    alp = regression_metrics(preds.alp_true_100, preds.alp_pred_100)
    timika = regression_metrics(timika_true, timika_pred)
    cavity = cavity_metrics(preds.cavity_true, preds.cavity_prob, threshold=cavity_threshold)

    mae_fn = lambda a, b: float(np.mean(np.abs(b - a)))
    pearson_fn = lambda a, b: _safe_pearson(a, b)
    alp["mae_ci95"] = bootstrap_ci(preds.alp_true_100, preds.alp_pred_100, mae_fn)
    timika["mae_ci95"] = bootstrap_ci(timika_true, timika_pred, mae_fn)
    timika["pearson_ci95"] = bootstrap_ci(timika_true, timika_pred, pearson_fn)

    return {"alp": alp, "timika": timika, "cavity": cavity}


def print_comparison_table(results_by_country: dict[str, dict]) -> None:
    """Print ours-vs-Kantipudi per held-out country."""
    print("\n" + "=" * 78)
    print("TB Portals Timika comparison vs Kantipudi A2  (ours | their reported)")
    print("=" * 78)
    for country, res in results_by_country.items():
        ref = KANTIPUDI_A2.get(country, {})
        t, a, c = res["timika"], res["alp"], res["cavity"]

        def cmp(ours: float, theirs: float | None, lower_better: bool) -> str:
            if theirs is None or np.isnan(ours):
                return ""
            win = (ours < theirs) if lower_better else (ours > theirs)
            return "  <-- BEAT" if win else "  (below)"

        print(f"\n[{country}]  (n={c['n']}, cavity+={c['positives']})")
        print(
            f"  Timika MAE      {t['mae']:6.2f} | {ref.get('timika_mae', float('nan')):6.2f}"
            f"{cmp(t['mae'], ref.get('timika_mae'), True)}   CI95={_fmt_ci(t.get('mae_ci95'))}"
        )
        print(
            f"  Timika MAE%     {t['mae_pct']:6.2f} | {ref.get('timika_mae_pct', float('nan')):6.2f}"
            f"{cmp(t['mae_pct'], ref.get('timika_mae_pct'), True)}"
        )
        print(
            f"  Timika Pearson  {t['pearson']:6.2f} | {ref.get('timika_pearson', float('nan')):6.2f}"
            f"{cmp(t['pearson'], ref.get('timika_pearson'), False)}   CI95={_fmt_ci(t.get('pearson_ci95'))}"
        )
        print(
            f"  ALP MAE         {a['mae']:6.2f} | {ref.get('alp_mae', float('nan')):6.2f}"
            f"{cmp(a['mae'], ref.get('alp_mae'), True)}   CI95={_fmt_ci(a.get('mae_ci95'))}"
        )
        print(
            f"  Cavity AUC      {c['auc']:6.2f} | {ref.get('cavity_auc', float('nan')):6.2f}"
            f"{cmp(c['auc'], ref.get('cavity_auc'), False)}"
        )
        print(
            f"  Cavity F1       {c['f1']:6.2f} | {ref.get('cavity_f1', float('nan')):6.2f}"
            f"{cmp(c['f1'], ref.get('cavity_f1'), False)}"
        )
    print("\n" + "=" * 78)


def _fmt_ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"
