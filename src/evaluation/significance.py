"""Paired-bootstrap significance + conformal decomposition utilities.

Builds the per-(mode, rung, country) Δ-MAE table the paper needs: for every cell,
the rung's improvement over the locked baseline (or any chosen reference) with a
percentile-bootstrap 95 % CI, computed by *pairing on image_id*. A CI excluding
zero is the conventional "p < 0.05" significance flag.

Conformal coverage decomposition: marginal, per-country, and per-severity-band so
the paper can claim "coverage holds across countries" or surface where it fails.

All inputs are the per-image prediction CSVs already written by ``train_agentic.py``
(columns: ``image_id, held_out, seed, rung, timika_true, timika_pred``).
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def paired_bootstrap_mae_delta(y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray,
                               *, n_boot: int = 2000, seed: int = 1337
                               ) -> tuple[float, float, float, float]:
    """95 % CI of MAE(a) - MAE(b) via paired resampling. Returns (delta, lo, hi, frac<0).

    ``frac<0`` is the fraction of bootstrap replicates with delta < 0 (a's MAE lower
    than b's = a wins). Useful as a one-sided significance proxy.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = _mae(y_true[idx], y_a[idx]) - _mae(y_true[idx], y_b[idx])
    point = _mae(y_true, y_a) - _mae(y_true, y_b)
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return point, lo, hi, float((deltas < 0).mean())


def load_preds(preds_glob: str) -> pd.DataFrame:
    """Concatenate every preds_*.csv matched by ``preds_glob``."""
    rows = [pd.read_csv(p) for p in sorted(glob(preds_glob))]
    if not rows:
        raise FileNotFoundError(f"no files matched {preds_glob!r}")
    return pd.concat(rows, ignore_index=True)


def paired_bootstrap_grid(preds_df: pd.DataFrame, *, reference_rung: str,
                          n_boot: int = 2000, seed: int = 1337) -> pd.DataFrame:
    """Per (held_out, rung) Δ-MAE vs ``reference_rung``, averaged across seeds.

    Predictions are paired on image_id within each (held_out, seed); we average
    each rung's per-image prediction across seeds first, then bootstrap on test
    images. Returns long-form DataFrame ready to pivot.
    """
    out = []
    avg = (preds_df.groupby(["held_out", "rung", "image_id"])
                  [["timika_true", "timika_pred"]].mean().reset_index())
    for ho in avg["held_out"].unique():
        sub = avg[avg["held_out"] == ho]
        ref = sub[sub["rung"] == reference_rung]
        if ref.empty:
            continue
        ref = ref.set_index("image_id")
        for rung in sub["rung"].unique():
            if rung == reference_rung:
                continue
            cur = sub[sub["rung"] == rung].set_index("image_id")
            common = ref.index.intersection(cur.index)
            if len(common) == 0:
                continue
            y_true = ref.loc[common, "timika_true"].to_numpy()
            y_a = cur.loc[common, "timika_pred"].to_numpy()
            y_b = ref.loc[common, "timika_pred"].to_numpy()
            d, lo, hi, frac_neg = paired_bootstrap_mae_delta(y_true, y_a, y_b,
                                                             n_boot=n_boot, seed=seed)
            sig = "YES" if (hi < 0 or lo > 0) else "no"
            out.append({"held_out": ho, "rung": rung, "n": len(common),
                        "delta_mae": d, "ci_lo": lo, "ci_hi": hi,
                        "frac_a_better": frac_neg, "significant": sig})
    return pd.DataFrame(out)


def conformal_decomposition(preds_df: pd.DataFrame, q_by_country: dict[str, float],
                            *, severity_bands: list[tuple[float, float]] | None = None
                            ) -> pd.DataFrame:
    """Empirical coverage of conformal intervals decomposed per-country + per-severity-band.

    ``q_by_country`` maps held-out country -> conformal half-width (q). Coverage =
    fraction of test images whose true Timika lies within [pred - q, pred + q].
    """
    if severity_bands is None:
        severity_bands = [(0, 30), (30, 60), (60, 100), (100, 140)]
    rows = []
    for ho, q in q_by_country.items():
        sub = preds_df[preds_df["held_out"] == ho]
        if sub.empty:
            continue
        in_band_all = np.abs(sub["timika_pred"] - sub["timika_true"]) <= q
        rows.append({"held_out": ho, "band": "ALL", "n": len(sub),
                     "coverage": float(in_band_all.mean()), "width": 2 * q})
        for lo, hi in severity_bands:
            mask = (sub["timika_true"] >= lo) & (sub["timika_true"] < hi)
            if mask.sum() == 0:
                continue
            cov = float((np.abs(sub.loc[mask, "timika_pred"] - sub.loc[mask, "timika_true"]) <= q).mean())
            rows.append({"held_out": ho, "band": f"[{lo},{hi})", "n": int(mask.sum()),
                         "coverage": cov, "width": 2 * q})
    return pd.DataFrame(rows)


def per_severity_mae(preds_df: pd.DataFrame,
                     severity_bands: list[tuple[float, float]] | None = None
                     ) -> pd.DataFrame:
    """Mean Timika MAE per (country, severity band). Surfaces *where* the model errs."""
    if severity_bands is None:
        severity_bands = [(0, 30), (30, 60), (60, 100), (100, 140)]
    rows = []
    for (ho, rung), sub in preds_df.groupby(["held_out", "rung"]):
        for lo, hi in severity_bands:
            mask = (sub["timika_true"] >= lo) & (sub["timika_true"] < hi)
            if mask.sum() == 0:
                continue
            mae = float(np.mean(np.abs(sub.loc[mask, "timika_pred"] - sub.loc[mask, "timika_true"])))
            rows.append({"held_out": ho, "rung": rung, "band": f"[{lo},{hi})",
                         "n": int(mask.sum()), "mae": mae})
    return pd.DataFrame(rows)
