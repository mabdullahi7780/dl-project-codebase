#!/usr/bin/env python3.11
"""WS-SP: Selective prediction / abstention analysis.

Reads the per-image prediction CSVs written by ``train_agentic.py`` (with ``--save``),
ranks each held-out country's test images by ensemble uncertainty (``ens_std``), defers
the most-uncertain fraction to a radiologist, and measures Timika MAE on the *retained*
cases plus severe-case false negatives.

The headline result is a **risk-coverage curve**: as we defer the most-uncertain cases,
retained-case MAE should decrease, and deferring by uncertainty should beat deferring at
random. A flat curve (uncertainty no better than random) is itself an honest, reportable
finding consistent with the shift-decomposition framing.

IMPORTANT (see sprint_selective_prediction.md §3): we rank by the deep-ensemble
inter-member std (``ens_std``), NOT the conformal interval width. Our split-conformal gives
every image the same width, so it cannot rank cases.

Usage
-----
    python3.11 scripts/selective_prediction.py \
        --preds-dir baseline_runs/agentic_runs/final_runs \
        --mode fusion --rung rung6_conformal \
        --out-dir ws_sp

Inputs
------
Per-image CSVs named ``preds_{mode}_{rung}_{country}_s{seed}.csv`` with at least columns:
    image_id, held_out, seed, rung, ensemble, timika_true, timika_pred, ens_std,
    cavity_true, cavity_prob, cavity_pred
(These are produced by the WS-SP trainer patch. Older CSVs lacking ``ens_std`` are rejected
with an instruction to re-run the patched trainer.)

Outputs (into --out-dir)
------------------------
    selective_prediction.csv   risk-coverage table (per country + pooled, per retained fraction)
    fig_risk_coverage.pdf/.png the risk-coverage figure (uncertainty vs random deferral)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RETAINED_FRACTIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
SEVERE_TIMIKA = 100.0          # "very-severe" band lower bound (paper §5.8)
N_RANDOM_DRAWS = 500           # Monte-Carlo draws for the random-deferral baseline
REQUIRED_COLS = ["held_out", "seed", "timika_true", "timika_pred", "ens_std"]

# Colour-blind-safe, distinct per country (Okabe-Ito subset).
COUNTRY_COLOURS = {
    "Romania": "#0072B2",
    "Moldova": "#D55E00",
    "Kazakhstan": "#009E73",
    "ALL": "#333333",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _mode_from_filename(path: str) -> str:
    """preds_{mode}_{rung...}_{country}_s{seed}.csv -> mode (first token)."""
    stem = Path(path).stem  # drop .csv
    assert stem.startswith("preds_"), stem
    return stem[len("preds_"):].split("_", 1)[0]


def load_preds(preds_dir: str, mode: str | None, rung: str | None) -> pd.DataFrame:
    """Load and concatenate matching per-image prediction CSVs."""
    files = sorted(glob.glob(os.path.join(preds_dir, "preds_*.csv")))
    if not files:
        sys.exit(f"[selective] no preds_*.csv found in {preds_dir!r}. "
                 f"Run the patched train_agentic.py with --save first (sprint Step 2).")
    frames = []
    for f in files:
        fmode = _mode_from_filename(f)
        if mode and fmode != mode:
            continue
        df = pd.read_csv(f)
        if rung and "rung" in df.columns and not (df["rung"].astype(str) == rung).all():
            df = df[df["rung"].astype(str) == rung]
        if len(df):
            df = df.copy()
            df["mode"] = fmode
            frames.append(df)
    if not frames:
        sys.exit(f"[selective] no rows matched mode={mode!r} rung={rung!r} in {preds_dir!r}.")
    out = pd.concat(frames, ignore_index=True)

    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing:
        sys.exit(f"[selective] preds CSVs are missing columns {missing}. "
                 f"These files predate the WS-SP trainer patch — re-run train_agentic.py "
                 f"with the patched version so ens_std is saved (sprint Step 1/2).")
    if not (out["ens_std"].abs() > 0).any():
        sys.exit("[selective] ens_std is all-zero — the run used ensemble M=1. "
                 "Selective prediction needs the R6 ensemble (M>1); re-run Fusion+R6.")
    return out


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------
def _abs_err(df: pd.DataFrame) -> np.ndarray:
    return np.abs(df["timika_true"].to_numpy(float) - df["timika_pred"].to_numpy(float))


def _severe_cavity_fn(df: pd.DataFrame) -> tuple[int, int]:
    """(#cavity false-negatives, #cavity-positive) among the given rows.

    A cavity FN under-scores Timika by the fixed 40-point penalty — the clinically
    costly error. Returns (fn, n_pos); NaN cavity columns (a3 mode) yield (0, 0).
    """
    if "cavity_true" not in df.columns or "cavity_pred" not in df.columns:
        return 0, 0
    ct = df["cavity_true"].to_numpy(float)
    cp = df["cavity_pred"].to_numpy(float)
    valid = ~np.isnan(ct) & ~np.isnan(cp)
    ct, cp = ct[valid], cp[valid]
    n_pos = int((ct == 1).sum())
    fn = int(((ct == 1) & (cp == 0)).sum())
    return fn, n_pos


def curve_for_group(g: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """Risk-coverage points for one (country, seed) group."""
    g = g.sort_values("ens_std", ascending=True).reset_index(drop=True)  # most confident first
    n = len(g)
    err_all = _abs_err(g)
    rows = []
    for f in RETAINED_FRACTIONS:
        k = max(1, int(round(f * n)))
        # uncertainty deferral: keep the k most-confident
        kept = g.iloc[:k]
        mae_unc = float(_abs_err(kept).mean())
        # random deferral: mean MAE over random size-k subsets
        rand = [float(err_all[rng.choice(n, size=k, replace=False)].mean())
                for _ in range(N_RANDOM_DRAWS)]
        mae_rand = float(np.mean(rand))
        fn, n_pos = _severe_cavity_fn(kept)
        n_severe_true = int((kept["timika_true"].to_numpy(float) >= SEVERE_TIMIKA).sum())
        rows.append({
            "retained_fraction": f, "n_retained": k,
            "mae_uncertainty": mae_unc, "mae_random": mae_rand,
            "cavity_fn": fn, "cavity_pos": n_pos,
            "n_severe_true_retained": n_severe_true,
        })
    return rows


def build_table(preds: pd.DataFrame, seed_rng: int = 0) -> pd.DataFrame:
    """Per (held_out, seed) curves, then aggregate across seeds. Adds pooled 'ALL'."""
    rng = np.random.default_rng(seed_rng)
    records = []
    # per-country, per-seed
    for (country, seed), g in preds.groupby(["held_out", "seed"]):
        for r in curve_for_group(g, rng):
            records.append({"held_out": country, "seed": seed, **r})
    # pooled across countries, per-seed
    for seed, g in preds.groupby("seed"):
        for r in curve_for_group(g, rng):
            records.append({"held_out": "ALL", "seed": seed, **r})

    per_seed = pd.DataFrame(records)

    def _agg(sub: pd.DataFrame) -> pd.Series:
        cav_pos = sub["cavity_pos"].sum()
        return pd.Series({
            "n_seeds": sub["seed"].nunique(),
            "n_retained_mean": sub["n_retained"].mean(),
            "mae_uncertainty_mean": sub["mae_uncertainty"].mean(),
            "mae_uncertainty_std": sub["mae_uncertainty"].std(ddof=0),
            "mae_random_mean": sub["mae_random"].mean(),
            "mae_random_std": sub["mae_random"].std(ddof=0),
            "cavity_fn_mean": sub["cavity_fn"].mean(),
            "cavity_fn_rate": (sub["cavity_fn"].sum() / cav_pos) if cav_pos else np.nan,
            "n_severe_true_retained_mean": sub["n_severe_true_retained"].mean(),
        })

    table = (per_seed.groupby(["held_out", "retained_fraction"], as_index=False)
             .apply(_agg, include_groups=False)
             .reset_index(drop=True))
    return table.sort_values(["held_out", "retained_fraction"], ascending=[True, False]) \
                .reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure + sanity checks
# ---------------------------------------------------------------------------
def make_figure(table: pd.DataFrame, out_dir: Path, mode: str, rung: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    countries = [c for c in ["Romania", "Moldova", "Kazakhstan", "ALL"]
                 if c in table["held_out"].unique()]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for c in countries:
        sub = table[table["held_out"] == c].sort_values("retained_fraction", ascending=False)
        x = sub["retained_fraction"].to_numpy() * 100.0
        col = COUNTRY_COLOURS.get(c, "#666666")
        lw = 2.4 if c == "ALL" else 1.8
        ax.plot(x, sub["mae_uncertainty_mean"], "-o", color=col, lw=lw, ms=5,
                label=f"{c} (defer by uncertainty)")
        ax.plot(x, sub["mae_random_mean"], "--", color=col, lw=1.0, alpha=0.55,
                label=f"{c} (random deferral)")
    ax.set_xlabel("Retained fraction of cases (%)  —  fewer deferred →")
    ax.set_ylabel("Timika MAE on retained cases")
    ax.set_title(f"Selective prediction / abstention  ({mode} · {rung})")
    ax.invert_xaxis()  # 100% (keep all) on the left, more deferral to the right
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_risk_coverage.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def sanity_checks(table: pd.DataFrame) -> None:
    print("\n=== Sanity checks (WS-SP Step 4) ===")
    for c in table["held_out"].unique():
        sub = table[table["held_out"] == c].sort_values("retained_fraction", ascending=False)
        mae_full = sub.iloc[0]["mae_uncertainty_mean"]
        mae_min_frac = sub.iloc[-1]["mae_uncertainty_mean"]
        monotone = mae_min_frac <= mae_full + 1e-9
        beats_random = bool((sub["mae_uncertainty_mean"] <= sub["mae_random_mean"] + 1e-9).all())
        drop_at_80 = None
        row80 = sub[np.isclose(sub["retained_fraction"], 0.8)]
        if len(row80):
            drop_at_80 = mae_full - float(row80.iloc[0]["mae_uncertainty_mean"])
        tag = "OK" if (monotone and beats_random) else "CHECK"
        d80 = f"{drop_at_80:+.2f}" if drop_at_80 is not None else "n/a"
        print(f"  [{tag}] {c:<11} full-MAE={mae_full:5.2f}  "
              f"MAE-drop@80%={d80}  monotone↓={monotone}  beats-random={beats_random}")
    print("  (a flat / non-monotone curve is a valid, reportable null — see sprint §8)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="WS-SP selective prediction / abstention analysis")
    ap.add_argument("--preds-dir", required=True,
                    help="dir with preds_*.csv from train_agentic.py --save")
    ap.add_argument("--mode", default="fusion",
                    help="prediction mode to analyse (fusion|a2|a3|a1); default fusion")
    ap.add_argument("--rung", default="rung6_conformal",
                    help="rung name to filter on (must be an M>1 ensemble rung); "
                         "default rung6_conformal")
    ap.add_argument("--out-dir", default="ws_sp")
    ap.add_argument("--no-fig", action="store_true", help="skip the figure")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = load_preds(args.preds_dir, args.mode, args.rung)
    print(f"[selective] loaded {len(preds)} rows  "
          f"({preds['held_out'].nunique()} countries × {preds['seed'].nunique()} seeds, "
          f"mode={args.mode}, rung={args.rung})")

    table = build_table(preds)
    csv_path = out_dir / "selective_prediction.csv"
    table.to_csv(csv_path, index=False)
    print(f"[selective] wrote {csv_path}")

    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print("\n=== Risk-coverage table ===")
        print(table[["held_out", "retained_fraction", "n_retained_mean",
                     "mae_uncertainty_mean", "mae_random_mean",
                     "cavity_fn_rate", "n_severe_true_retained_mean"]].to_string(index=False))

    sanity_checks(table)

    if not args.no_fig:
        make_figure(table, out_dir, args.mode, args.rung)
        print(f"[selective] wrote {out_dir/'fig_risk_coverage.pdf'} (+ .png)")


if __name__ == "__main__":
    main()
