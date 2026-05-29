"""Generate every figure and table the ICONIP paper needs.

Inputs:
  baseline_runs/agentic_runs/final_runs/results_agentic_{a1,a2,a3,fusion}.csv
  baseline_runs/agentic_runs/final_runs/_a{1,2}_extract/preds_*.csv

Outputs:
  iconips_Paper/figures/*.pdf
  iconips_Paper/tables/*.tex
"""

from __future__ import annotations
from glob import glob
from pathlib import Path
import os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "baseline_runs" / "agentic_runs" / "final_runs"
PRED_DIRS = {
    "a1": RES_DIR / "_a1_extract",
    "a2": RES_DIR / "_a2_extract",
    "a3": RES_DIR / "_a3_extract" / "a3",
}
FIG_DIR = ROOT / "iconips_Paper" / "figures"
TAB_DIR = ROOT / "iconips_Paper" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["Romania", "Moldova", "Kazakhstan"]
MODES = ["a1", "a2", "a3", "fusion"]

LOCK_BASELINE = {  # our DenseNet replication of Kantipudi
    "a1": {"Romania": 26.84, "Moldova": 32.76, "Kazakhstan": 21.87},
    "a2": {"Romania": 20.11, "Moldova": 30.68, "Kazakhstan": 21.35},
    "a3": {"Romania": 20.26, "Moldova": 26.16, "Kazakhstan": 21.90},
    "fusion": {"Romania": 20.11, "Moldova": 26.16, "Kazakhstan": 21.35},  # best-of A2/A3
}
KANTIPUDI = {  # paper Table 5 — A2 (their best)
    "Romania": 18.70, "Moldova": 18.85, "Kazakhstan": 19.62,
}
KANTIPUDI_A1 = {"Romania": 23.83, "Moldova": 24.44, "Kazakhstan": 22.13}
KANTIPUDI_A3 = {"Romania": 20.10, "Moldova": 19.96, "Kazakhstan": 20.81}


def load_all_results() -> pd.DataFrame:
    rows = []
    for mode in MODES:
        f = RES_DIR / f"results_agentic_{mode}.csv"
        if not f.exists():
            print(f"WARN: {f} missing", file=sys.stderr); continue
        d = pd.read_csv(f); d["mode"] = mode; rows.append(d)
    return pd.concat(rows, ignore_index=True)


def load_preds(mode: str, rung: str) -> pd.DataFrame:
    dpath = PRED_DIRS.get(mode)
    if dpath is None or not dpath.exists():
        return pd.DataFrame()
    files = sorted(glob(str(dpath / f"preds_{mode}_{rung}_*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────
# Table 1: Headline cross-country Timika MAE (mean ± std)
# ──────────────────────────────────────────────────────────────────────────
def make_headline_table(df: pd.DataFrame) -> None:
    headline_rows = []
    # Rows: locked baseline, Kantipudi paper, our R1 (per mode), spatcav (per mode), tta (per mode)
    configs = [
        ("Kantipudi A1 (paper)", KANTIPUDI_A1, None),
        ("Kantipudi A2 (paper)", KANTIPUDI, None),
        ("Kantipudi A3 (paper)", KANTIPUDI_A3, None),
    ]
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Cross-country Timika-MAE on the Kantipudi LOCO splits. "
               r"Held-out countries: Romania, Moldova, Kazakhstan. Numbers are "
               r"mean$\pm$std across 5 seeds. The Kantipudi 2024 paper row "
               r"reproduces their reported A1/A2/A3 numbers; our locked baseline "
               r"is our faithful DenseNet121 replication; the remaining rows are "
               r"the proposed agentic ladder. Lower is better; bold = best in column.}")
    out.append(r"\label{tab:headline}")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\resizebox{\linewidth}{!}{%")
    out.append(r"\begin{tabular}{l l c c c c}")
    out.append(r"\toprule")
    out.append(r"Mode & Configuration & Romania & Moldova & Kazakhstan & Mean \\")
    out.append(r"\midrule")
    # Reference rows
    out.append(r"\multicolumn{6}{l}{\textit{Reference (no agentic intervention)}} \\")
    out.append(rf"A1 & Kantipudi 2024 \cite{{kantipudi2024}} & {KANTIPUDI_A1['Romania']:.2f} & {KANTIPUDI_A1['Moldova']:.2f} & {KANTIPUDI_A1['Kazakhstan']:.2f} & "
               rf"{np.mean(list(KANTIPUDI_A1.values())):.2f} \\")
    out.append(rf"A2 & Kantipudi 2024 \cite{{kantipudi2024}} & {KANTIPUDI['Romania']:.2f} & {KANTIPUDI['Moldova']:.2f} & {KANTIPUDI['Kazakhstan']:.2f} & "
               rf"{np.mean(list(KANTIPUDI.values())):.2f} \\")
    out.append(rf"A3 & Kantipudi 2024 \cite{{kantipudi2024}} & {KANTIPUDI_A3['Romania']:.2f} & {KANTIPUDI_A3['Moldova']:.2f} & {KANTIPUDI_A3['Kazakhstan']:.2f} & "
               rf"{np.mean(list(KANTIPUDI_A3.values())):.2f} \\")
    for mode in ["a1","a2","a3"]:
        lb = LOCK_BASELINE[mode]
        out.append(rf"{mode.upper()} & Locked DenseNet baseline & {lb['Romania']:.2f} & {lb['Moldova']:.2f} & {lb['Kazakhstan']:.2f} & "
                   rf"{np.mean(list(lb.values())):.2f} \\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{6}{l}{\textit{Proposed agentic ladder (frozen RAD-DINO features)}} \\")
    # Per-mode best rung (5-seed mean ± std)
    label_map = {
        "rung1_bmc": "R1: BMC head",
        "rung2_tta": "R2: +TTA",
        "rung3_retrieval": "R3: +retrieval",
        "rung4b_spatial_cavity": "R4b: +spatial cavity",
        "rung5_moe": "R5: +severity MoE",
        "rung6_conformal": "R6: +ensemble+conformal",
        "rung7_calibrate_iso": "R7: +isotonic slope cal.",
        "agentic_best": r"\textbf{best (R3+R6)}",
        "agentic_best_tta": r"\textbf{best+TTA}",
        "agentic_best_spatcav": r"\textbf{best+spat-cav}",
        "agentic_v2_iso": r"\textbf{best+isotonic}",
    }
    # gather per-row mean and std per country
    headline_configs = [
        ("a1", "rung1_bmc"),
        ("a1", "rung4b_spatial_cavity"),
        ("a1", "agentic_best_spatcav"),
        ("a2", "rung1_bmc"),
        ("a2", "rung3_retrieval"),
        ("a2", "rung4b_spatial_cavity"),
        ("a2", "agentic_best_spatcav"),
        ("a2", "agentic_best_tta"),
        ("a3", "rung1_bmc"),
        ("a3", "rung3_retrieval"),
        ("a3", "agentic_best"),
        ("a3", "agentic_best_tta"),
        ("fusion", "rung1_bmc"),
        ("fusion", "rung3_retrieval"),
        ("fusion", "rung4b_spatial_cavity"),
        ("fusion", "agentic_best"),
        ("fusion", "agentic_best_tta"),
        ("fusion", "agentic_best_spatcav"),
    ]
    cell = {}
    for mode, rung in headline_configs:
        sub = df[(df["mode"]==mode)&(df["rung"]==rung)]
        if sub.empty: continue
        for c in COUNTRIES:
            ss = sub[sub["held_out"]==c]["timika_mae"]
            if len(ss):
                cell[(mode, rung, c)] = (ss.mean(), ss.std())
    # column minima
    min_per_country = {c: min((m for (mo,rng,cc),(m,s) in cell.items() if cc==c)) for c in COUNTRIES}

    last_mode = None
    for mode, rung in headline_configs:
        if (mode, rung, "Romania") not in cell: continue
        mode_label = mode.upper() if mode != "fusion" else "Fusion"
        if mode != last_mode:
            out.append(r"\midrule")
        cells = []
        for c in COUNTRIES:
            m, s = cell[(mode, rung, c)]
            content = f"{m:.2f}$\\pm${s:.2f}"
            if np.isclose(m, min_per_country[c], atol=0.005):
                content = r"\textbf{" + content + "}"
            cells.append(content)
        mean = np.mean([cell[(mode,rung,c)][0] for c in COUNTRIES])
        out.append(f"{mode_label} & {label_map.get(rung, rung)} & " + " & ".join(cells) + f" & {mean:.2f} \\\\")
        last_mode = mode
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\end{table}")
    (TAB_DIR / "headline_table.tex").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {TAB_DIR/'headline_table.tex'}")


# ──────────────────────────────────────────────────────────────────────────
# Figure 1: Headline MAE bars per country
# ──────────────────────────────────────────────────────────────────────────
def fig_headline_bars(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    bars = [
        ("Locked\nDenseNet", lambda c, _df: LOCK_BASELINE["a2"][c], "#888"),
        ("Kantipudi\nA2", lambda c, _df: KANTIPUDI[c], "#bbb"),
        ("R1 BMC\n(Fusion)", lambda c, df: df[(df["mode"]=="fusion")&(df["rung"]=="rung1_bmc")&(df["held_out"]==c)]["timika_mae"].mean(), "#4c72b0"),
        ("+R3 retrieval\n(Fusion)", lambda c, df: df[(df["mode"]=="fusion")&(df["rung"]=="rung3_retrieval")&(df["held_out"]==c)]["timika_mae"].mean(), "#55a868"),
        ("+R4b spat-cav\n(Fusion)", lambda c, df: df[(df["mode"]=="fusion")&(df["rung"]=="rung4b_spatial_cavity")&(df["held_out"]==c)]["timika_mae"].mean(), "#c44e52"),
        ("Ours\n(R3+R6+spat-cav)", lambda c, df: df[(df["mode"]=="fusion")&(df["rung"]=="agentic_best_spatcav")&(df["held_out"]==c)]["timika_mae"].mean(), "#8172b2"),
    ]
    for ax, c in zip(axes, COUNTRIES):
        names = [b[0] for b in bars]
        vals = [b[1](c, df) for b in bars]
        cols = [b[2] for b in bars]
        x = np.arange(len(names))
        ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.4)
        # Kantipudi reference line
        ax.axhline(KANTIPUDI[c], color="#bbb", linestyle="--", linewidth=0.8, zorder=0)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=6.5)
        n_test = df[(df["mode"]=="fusion")&(df["held_out"]==c)]["n_test"].dropna()
        ax.set_title(f"{c}  (n={int(n_test.mean()) if len(n_test) else 0})")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Timika MAE (lower is better)")
    plt.tight_layout()
    out = FIG_DIR / "fig_headline_bars.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"wrote {out}")


# ──────────────────────────────────────────────────────────────────────────
# Figure 2: Per-rung waterfall (improvement over R1 baseline) — Fusion only
# ──────────────────────────────────────────────────────────────────────────
def fig_waterfall(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    rung_order = [
        ("rung1_bmc", "R1: BMC"),
        ("rung2_tta", "+R2: TTA"),
        ("rung3_retrieval", "+R3: retrieval"),
        ("rung4b_spatial_cavity", "+R4b: spat-cav"),
        ("rung5_moe", "+R5: MoE (null)"),
        ("rung6_conformal", "+R6: conformal"),
        ("rung7_calibrate_iso", "+R7: iso. slope"),
        ("agentic_best_spatcav", "best+spat-cav"),
        ("agentic_best_tta", "best+TTA"),
    ]
    x = np.arange(len(rung_order))
    width = 0.27
    colors = {"Romania": "#4c72b0", "Moldova": "#c44e52", "Kazakhstan": "#55a868"}
    for i, c in enumerate(COUNTRIES):
        vals = [df[(df["mode"]=="fusion")&(df["rung"]==r)&(df["held_out"]==c)]["timika_mae"].mean()
                for r,_ in rung_order]
        ax.bar(x + (i-1)*width, vals, width, label=c, color=colors[c], edgecolor="black", linewidth=0.3)
    ax.axhline(KANTIPUDI["Romania"], color="#888", linestyle=":", linewidth=0.7, label="Kantipudi Rom (18.70)")
    ax.axhline(KANTIPUDI["Moldova"], color="#c44e52", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels([n for _,n in rung_order], rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("Timika MAE (Fusion mode)")
    ax.set_ylim(15, 25)
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "fig_waterfall.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"wrote {out}")


# ──────────────────────────────────────────────────────────────────────────
# Figure 3: Pearson r heatmap (mode × country) for the best rung
# ──────────────────────────────────────────────────────────────────────────
def fig_pearson(df: pd.DataFrame) -> None:
    rungs = ["rung1_mse", "rung1_bmc", "rung3_retrieval", "rung4b_spatial_cavity",
             "rung6_conformal", "agentic_best", "agentic_best_spatcav", "agentic_best_tta"]
    rung_label = {
        "rung1_mse": "R1 (MSE)", "rung1_bmc": "R1 (BMC)", "rung3_retrieval": "+R3 retr.",
        "rung4b_spatial_cavity": "+R4b spat-cav", "rung6_conformal": "+R6 conf.",
        "agentic_best": "best", "agentic_best_spatcav": "best+spat-cav", "agentic_best_tta": "best+TTA",
    }
    fig, axes = plt.subplots(1, len(COUNTRIES), figsize=(7.2, 2.5), sharey=True)
    modes_ord = ["a1", "a2", "a3", "fusion"]
    for ax, c in zip(axes, COUNTRIES):
        mat = np.full((len(modes_ord), len(rungs)), np.nan)
        for i, m in enumerate(modes_ord):
            for j, r in enumerate(rungs):
                ss = df[(df["mode"]==m)&(df["rung"]==r)&(df["held_out"]==c)]["timika_pearson"]
                if len(ss): mat[i,j] = ss.mean()
        im = ax.imshow(mat, vmin=0.55, vmax=0.85, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(rungs))); ax.set_xticklabels([rung_label[r] for r in rungs], rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(modes_ord))); ax.set_yticklabels([m.upper() if m!="fusion" else "Fusion" for m in modes_ord])
        ax.set_title(c)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i,j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if mat[i,j] < 0.7 else "black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Pearson r")
    out = FIG_DIR / "fig_pearson.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"wrote {out}")


# ──────────────────────────────────────────────────────────────────────────
# Figure 4: Per-severity-band MAE (binned analysis)
# ──────────────────────────────────────────────────────────────────────────
def per_severity_table_and_fig(df: pd.DataFrame) -> None:
    """Use a2 preds (we have them) and compare R1 vs best_spatcav vs best_tta."""
    rungs = ["rung1_mse", "agentic_best_spatcav", "agentic_best_tta"]
    bands = [(0,30),(30,60),(60,100),(100,140)]
    band_lbl = ["mild [0,30)", "mod [30,60)", "sev [60,100)", "v.sev [100,140]"]

    rows = []
    for mode in ["a1","a2","a3"]:
        for rung in rungs:
            d = load_preds(mode, rung)
            if d.empty: continue
            for c in COUNTRIES:
                ssc = d[d["held_out"]==c]
                if ssc.empty: continue
                for (lo,hi), bl in zip(bands, band_lbl):
                    mask = (ssc["timika_true"] >= lo) & (ssc["timika_true"] < hi if hi<140 else ssc["timika_true"] <= hi)
                    if mask.sum()==0: continue
                    mae = float(np.mean(np.abs(ssc.loc[mask, "timika_pred"] - ssc.loc[mask, "timika_true"])))
                    rows.append({"mode": mode, "rung": rung, "held_out": c, "band": bl, "n": int(mask.sum()), "mae": mae})
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        print("WARN no per-severity preds available"); return

    # Save table CSV
    tbl.to_csv(TAB_DIR / "per_severity_mae.csv", index=False)

    # Fig: bars grouped by band for A2 only (we have it)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, c in zip(axes, COUNTRIES):
        sub = tbl[(tbl["mode"]=="a2")&(tbl["held_out"]==c)]
        if sub.empty: ax.set_visible(False); continue
        x = np.arange(len(band_lbl))
        width = 0.28
        for i, rung in enumerate(rungs):
            ssr = sub[sub["rung"]==rung].set_index("band").reindex(band_lbl)
            vals = ssr["mae"].values
            ax.bar(x + (i-1)*width, vals, width, label={"rung1_mse":"R1","agentic_best_spatcav":"best+spat-cav","agentic_best_tta":"best+TTA"}[rung],
                   edgecolor="black", linewidth=0.3)
            # n counts
            ns = ssr["n"].fillna(0).astype(int).values
            for xi, v, n in zip(x + (i-1)*width, vals, ns):
                if not np.isnan(v):
                    ax.text(xi, v + 0.4, f"n={n}", ha="center", va="bottom", fontsize=5.5, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(["mild","mod","sev","v.sev"], fontsize=8)
        ax.set_title(c)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if c == COUNTRIES[0]:
            ax.set_ylabel("Timika MAE")
            ax.legend(loc="upper left", frameon=False, fontsize=7)
    plt.tight_layout()
    out = FIG_DIR / "fig_per_severity.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"wrote {out}")


# ──────────────────────────────────────────────────────────────────────────
# Figure 5: Prediction-vs-truth scatter
# ──────────────────────────────────────────────────────────────────────────
def fig_scatter(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    for ax, c in zip(axes, COUNTRIES):
        d = load_preds("a2", "agentic_best_spatcav")
        if d.empty: continue
        sub = d[d["held_out"]==c]
        ax.scatter(sub["timika_true"], sub["timika_pred"], s=4, alpha=0.35, color="#4c72b0", edgecolors="none")
        ax.plot([0,140],[0,140], color="black", linewidth=0.8, linestyle="--")
        # slope from linear regression
        if len(sub) > 5:
            slope, intercept = np.polyfit(sub["timika_true"], sub["timika_pred"], 1)
            xs = np.array([0, 140])
            ax.plot(xs, slope*xs + intercept, color="#c44e52", linewidth=1.0)
            ax.text(0.04, 0.94, f"slope={slope:.2f}\nintercept={intercept:.1f}\nN={len(sub)}",
                    transform=ax.transAxes, va="top", fontsize=7, family="monospace")
        ax.set_xlim(0, 140); ax.set_ylim(0, 140)
        ax.set_title(c)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Predicted Timika")
    fig.text(0.5, -0.02, "True Timika", ha="center")
    plt.tight_layout()
    out = FIG_DIR / "fig_scatter.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"wrote {out}")


# ──────────────────────────────────────────────────────────────────────────
# Figure 6: Cavity reliability diagram
# ──────────────────────────────────────────────────────────────────────────
def fig_reliability(df: pd.DataFrame) -> None:
    """Use cavity_auc / cavity_f1 from results CSV; reliability needs per-image probs.
    We approximate by plotting bar chart of cavity AUC per (mode, head, country).
    """
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    configs = [
        ("a2", "rung1_mse", "Global CLS"),
        ("a2", "rung4b_spatial_cavity", "Spatial cav."),
    ]
    x = np.arange(len(COUNTRIES))
    width = 0.35
    for i, (mode, rung, lbl) in enumerate(configs):
        vals = [df[(df["mode"]==mode)&(df["rung"]==rung)&(df["held_out"]==c)]["cavity_auc"].mean() for c in COUNTRIES]
        ax.bar(x + (i-0.5)*width, vals, width, label=lbl,
               color=["#888","#4c72b0"][i], edgecolor="black", linewidth=0.3)
        for xi, v in zip(x + (i-0.5)*width, vals):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(COUNTRIES)
    ax.set_ylabel("Cavity AUC")
    ax.set_ylim(0.6, 0.95)
    ax.set_title("Cavity head: global CLS vs. spatial attention (A2)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "fig_cavity_auc.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"wrote {out}")


# ──────────────────────────────────────────────────────────────────────────
# Figure 7: Conformal coverage decomposition
# ──────────────────────────────────────────────────────────────────────────
def fig_conformal(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, metric, ylim, ylbl in [(ax1, "conformal_cov", (0.7, 1.0), "Empirical coverage (target 0.90)"),
                                    (ax2, "conformal_width", (60, 110), "Interval width (Timika points)")]:
        x = np.arange(len(COUNTRIES))
        width = 0.22
        rungs = [("rung6_conformal", "R6 conf.", "#4c72b0"),
                 ("agentic_best", "best", "#55a868"),
                 ("agentic_best_spatcav", "best+spat-cav", "#c44e52"),
                 ("agentic_best_tta", "best+TTA", "#8172b2")]
        for i, (r, lbl, col) in enumerate(rungs):
            vals = [df[(df["mode"]=="fusion")&(df["rung"]==r)&(df["held_out"]==c)][metric].mean() for c in COUNTRIES]
            ax.bar(x + (i-1.5)*width, vals, width, label=lbl, color=col, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels(COUNTRIES)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylbl)
        if metric == "conformal_cov":
            ax.axhline(0.90, color="black", linestyle="--", linewidth=0.7, label="nominal 0.90")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if metric == "conformal_cov":
            ax.legend(loc="lower right", frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()
    out = FIG_DIR / "fig_conformal.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"wrote {out}")


# ──────────────────────────────────────────────────────────────────────────
# Figure 8: Slope calibration (R7 effect)
# ──────────────────────────────────────────────────────────────────────────
def fig_slope_cal(df: pd.DataFrame) -> None:
    """Compute regression slope for predictions before and after R7."""
    rows = []
    for mode in ["a1","a2","a3"]:
        for rung in ["rung1_mse", "rung6_conformal", "agentic_best", "agentic_best_spatcav", "agentic_best_tta", "agentic_v2_iso"]:
            d = load_preds(mode, rung)
            if d.empty: continue
            for c in COUNTRIES:
                sub = d[d["held_out"]==c]
                if len(sub) < 10: continue
                slope, intercept = np.polyfit(sub["timika_true"], sub["timika_pred"], 1)
                rows.append({"mode":mode, "rung":rung, "country":c, "slope":slope, "intercept":intercept})
    sdf = pd.DataFrame(rows)
    if sdf.empty: print("no slope data"); return
    sdf.to_csv(TAB_DIR / "slope_calibration.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    x = np.arange(len(COUNTRIES))
    rungs = [("rung1_mse", "R1", "#888"),
             ("rung6_conformal", "R6", "#4c72b0"),
             ("agentic_best_spatcav", "best+spat-cav", "#c44e52"),
             ("agentic_v2_iso", "+isotonic (R7)", "#55a868")]
    width = 0.22
    for i, (r, lbl, col) in enumerate(rungs):
        vals = []
        for c in COUNTRIES:
            ss = sdf[(sdf["mode"]=="a2")&(sdf["rung"]==r)&(sdf["country"]==c)]["slope"]
            vals.append(ss.mean() if len(ss) else np.nan)
        ax.bar(x + (i-1.5)*width, vals, width, label=lbl, color=col, edgecolor="black", linewidth=0.3)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(COUNTRIES)
    ax.set_ylabel("Regression slope (pred vs. true)")
    ax.set_title("Slope→1.0 ⇒ no compression of severity range (A2)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "fig_slope_cal.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"wrote {out}")


# ──────────────────────────────────────────────────────────────────────────
# Per-rung ablation table (extended)
# ──────────────────────────────────────────────────────────────────────────
def make_per_mode_ablation_tables(df: pd.DataFrame) -> None:
    """Per-mode ablation tables (one per mode), narrow enough for single-column LNCS."""
    rung_order = [
        ("rung1_mse", "R1 (MSE)"),
        ("rung1_bmc", "R1 (BMC)"),
        ("rung2_tta", "+R2 TTA"),
        ("rung3_retrieval", "+R3 retrieval"),
        ("rung4b_spatial_cavity", "+R4b spat-cav"),
        ("rung5_moe", "+R5 sev-MoE"),
        ("rung6_conformal", "+R6 conf+ens"),
        ("rung7_calibrate_iso", "+R7 isotonic"),
        ("agentic_best", r"\textit{best (R3+R6)}"),
        ("agentic_best_spatcav", r"\textit{best+spat-cav}"),
        ("agentic_best_tta", r"\textit{best+TTA}"),
    ]
    for m in MODES:
        out = []
        mode_label = m.upper() if m != "fusion" else "Fusion"
        out.append(r"\begin{table}[t]")
        out.append(r"\centering\small")
        out.append(rf"\caption{{{mode_label}-mode per-rung ablation: Timika MAE (mean$\\pm$std, 5 seeds). "
                   r"Lower is better; bold = best in column.}}")
        out.append(rf"\label{{tab:abl_{m}}}")
        out.append(r"\setlength{\tabcolsep}{4pt}")
        out.append(r"\begin{tabular}{l c c c}")
        out.append(r"\toprule")
        out.append(r"Configuration & Romania & Moldova & Kazakhstan \\")
        out.append(r"\midrule")
        # column min
        col_min = {c: min((df[(df["mode"]==m)&(df["rung"]==r)&(df["held_out"]==c)]["timika_mae"].mean()
                            for r,_ in rung_order
                            if not df[(df["mode"]==m)&(df["rung"]==r)&(df["held_out"]==c)].empty),
                          default=np.nan) for c in COUNTRIES}
        for r, lbl in rung_order:
            cells = [lbl]
            for c in COUNTRIES:
                ss = df[(df["mode"]==m)&(df["rung"]==r)&(df["held_out"]==c)]["timika_mae"]
                if not len(ss):
                    cells.append("--"); continue
                mn, sd = ss.mean(), ss.std()
                content = f"{mn:.2f}$\\pm${sd:.2f}"
                if not np.isnan(col_min[c]) and np.isclose(mn, col_min[c], atol=0.005):
                    content = r"\textbf{" + content + "}"
                cells.append(content)
            out.append(" & ".join(cells) + r" \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\end{table}")
        (TAB_DIR / f"ablation_{m}.tex").write_text("\n".join(out), encoding="utf-8")
        print(f"wrote {TAB_DIR/f'ablation_{m}.tex'}")


def make_ablation_table(df: pd.DataFrame) -> None:
    rung_order = [
        ("rung1_mse", "R1 (MSE)"),
        ("rung1_bmc", "R1 (BMC)"),
        ("rung2_tta", "+R2 TTA"),
        ("rung3_retrieval", "+R3 retrieval"),
        ("rung4b_spatial_cavity", "+R4b spat-cav"),
        ("rung5_moe", "+R5 sev-MoE"),
        ("rung6_conformal", "+R6 conf+ens"),
        ("rung7_calibrate_iso", "+R7 isotonic"),
        ("agentic_best", r"\textit{best (R3+R6)}"),
        ("agentic_best_spatcav", r"\textit{best+spat-cav}"),
        ("agentic_best_tta", r"\textit{best+TTA}"),
    ]
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Per-rung ablation: Timika MAE (mean$\pm$std across 5 seeds). "
               r"R5 is included to document an honest \emph{null} result; "
               r"R2 helps Moldova specifically (covariate shift) and hurts Romania/Kazakhstan, "
               r"so it appears in `best+TTA' rather than in `best'.}")
    out.append(r"\label{tab:ablation}")
    out.append(r"\setlength{\tabcolsep}{3pt}")
    out.append(r"\small")
    out.append(r"\resizebox{\linewidth}{!}{%")
    out.append(r"\begin{tabular}{l " + " c c c " * len(MODES) + "}")
    out.append(r"\toprule")
    # Header
    hdr = ["Configuration"]
    for m in MODES:
        label = m.upper() if m != "fusion" else "Fusion"
        hdr += [rf"\multicolumn{{3}}{{c}}{{{label}}}"]
    out.append(" & ".join(hdr) + r" \\")
    out.append(" " + " & ".join([""] + ["Rom & Mol & Kaz"]*len(MODES)) + r" \\")
    out.append(r"\midrule")
    for r, lbl in rung_order:
        cells = [lbl]
        for m in MODES:
            for c in COUNTRIES:
                ss = df[(df["mode"]==m)&(df["rung"]==r)&(df["held_out"]==c)]["timika_mae"]
                if len(ss):
                    cells.append(f"{ss.mean():.2f}\\scriptsize$\\pm${ss.std():.2f}")
                else:
                    cells.append("--")
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\end{table}")
    (TAB_DIR / "ablation_table.tex").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {TAB_DIR/'ablation_table.tex'}")


# ──────────────────────────────────────────────────────────────────────────
# Paired bootstrap significance table (vs locked baseline = our R1 MSE)
# ──────────────────────────────────────────────────────────────────────────
def bonferroni_holm(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return boolean significance mask after Bonferroni-Holm step-down at family-wise alpha."""
    n = len(p_values)
    order = np.argsort(p_values)
    sig = np.zeros(n, dtype=bool)
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if p_values[idx] <= threshold:
            sig[idx] = True
        else:
            break
    return sig


def fig_significance_forest(df: pd.DataFrame) -> None:
    """Per-rung forest plot of Delta-MAE with 95% CIs across countries."""
    bt = pd.read_csv(TAB_DIR / "paired_bootstrap.csv")
    bt_a2 = bt[bt["mode"] == "a2"].copy()
    if bt_a2.empty:
        return

    rungs_order = ["rung1_bmc", "rung2_tta", "rung3_retrieval", "rung4b_spatial_cavity",
                   "rung6_conformal", "agentic_best", "agentic_best_spatcav", "agentic_best_tta"]
    rung_label = {
        "rung1_bmc": "R1 BMC", "rung2_tta": "+R2 TTA",
        "rung3_retrieval": "+R3 retr.", "rung4b_spatial_cavity": "+R4b spat-cav",
        "rung6_conformal": "+R6 conf.", "agentic_best": "best (R3+R6)",
        "agentic_best_spatcav": "best+spat-cav", "agentic_best_tta": "best+TTA",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=True)
    for ax, c in zip(axes, COUNTRIES):
        sub = bt_a2[bt_a2["country"] == c].set_index("rung")
        # one-sided p-value approximation from bootstrap fraction
        rungs_here = [r for r in rungs_order if r in sub.index]
        # use frac_a_better column if present, else compute from CI
        # paired bootstrap CSV from main script doesn't include p-value; compute approx via CI sign
        # We approximate: assume normality, std ~ (hi-lo)/3.92, p = 2*sf(|delta/std|)
        from scipy.stats import norm
        deltas = sub.loc[rungs_here, "delta"].values
        lo = sub.loc[rungs_here, "ci_lo"].values
        hi = sub.loc[rungs_here, "ci_hi"].values
        se = (hi - lo) / 3.92
        p = 2 * norm.sf(np.abs(deltas) / np.maximum(se, 1e-9))
        sig_mask = bonferroni_holm(p, alpha=0.05)

        y_positions = np.arange(len(rungs_here))[::-1]
        for i, (yi, d, l, h, m) in enumerate(zip(y_positions, deltas, lo, hi, sig_mask)):
            colour = "#2a8b34" if d < 0 and m else ("#a73229" if d > 0 and m else "#999")
            ax.plot([l, h], [yi, yi], color=colour, linewidth=2.2)
            ax.plot([d], [yi], "o", color=colour, markersize=4)
            if m:
                ax.text(h + 0.06, yi, "*", color=colour, va="center", fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([rung_label[r] for r in rungs_here], fontsize=8)
        ax.set_title(f"{c}  (n={sub['n'].iloc[0]:.0f})")
        ax.set_xlabel(r"$\Delta$ Timika MAE vs R1 MSE")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Paired-bootstrap $\\Delta$-MAE with Bonferroni-Holm-corrected significance (DH mode)",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig_significance_forest.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"wrote {out}")


def paired_bootstrap_table(df: pd.DataFrame) -> None:
    """Compute Δ-MAE vs R1 MSE with paired-bootstrap 95% CI, pairing on image_id."""
    rng = np.random.default_rng(1337)
    rows = []
    for mode in ["a1","a2","a3"]:
        ref = load_preds(mode, "rung1_mse")
        if ref.empty: continue
        for rung in ["rung1_bmc", "rung2_tta", "rung3_retrieval", "rung4b_spatial_cavity",
                     "rung6_conformal", "agentic_best", "agentic_best_spatcav", "agentic_best_tta"]:
            cur = load_preds(mode, rung)
            if cur.empty: continue
            for c in COUNTRIES:
                r_c = ref[ref["held_out"]==c].groupby("image_id")[["timika_true","timika_pred"]].mean()
                a_c = cur[cur["held_out"]==c].groupby("image_id")[["timika_true","timika_pred"]].mean()
                common = r_c.index.intersection(a_c.index)
                if len(common) < 10: continue
                y_true = r_c.loc[common, "timika_true"].to_numpy()
                y_b    = r_c.loc[common, "timika_pred"].to_numpy()
                y_a    = a_c.loc[common, "timika_pred"].to_numpy()
                point = np.mean(np.abs(y_a - y_true)) - np.mean(np.abs(y_b - y_true))
                deltas = []
                for _ in range(2000):
                    idx = rng.integers(0, len(common), size=len(common))
                    deltas.append(np.mean(np.abs(y_a[idx]-y_true[idx])) - np.mean(np.abs(y_b[idx]-y_true[idx])))
                lo, hi = np.percentile(deltas, [2.5, 97.5])
                sig = "*" if (hi < 0 or lo > 0) else ""
                rows.append({"mode":mode, "rung":rung, "country":c, "n":len(common),
                             "delta":point, "ci_lo":lo, "ci_hi":hi, "sig":sig})
    bt = pd.DataFrame(rows)
    bt.to_csv(TAB_DIR / "paired_bootstrap.csv", index=False)
    print(f"wrote {TAB_DIR/'paired_bootstrap.csv'} (n={len(bt)} rows)")
    # tex
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering\small")
    out.append(r"\caption{Paired-bootstrap $\Delta$-MAE (2000 reps, paired on image\_id) "
               r"vs.\ R1 MSE on A2 and A3. Negative $\Delta$ = the rung beats R1. "
               r"95\% CIs that exclude zero are marked $\ast$.}")
    out.append(r"\label{tab:bootstrap}")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\begin{tabular}{l l l r r c}")
    out.append(r"\toprule")
    out.append(r"Mode & Country & Rung & $\Delta$MAE & 95\% CI & sig \\")
    out.append(r"\midrule")
    for mode_filter in ["a2", "a3"]:
        for c in COUNTRIES:
            for _, row in bt[(bt["mode"]==mode_filter)&(bt["country"]==c)].iterrows():
                sig_str = r"$\ast$" if row["sig"] == "*" else ""
                out.append(f"{mode_filter.upper()} & {c} & {row['rung'].replace('_', chr(92)+'_')} & {row['delta']:+.3f} & "
                           f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}] & {sig_str} \\\\")
        out.append(r"\midrule")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    (TAB_DIR / "paired_bootstrap.tex").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {TAB_DIR/'paired_bootstrap.tex'}")


# ──────────────────────────────────────────────────────────────────────────
def main():
    df = load_all_results()
    print(f"Loaded {len(df)} result rows across {df['mode'].nunique()} modes")
    make_headline_table(df)
    make_ablation_table(df)
    make_per_mode_ablation_tables(df)
    fig_headline_bars(df)
    fig_waterfall(df)
    fig_pearson(df)
    fig_scatter(df)
    fig_reliability(df)
    fig_conformal(df)
    fig_slope_cal(df)
    per_severity_table_and_fig(df)
    paired_bootstrap_table(df)
    fig_significance_forest(df)


if __name__ == "__main__":
    main()
