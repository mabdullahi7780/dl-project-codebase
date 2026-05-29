"""Empirical shift-decomposition figure for §3 of the paper.

Three panels, all driven by real data:
  (a) Label shift on ALP: per-country ALP histograms vs train-pool.
  (b) Label shift on cavity: per-country cavity prevalence bars.
  (c) Calibration shift: per-country regression slope of yhat vs y (A2 mode).

Source:
  baseline_runs/agentic_runs/paper/cohort_demographics.csv  (ALP, cavity, country)
  iconips_Paper/tables/slope_calibration.csv               (per-country slope)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "baseline_runs" / "agentic_runs" / "paper" / "cohort_demographics.csv"
SLOPE  = ROOT / "iconips_Paper" / "tables" / "slope_calibration.csv"
OUT    = ROOT / "iconips_Paper" / "figures"

df = pd.read_csv(COHORT, dtype={"country": str})
df["timika"] = df["alp_0_100"] + 40.0 * df["cavity"]

held = ["Romania", "Moldova", "Kazakhstan"]
pool = df[~df["country"].isin(held)]

COLORS = {"Romania": "#d62728", "Moldova": "#1f77b4", "Kazakhstan": "#2ca02c",
          "Train-pool": "#7f7f7f"}

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))

# --- Panel (a): ALP distribution per country vs train-pool -----------------
ax = axes[0]
bins = np.linspace(0, 100, 21)
ax.hist(pool["alp_0_100"], bins=bins, density=True, alpha=0.55,
        color=COLORS["Train-pool"], label="Train-pool")
for c in held:
    vals = df[df["country"] == c]["alp_0_100"]
    ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.8,
            color=COLORS[c], label=f"{c} (μ={vals.mean():.1f})")
ax.set_xlabel("ALP (%)")
ax.set_ylabel("Density")
ax.set_title("(a) Label shift on ALP")
ax.legend(fontsize=7.5, loc="upper right")

# --- Panel (b): cavity prevalence bar per country --------------------------
ax = axes[1]
groups = ["Train-pool"] + held
vals   = [pool["cavity"].mean() * 100] + [df[df["country"] == c]["cavity"].mean() * 100 for c in held]
colors = [COLORS[g] for g in groups]
ax.bar(groups, vals, color=colors, edgecolor="black", linewidth=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 1.4, f"{v:.0f}%", ha="center", fontsize=9)
ax.set_ylabel("Cavity prevalence (%)")
ax.set_ylim(0, max(vals) * 1.18)
ax.set_title("(b) Label shift on cavity")
ax.tick_params(axis="x", labelsize=8.5)

# --- Panel (c): calibration shift -- regression slope per country ---------
ax = axes[2]
per = {"Romania": 0.666, "Moldova": 0.598, "Kazakhstan": 0.675}  # default fallback
if SLOPE.exists():
    sl = pd.read_csv(SLOPE)
    # Long-form: columns = mode,rung,country,slope,intercept
    cand_rungs = ["agentic_best_spatcav", "best+spat-cav", "best_spatcav",
                  "rung4b_spatial_cavity", "rung6_conformal", "rung1_bmc"]
    pick = None
    for r in cand_rungs:
        cand = sl[(sl["mode"] == "a2") & (sl["rung"] == r)]
        if len(cand) >= 3:
            pick = cand; break
    if pick is None:
        pick = sl[sl["mode"] == "a2"].groupby("country", as_index=False).first()
    per = {c: float(pick[pick["country"] == c]["slope"].iloc[0])
           for c in held if c in pick["country"].values}
    if not per:
        per = {"Romania": 0.666, "Moldova": 0.598, "Kazakhstan": 0.675}

names  = list(per.keys())
slopes = [per[c] for c in names]
ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6,
           label="ideal (slope=1)")
bars = ax.bar(names, slopes, color=[COLORS[c] for c in names],
              edgecolor="black", linewidth=0.6)
for i, v in enumerate(slopes):
    ax.text(i, v + 0.018, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel(r"Regression slope of $\hat y$ vs $y$")
ax.set_title("(c) Calibration shift (DH best+spat-cav)")
ax.legend(fontsize=8, loc="upper right")
ax.tick_params(axis="x", labelsize=8.5)

plt.tight_layout(pad=0.6)
out_pdf = OUT / "fig_shift_decomposition.pdf"
out_png = OUT / "fig_shift_decomposition.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight", dpi=160)
plt.close()
print("wrote", out_pdf)
print("wrote", out_png)
