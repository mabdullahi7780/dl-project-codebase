"""Single-panel ALP-density shift figure (v3 for paper_v2 revision).

Writes:
  paper_v2/_v2_extract/ICONIP2026_G1_ChestXRays 2/fig_shift_decomposition.pdf
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "baseline_runs" / "agentic_runs" / "paper" / "cohort_demographics.csv"
OUT_DIR = ROOT / "paper_v2" / "_v2_extract" / "ICONIP2026_G1_ChestXRays 2"

df = pd.read_csv(COHORT, dtype={"country": str})

held = ["Romania", "Moldova", "Kazakhstan"]
pool = df[~df["country"].isin(held)]

COLORS = {"Romania": "#d62728", "Moldova": "#1f77b4", "Kazakhstan": "#2ca02c",
          "Train-pool": "#7f7f7f"}

fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.4))
bins = np.linspace(0, 100, 21)
ax.hist(pool["alp_0_100"], bins=bins, density=True, alpha=0.55,
        color=COLORS["Train-pool"], label=f"Train-pool ($\\mu$={pool['alp_0_100'].mean():.1f})")
for c in held:
    vals = df[df["country"] == c]["alp_0_100"]
    ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=2.0,
            color=COLORS[c], label=f"{c} ($\\mu$={vals.mean():.1f})")
ax.set_xlabel("ALP (%)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_xlim(0, 100)
ax.legend(fontsize=9.5, loc="upper right", framealpha=0.92)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(pad=0.4)
out_pdf = OUT_DIR / "fig_shift_decomposition.pdf"
out_png = OUT_DIR / "fig_shift_decomposition.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight", dpi=160)
plt.close()
print("wrote", out_pdf)
print("wrote", out_png)
