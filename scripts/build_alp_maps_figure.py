"""Build a 3x2 composite figure of ALP spatial maps for the paper.

Output: iconips_Paper/figures/fig_alp_spatial.pdf (+ .png)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "iconips_Paper" / "figures" / "alp_maps"
OUT = ROOT / "iconips_Paper" / "figures"

meta = pd.read_csv(SRC / "meta.csv")
meta["fname"] = meta["png"].apply(lambda p: Path(p).name)

countries = ["Romania", "Moldova", "Kazakhstan"]

fig, axes = plt.subplots(3, 2, figsize=(5.4, 7.4))
for r, country in enumerate(countries):
    sub = meta[meta["country"] == country]
    mild   = sub[sub["tag"] == "mild"].iloc[0]
    severe = sub[sub["tag"] == "severe"].iloc[0]
    for c, item in enumerate([mild, severe]):
        ax = axes[r, c]
        img = mpimg.imread(SRC / item.fname)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{country} {item.tag}: y={item.gt_alp:.0f}, $\\hat y$={item.pred_alp:.1f}",
                     fontsize=8.2)

plt.tight_layout(pad=0.4)
out_pdf = OUT / "fig_alp_spatial.pdf"
out_png = OUT / "fig_alp_spatial.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight", dpi=160)
plt.close()
print("wrote", out_pdf)
print("wrote", out_png)
