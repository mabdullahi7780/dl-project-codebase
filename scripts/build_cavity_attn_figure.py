"""Build a single 3x3 composite figure of cavity-attention overlays for the paper.

Output: iconips_Paper/figures/fig_cavity_attention.pdf (+ .png)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "iconips_Paper" / "figures" / "cavity_attn"
OUT = ROOT / "iconips_Paper" / "figures"

meta = pd.read_csv(SRC / "meta.csv")
meta["fname"] = meta["png"].apply(lambda p: Path(p).name)

countries = ["Romania", "Moldova", "Kazakhstan"]

# For each country: pick 2 positives + 1 negative, in that column order
fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.2))
for r, country in enumerate(countries):
    sub = meta[meta["country"] == country]
    pos = sub[sub["tag"] == "pos"].iloc[:2]
    neg = sub[sub["tag"] == "neg"].iloc[:1]
    row_items = list(pos.itertuples()) + list(neg.itertuples())
    for c, item in enumerate(row_items):
        ax = axes[r, c]
        img = mpimg.imread(SRC / item.fname)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        tag_label = "cavity+" if item.tag == "pos" else "cavity-"
        title = f"{country} {tag_label}: p={item.pred_prob:.2f}"
        ax.set_title(title, fontsize=8.2)

plt.tight_layout(pad=0.4)
out_pdf = OUT / "fig_cavity_attention.pdf"
out_png = OUT / "fig_cavity_attention.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight", dpi=160)
plt.close()
print("wrote", out_pdf)
print("wrote", out_png)
