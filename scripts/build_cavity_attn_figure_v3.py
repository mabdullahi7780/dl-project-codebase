"""Clean 3x3 cavity-attention composite for the paper_v2 revision.

Improvements over v1:
- Crops the raw PNGs to remove the embedded matplotlib titles
  (they were redundant with the new figure titles)
- Tighter inter-panel spacing
- Square aspect ratio so it doesn't get stretched at \\linewidth

Writes:
  paper_v2/_v2_extract/ICONIP2026_G1_ChestXRays 2/fig_cavity_attention.pdf
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "iconips_Paper" / "figures" / "cavity_attn"
OUT_DIR = ROOT / "paper_v2" / "_v2_extract" / "ICONIP2026_G1_ChestXRays 2"

meta = pd.read_csv(SRC / "meta.csv")
meta["fname"] = meta["png"].apply(lambda p: Path(p).name)

countries = ["Romania", "Moldova", "Kazakhstan"]


def crop_title(img):
    """Strip the embedded matplotlib title bar at the top of each raw panel.

    The cavity_attn PNGs were generated with ax.set_title(); the resulting
    image has a ~9% blank/text band at the top. We crop it.
    """
    h = img.shape[0]
    return img[int(0.10 * h):, :, :]


fig, axes = plt.subplots(3, 3, figsize=(6.5, 6.5))
for r, country in enumerate(countries):
    sub = meta[meta["country"] == country]
    pos = sub[sub["tag"] == "pos"].iloc[:2]
    neg = sub[sub["tag"] == "neg"].iloc[:1]
    row_items = list(pos.itertuples()) + list(neg.itertuples())
    for c, item in enumerate(row_items):
        ax = axes[r, c]
        img = mpimg.imread(SRC / item.fname)
        img = crop_title(img)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        tag_label = "cav+" if item.tag == "pos" else "cav-"
        ax.set_title(f"{country} {tag_label}: $p={item.pred_prob:.2f}$",
                     fontsize=8.5, pad=3)

plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01,
                    wspace=0.06, hspace=0.20)
out_pdf = OUT_DIR / "fig_cavity_attention.pdf"
out_png = OUT_DIR / "fig_cavity_attention.png"
plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
plt.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=160)
plt.close()
print("wrote", out_pdf)
print("wrote", out_png)
