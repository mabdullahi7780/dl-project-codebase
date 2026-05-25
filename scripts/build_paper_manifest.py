"""Subsample the TB Portals manifest to Kantipudi et al. (JIIM 2024) Table 1.

The paper uses the January 2023 TB Portals release: exactly 5,010 annotated
CXRs across 8 countries (Table 1). That specific release cannot be recovered
from the August 2023 archive, but the August 2023 annotated manifest contains
enough images in every (country, cavity-class) cell to subsample down to
Table 1's exact counts. This reproduces the paper's data *distribution*
deterministically (fixed seed), which is what matters for replicating the
country-segregated results.

Usage::

    python scripts/build_paper_manifest.py \
        --in  local_work/kaggle_export/manifest.csv \
        --out local_work/data/processed/tbportals_manifest_paper.csv

The same ``subsample()`` function is imported by the Kaggle notebook so the
manifest is built identically on Kaggle from the uploaded PNG dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Kantipudi et al. (JIIM 2024) Table 1 -> (n_cavity, n_no_cavity) per country.
# Sums to exactly 5,010 images.
TABLE1: dict[str, tuple[int, int]] = {
    "Georgia":    (713, 701),
    "Belarus":    (254, 798),
    "Ukraine":    (500, 816),
    "Kazakhstan": (159, 240),
    "Romania":    (143, 77),
    "Moldova":    (193, 396),
    "Azerbaijan": (5, 12),
    "India":      (0, 3),
}
PAPER_TOTAL = sum(c + n for c, n in TABLE1.values())  # 5010


def subsample(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return a copy of ``df`` subsampled to Table 1's exact per-cell counts.

    Sampling is at the image level with a fixed RNG, so the 5,010-image set is
    deterministic. Patient-disjoint train/val/test splits are handled later by
    ``make_country_split`` / ``make_balanced_cavity_split`` at training time.
    """
    df = df.copy()
    df["country"] = df["country"].astype(str).str.strip()
    df["cavity"] = df["cavity"].astype(int)

    rng = np.random.default_rng(seed)
    keep_idx: list = []

    print(f"{'country':12s} {'need_c':>7s} {'have_c':>7s} {'need_n':>7s} {'have_n':>7s}")
    for country, (n_cav, n_nocav) in TABLE1.items():
        sub = df[df["country"] == country]
        cav = sub[sub["cavity"] == 1]
        nocav = sub[sub["cavity"] == 0]
        print(f"{country:12s} {n_cav:7d} {len(cav):7d} {n_nocav:7d} {len(nocav):7d}")
        if len(cav) < n_cav or len(nocav) < n_nocav:
            raise SystemExit(
                f"[build_paper_manifest] Not enough images for {country}: "
                f"need {n_cav} cavity / {n_nocav} no-cavity, "
                f"have {len(cav)} / {len(nocav)}."
            )
        if n_cav > 0:
            keep_idx += rng.choice(cav.index.values, size=n_cav, replace=False).tolist()
        if n_nocav > 0:
            keep_idx += rng.choice(nocav.index.values, size=n_nocav, replace=False).tolist()

    out = df.loc[keep_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Subsample manifest to Kantipudi Table 1 (5010).")
    p.add_argument("--in", dest="inp", required=True, help="Input manifest CSV (Aug2023 annotated).")
    p.add_argument("--out", required=True, help="Output paper manifest CSV.")
    p.add_argument("--seed", type=int, default=42, help="Fixed seed for the 5010-image set.")
    args = p.parse_args(argv)

    df = pd.read_csv(args.inp, dtype={"image_id": str, "patient_id": str, "country": str})
    out = subsample(df, seed=args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\n[build_paper_manifest] wrote {len(out)} images -> {args.out}")
    print(f"[build_paper_manifest] Table 1 target total: {PAPER_TOTAL}")
    g = out.groupby("country").agg(n=("image_id", "size"), cavity=("cavity", "sum"))
    g["no_cavity"] = g["n"] - g["cavity"]
    print(g.to_string())


if __name__ == "__main__":
    main()
