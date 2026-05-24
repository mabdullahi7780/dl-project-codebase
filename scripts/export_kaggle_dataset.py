"""Convert annotated TB Portals DICOMs to PNGs for Kaggle upload.

Reads the local manifest (absolute Windows paths), converts each DICOM to a
512x512 grayscale PNG, and writes a Kaggle-ready manifest whose image_path
column uses relative paths (``images/<stem>.png``).

Usage (run from repo root in the timika venv)::

    python scripts/export_kaggle_dataset.py \\
        --manifest local_work/data/processed/tbportals_manifest.csv \\
        --out-dir  local_work/kaggle_export \\
        --size 512

Output layout::

    local_work/kaggle_export/
        images/          <- 8619 PNG files  (~1.5 GB)
        manifest.csv     <- same columns, image_path = "images/<stem>.png"

Upload BOTH the ``images/`` folder and ``manifest.csv`` as a single Kaggle
dataset (e.g. "tb-portals-cxr-pngs").  Then in the Kaggle notebooks set::

    DATASET_DIR = "/kaggle/input/tb-portals-cxr-pngs"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work when the script is
# invoked from any working directory (e.g. python scripts/export_kaggle_dataset.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

from src.data.tbportals_dataset import _load_image
from src.data.tbportals import load_manifest


def _to_grayscale_png(img: Image.Image, size: int) -> Image.Image:
    """Convert any PIL image to a square grayscale PNG, no padding."""
    return img.convert("L").resize((size, size), Image.LANCZOS)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Export manifest DICOMs as PNGs for Kaggle.")
    p.add_argument("--manifest", required=True, help="Local manifest CSV path.")
    p.add_argument("--out-dir",  required=True, help="Output directory.")
    p.add_argument("--size",     type=int, default=512, help="Square output size (default 512).")
    p.add_argument("--limit",    type=int, default=0,   help="Debug: process only N images.")
    args = p.parse_args(argv)

    out_dir   = Path(args.out_dir)
    img_dir   = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.manifest)
    if args.limit:
        df = df.iloc[:args.limit]

    n_ok = n_skip = n_err = 0
    new_paths = []

    for i, row in df.iterrows():
        src_path = str(row["image_path"])
        # Derive a safe filename from the DICOM instance UID (last path component, stem).
        stem = Path(src_path.split("!DICOM!")[-1] if "!DICOM!" in src_path
                    else src_path).stem
        out_file = img_dir / f"{stem}.png"
        rel_path = f"images/{stem}.png"

        if out_file.is_file():
            new_paths.append(rel_path)
            n_skip += 1
            continue

        try:
            img = _load_image(src_path)
            _to_grayscale_png(img, args.size).save(out_file, optimize=True)
            new_paths.append(rel_path)
            n_ok += 1
        except Exception as exc:
            print(f"[export] ERROR row {i} ({stem[:40]}): {exc}")
            new_paths.append(None)
            n_err += 1

        if (n_ok + n_skip) % 500 == 0 and (n_ok + n_skip) > 0:
            done = n_ok + n_skip
            print(f"[export] {done}/{len(df)} done  (new={n_ok}, cached={n_skip}, err={n_err})")

    # Write Kaggle manifest with relative image paths.
    out_df = df.copy()
    out_df["image_path"] = new_paths
    out_df = out_df.dropna(subset=["image_path"]).reset_index(drop=True)
    manifest_out = out_dir / "manifest.csv"
    out_df.to_csv(manifest_out, index=False)

    print(f"\n[export] Done: {n_ok} converted, {n_skip} cached, {n_err} errors")
    print(f"[export] Kaggle manifest: {manifest_out}  ({len(out_df)} rows)")
    size_mb = sum(f.stat().st_size for f in img_dir.glob("*.png")) / 1024 / 1024
    print(f"[export] images/ folder:  {size_mb:.0f} MB  ({img_dir})")
    print()
    print("Next steps:")
    print("  1. kaggle datasets create  -p local_work/kaggle_export")
    print("  Or upload manually at kaggle.com > Datasets > New Dataset")
    print("  Dataset name suggestion: tb-portals-cxr-pngs")


if __name__ == "__main__":
    main()
