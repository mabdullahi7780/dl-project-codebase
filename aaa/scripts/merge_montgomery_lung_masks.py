"""Merge Montgomery left/right lung masks into single binary PNGs.

Montgomery distributes masks as two PNGs per CXR under:
    <montgomery_root>/ManualMask/leftMask/<id>.png
    <montgomery_root>/ManualMask/rightMask/<id>.png

This script takes the set-union of the two and writes one merged mask per id
into ``<out_dir>/<id>.png`` as a binary {0, 255} PNG.

Usage:
    python3.11 -m scripts.merge_montgomery_lung_masks \\
        --montgomery-root /path/to/montgomery \\
        --out-dir /path/to/montgomery/MergedMask
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--montgomery-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--left-subdir", default="ManualMask/leftMask")
    parser.add_argument("--right-subdir", default="ManualMask/rightMask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_dir: Path = args.montgomery_root / args.left_subdir
    right_dir: Path = args.montgomery_root / args.right_subdir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not left_dir.is_dir() or not right_dir.is_dir():
        raise FileNotFoundError(
            f"Expected leftMask at {left_dir} and rightMask at {right_dir}."
        )

    left_files = {p.name: p for p in sorted(left_dir.glob("*.png")) if not p.name.startswith("._")}
    right_files = {p.name: p for p in sorted(right_dir.glob("*.png")) if not p.name.startswith("._")}
    common = sorted(set(left_files) & set(right_files))
    missing_right = sorted(set(left_files) - set(right_files))
    missing_left = sorted(set(right_files) - set(left_files))

    if missing_right:
        print(f"WARNING: {len(missing_right)} left masks have no right counterpart (skipped).")
    if missing_left:
        print(f"WARNING: {len(missing_left)} right masks have no left counterpart (skipped).")

    n_written = 0
    for name in common:
        with Image.open(left_files[name]) as li:
            left = np.asarray(li.convert("L"))
        with Image.open(right_files[name]) as ri:
            right = np.asarray(ri.convert("L"))
        if left.shape != right.shape:
            print(f"SKIP {name}: shape mismatch {left.shape} vs {right.shape}")
            continue
        merged = np.maximum((left > 0).astype(np.uint8), (right > 0).astype(np.uint8))
        if merged.sum() == 0:
            print(f"SKIP {name}: merged mask is empty")
            continue
        Image.fromarray((merged * 255).astype(np.uint8), mode="L").save(out_dir / name)
        n_written += 1

    print(f"Wrote {n_written} merged masks to {out_dir}")


if __name__ == "__main__":
    main()
