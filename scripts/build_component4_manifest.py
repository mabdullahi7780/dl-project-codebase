"""Build a Component 4 manifest CSV from dataset folders.

Supported layouts (pass only the ones you have):

Montgomery:
    --montgomery-images  <root>/CXR_png
    --montgomery-masks   <merged_masks_dir>   (one PNG per id, same stem as image)

Shenzhen:
    --shenzhen-images    <root>/images
    --shenzhen-masks     <root>/mask          (filenames may have a _mask suffix, e.g. CHNCXR_0001_0_mask.png)

Each image must have a matching mask file with the same stem. Pairs where the
mask is missing are skipped with a warning.

Splits are assigned by seeded shuffle per dataset using
``--val-frac`` and ``--test-frac``. Output is written to ``--out``.

Usage:
    python3.11 -m scripts.build_component4_manifest \\
        --montgomery-images /data/montgomery/CXR_png \\
        --montgomery-masks  /data/montgomery/MergedMask \\
        --shenzhen-images   /data/shenzhen/images \\
        --shenzhen-masks    /data/shenzhen/mask \\
        --out data/splits/component4_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--montgomery-images", type=Path, default=None)
    parser.add_argument("--montgomery-masks", type=Path, default=None)
    parser.add_argument("--shenzhen-images", type=Path, default=None)
    parser.add_argument("--shenzhen-masks", type=Path, default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def _iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _pair_images_masks(images_dir: Path, masks_dir: Path) -> list[tuple[str, Path, Path]]:
    mask_by_stem: dict[str, Path] = {}
    for p in _iter_images(masks_dir):
        if p.name.startswith("._"):
            continue
        stem = p.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]
        mask_by_stem.setdefault(stem, p)
    pairs: list[tuple[str, Path, Path]] = []
    for image_path in _iter_images(images_dir):
        if image_path.name.startswith("._"):
            continue
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            print(f"WARN no mask for {image_path.name}; skipped.")
            continue
        pairs.append((image_path.stem, image_path, mask_path))
    return pairs


def _split_assignments(n: int, val_frac: float, test_frac: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    val_set = set(indices[:n_val])
    test_set = set(indices[n_val : n_val + n_test])
    return [
        "val" if i in val_set else "test" if i in test_set else "train"
        for i in range(n)
    ]


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    datasets: list[tuple[str, Path, Path]] = []
    if args.montgomery_images and args.montgomery_masks:
        datasets.append(("montgomery", args.montgomery_images, args.montgomery_masks))
    if args.shenzhen_images and args.shenzhen_masks:
        datasets.append(("shenzhen", args.shenzhen_images, args.shenzhen_masks))

    if not datasets:
        raise SystemExit("Provide at least one --<dataset>-images / --<dataset>-masks pair.")

    rows: list[dict[str, str]] = []
    for dataset_id, images_dir, masks_dir in datasets:
        if not images_dir.is_dir() or not masks_dir.is_dir():
            raise FileNotFoundError(f"Missing dir for {dataset_id}: {images_dir} / {masks_dir}")
        pairs = _pair_images_masks(images_dir, masks_dir)
        if not pairs:
            print(f"WARN {dataset_id}: 0 paired samples.")
            continue
        splits = _split_assignments(len(pairs), args.val_frac, args.test_frac, args.seed)
        for (stem, image_path, mask_path), split in zip(pairs, splits):
            rows.append(
                {
                    "sample_id": f"{dataset_id}/{stem}",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "dataset": dataset_id,
                    "split": split,
                }
            )
        print(f"{dataset_id}: {len(pairs)} pairs -> {args.out}")

    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "mask_path", "dataset", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote manifest with {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
