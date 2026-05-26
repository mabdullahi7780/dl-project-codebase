"""Convert TBX11K lesion annotations to a YOLO dataset for Kantipudi A1.

TBX11K (Liu et al. 2020) ships TB images with Pascal-VOC XML bounding boxes
(classes ``ActiveTuberculosis`` / ``ObsoletePulmonaryTuberculosis``). For the A1
lesion detector we only need a single ``lesion`` class, so all TB lesion classes
are merged to class 0. The paper trains the detector on the ~799 annotated TB
images only (healthy/sick images have no boxes), so by default we include just
the annotated images.

This script is defensive about layout: it globs for ``*.xml`` under ``--tbx-root``
and resolves each image by filename anywhere under the root. It PRINTS what it
finds first, so if the Kaggle dataset layout differs you can see it immediately.

Output (ultralytics-ready)::

    <out>/images/{train,val}/*.png      (symlinks to the originals)
    <out>/labels/{train,val}/*.txt      (YOLO: "0 cx cy w h", normalised)
    <out>/tbx11k.yaml                   (dataset config)

Usage::

    python scripts/prepare_tbx11k_yolo.py \
        --tbx-root /kaggle/input/datasets/usmanshams/tbx-11/TBX11K \
        --out /kaggle/working/tbx11k_yolo --val-frac 0.2
"""

from __future__ import annotations

import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def _find_xml(root: Path) -> list[Path]:
    return sorted(root.rglob("*.xml"))


def _index_images(root: Path) -> dict[str, Path]:
    """filename (lowercased) -> path, for png/jpg images under root."""
    idx: dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
        for p in root.rglob(ext):
            idx.setdefault(p.name.lower(), p)
    return idx


def _read_list(path: Path) -> set[str]:
    """Lowercased image stems from a TBX11K list file (first token per line)."""
    names: set[str] = set()
    sample: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        names.add(Path(line.split()[0]).stem.lower())
        if len(sample) < 2:
            sample.append(line)
    print(f"[tbx11k] {path.name}: {len(names)} entries; sample={sample}")
    return names


def _parse_voc(xml_path: Path) -> tuple[str, int, int, list[tuple[float, float, float, float]]]:
    """Return (image_filename, width, height, [(xmin,ymin,xmax,ymax), ...])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    fname = root.findtext("filename") or (xml_path.stem + ".png")
    size = root.find("size")
    w = int(float(size.findtext("width"))) if size is not None else 0
    h = int(float(size.findtext("height"))) if size is not None else 0
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        boxes.append((
            float(bb.findtext("xmin")), float(bb.findtext("ymin")),
            float(bb.findtext("xmax")), float(bb.findtext("ymax")),
        ))
    return fname, w, h, boxes


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="TBX11K (VOC XML) -> YOLO lesion dataset.")
    ap.add_argument("--tbx-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--train-list", default=None,
                    help="TBX11K official train list (e.g. lists/TBX11K_train.txt). With --val-list "
                         "this reproduces the paper's split instead of a random --val-frac split.")
    ap.add_argument("--val-list", default=None, help="TBX11K official val list (e.g. lists/TBX11K_val.txt).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    root = Path(args.tbx_root)
    if not root.is_dir():
        raise SystemExit(f"[tbx11k] --tbx-root not found: {root}")

    print(f"[tbx11k] scanning {root} ...")
    print("[tbx11k] top-level entries:", sorted(p.name for p in root.iterdir())[:20])

    xmls = _find_xml(root)
    print(f"[tbx11k] found {len(xmls)} XML annotation files")
    if not xmls:
        raise SystemExit(
            "[tbx11k] No *.xml found. Inspect the printed layout — annotations may be "
            "COCO JSON instead of VOC XML, or under a different path. Report the structure."
        )

    images = _index_images(root)
    print(f"[tbx11k] indexed {len(images)} image files")

    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    records = []  # (img_path, [yolo_lines])
    n_boxes = n_skipped = 0
    for xml in xmls:
        fname, w, h, boxes = _parse_voc(xml)
        if not boxes:
            continue
        img_path = images.get(fname.lower()) or images.get(Path(fname).name.lower())
        if img_path is None:
            n_skipped += 1
            continue
        if w <= 0 or h <= 0:
            # fall back to reading size from the image if XML lacked it
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                n_skipped += 1
                continue
        lines = []
        for (x0, y0, x1, y1) in boxes:
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        records.append((img_path, lines))
        n_boxes += len(lines)

    if not records:
        raise SystemExit("[tbx11k] Parsed XML but matched 0 images. Check filename<->image mapping.")

    if args.train_list and args.val_list:
        train_names = _read_list(Path(args.train_list))
        val_names = _read_list(Path(args.val_list))
        tr, va, not_in_split = [], [], 0
        for img_path, lines in records:
            stem = img_path.stem.lower()
            if stem in val_names:
                va.append((img_path, lines))
            elif stem in train_names:
                tr.append((img_path, lines))
            else:
                not_in_split += 1  # e.g. held-out test images
        if len(tr) < 50 or len(va) < 10:
            raise SystemExit(
                f"[tbx11k] Official-split match too low (train={len(tr)}, val={len(va)}). The list "
                f"line format likely differs from what was parsed (see sample above). Re-run without "
                f"--train-list/--val-list for a random {args.val_frac:.0%} split."
            )
        splits = {"train": tr, "val": va}
        print(f"[tbx11k] official split: train={len(tr)} val={len(va)} "
              f"(skipped {not_in_split} annotated images not in train/val lists, e.g. test)")
    else:
        rng.shuffle(records)
        n_val = max(1, int(round(len(records) * args.val_frac)))
        splits = {"val": records[:n_val], "train": records[n_val:]}

    for split, recs in splits.items():
        for img_path, lines in recs:
            stem = img_path.stem
            link = out / f"images/{split}/{stem}{img_path.suffix}"
            if not link.exists():
                try:
                    os.symlink(img_path, link)
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.copy(img_path, link)
            (out / f"labels/{split}/{stem}.txt").write_text("\n".join(lines))

    yaml_path = out / "tbx11k.yaml"
    yaml_path.write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['lesion']\n"
    )
    print(f"[tbx11k] train={len(splits['train'])} val={len(splits['val'])} images, "
          f"{n_boxes} lesion boxes, {n_skipped} unmatched.")
    print(f"[tbx11k] dataset config -> {yaml_path}")


if __name__ == "__main__":
    main()
