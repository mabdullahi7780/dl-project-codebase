"""Cache frozen backbone embeddings for every manifest image.

Runs a (frozen) backbone over the manifest once and writes an ``.npz`` with two
parallel arrays — ``image_id`` and ``features`` [N, dim] — plus the backbone name.
Downstream heads load this and train on cached vectors in seconds.

Usage::

    python scripts/cache_features.py \
        --manifest tbportals_manifest_paper.csv \
        --out /kaggle/working/raddino_features.npz \
        --backbone rad-dino --batch-size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.components.backbones import build_backbone
from src.data.tbportals_dataset import _load_image


def load_features(path: str) -> tuple[dict[str, np.ndarray], int]:
    """Return ({image_id: vector}, dim) from a cache produced by this script."""
    z = np.load(path, allow_pickle=True)
    ids = [str(i) for i in z["image_id"]]
    feats = z["features"].astype(np.float32)
    return {i: feats[k] for k, i in enumerate(ids)}, int(feats.shape[1])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Cache frozen backbone features for the agentic pipeline.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="rad-dino", choices=["rad-dino", "txrv", "densenet"])
    ap.add_argument("--model-id", default=None, help="Override HF model id (rad-dino only).")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cache] backbone={args.backbone} device={device}")
    backbone = build_backbone(args.backbone, device, args.model_id)
    print(f"[cache] feature dim = {backbone.dim}")

    df = pd.read_csv(args.manifest, dtype={"image_id": str})
    ids: list[str] = []
    feats: list[np.ndarray] = []
    batch_imgs, batch_ids = [], []

    def flush():
        if not batch_imgs:
            return
        feats.append(backbone.embed(batch_imgs))
        ids.extend(batch_ids)
        batch_imgs.clear()
        batch_ids.clear()

    n = len(df)
    for k, row in enumerate(df.itertuples(index=False)):
        try:
            batch_imgs.append(_load_image(str(row.image_path)))
            batch_ids.append(str(row.image_id))
        except Exception as exc:  # skip unreadable images, keep going
            print(f"[cache] skip {row.image_id}: {exc}")
        if len(batch_imgs) >= args.batch_size:
            flush()
            if (k + 1) % (args.batch_size * 10) == 0:
                print(f"[cache] {k + 1}/{n}")
    flush()

    features = np.concatenate(feats, axis=0).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, image_id=np.array(ids, dtype=object),
                        features=features, backbone=args.backbone)
    print(f"[cache] wrote {features.shape[0]} x {features.shape[1]} features -> {out}")


if __name__ == "__main__":
    main()
