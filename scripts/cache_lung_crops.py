"""Pre-compute lung-cropped 224x224 images for the TB Portals manifest.

Reuses ``Component4MedSAM`` (Sprint-1 lung module) to segment the lungs, takes
the bounding box of the lung mask, crops the original image, and caches a PNG
named ``{image_id}.png`` under ``--out-dir``. The dataset then loads these crops
when ``use_lung_crop=True``.

This step is OPTIONAL. Validate the full pipeline first with whole images
(``use_lung_crop=False``); add lung crops to close the gap to Kantipudi (who
regresses ALP from the cropped lung region).

Requires ``segment_anything`` + a MedSAM checkpoint. If only the mock backend is
available, the script refuses to run (so you never get fake whole-image "crops"
mistaken for real ones) and tells you to set ``use_lung_crop=False`` instead.

Example::

    python scripts/cache_lung_crops.py \
        --manifest data/processed/tbportals_manifest.csv \
        --out-dir data/processed/tbportals_crops \
        --medsam-ckpt checkpoints/medsam/medsam_vit_b.pth \
        --lung-decoder-ckpt checkpoints/baseline/component4_lung_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.components.component4_lung import Component4MedSAM
from src.core.device import describe_device, pick_device
from src.data.tbportals import load_manifest

MEDSAM_SIZE = 1024


def _load_1024_3ch(path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    img = Image.open(path).convert("L")
    w, h = img.size
    arr = np.asarray(img.resize((MEDSAM_SIZE, MEDSAM_SIZE)), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None].repeat(1, 3, 1, 1)  # [1,3,1024,1024]
    return t, (w, h)


def _mask_bbox(mask_1024: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_1024 > 0.5)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(MEDSAM_SIZE - 1, x1 + pad)
    y1 = min(MEDSAM_SIZE - 1, y1 + pad)
    return x0, y0, x1, y1


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Cache lung-cropped 224 images.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--medsam-ckpt", required=True)
    p.add_argument("--lung-decoder-ckpt", default=None)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="0 = all (debug only).")
    args = p.parse_args(argv)

    device = pick_device()
    print(f"[crops] device={describe_device(device)}")

    df = load_manifest(args.manifest)
    if args.limit:
        df = df.iloc[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If every crop already exists, skip loading MedSAM entirely. Its ViT-B
    # encoder otherwise holds GPU memory for the rest of the kernel session,
    # which can OOM a subsequent training run in the same notebook.
    n_have = sum((out_dir / f"{row['image_id']}.png").is_file() for _, row in df.iterrows())
    if n_have >= len(df):
        print(f"[crops] all {len(df)} crops already cached in {out_dir}; skipping MedSAM.")
        return
    print(f"[crops] {n_have}/{len(df)} cached; generating the remaining {len(df) - n_have}.")

    model = Component4MedSAM(backend="medsam", checkpoint_path=args.medsam_ckpt).to(device)
    if model.active_backend != "medsam":
        raise SystemExit(
            "[crops] MedSAM backend unavailable (segment_anything missing or bad checkpoint). "
            "Refusing to write fake crops. Run training with use_lung_crop=False instead."
        )
    if args.lung_decoder_ckpt:
        model.load_trained_decoder(args.lung_decoder_ckpt)
        print(f"[crops] loaded fine-tuned lung decoder: {args.lung_decoder_ckpt}")
    model.eval()

    n_ok = n_fallback = 0
    for i, row in df.iterrows():
        out_path = out_dir / f"{row['image_id']}.png"
        if out_path.is_file():
            continue
        try:
            x, (w, h) = _load_1024_3ch(str(row["image_path"]))
        except Exception as exc:  # noqa: BLE001 - skip unreadable images
            print(f"[crops] skip {row['image_id']}: {exc}")
            continue

        with torch.no_grad():
            mask = model.predict_masks(x.to(device)).lung_mask_1024[0, 0].cpu().numpy()

        bbox = _mask_bbox(mask, args.pad)
        orig = Image.open(str(row["image_path"])).convert("L")
        if bbox is None:
            crop = orig  # empty mask -> keep whole image
            n_fallback += 1
        else:
            sx, sy = w / MEDSAM_SIZE, h / MEDSAM_SIZE
            x0, y0, x1, y1 = bbox
            crop = orig.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))
            n_ok += 1

        crop.resize((args.size, args.size)).save(out_path)
        if (i + 1) % 200 == 0:
            print(f"[crops] {i + 1}/{len(df)} (lung={n_ok}, fallback={n_fallback})")

    print(f"[crops] DONE -> {out_dir} (lung-cropped={n_ok}, whole-image fallback={n_fallback})")

    # Free MedSAM so it does not hold GPU memory for a later training cell.
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
