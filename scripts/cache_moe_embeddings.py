"""Pre-compute Component 1 image embeddings for MoE training.

Running the frozen MedSAM ViT-B encoder on every batch is the dominant
cost during MoE training.  This script does it once, caches the
``[256, 64, 64]`` embedding per image to a single ``.pt`` file, and
also stores the lesion mask, lung mask, and pathology pseudo-labels
so the training scripts can iterate quickly.

Usage::

    python scripts/cache_moe_embeddings.py \
        --config configs/moe.yaml \
        --paths configs/paths.yaml \
        --output outputs/embedding_cache \
        --datasets tbx11k shenzhen montgomery \
        --limit-per-domain 500
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.components.component0_qc import harmonise_sample
from src.components.component1_dann import Component1DANNModel, DANNHead
from src.components.component1_encoder import (
    Component1EncoderConfig,
    build_component1_encoder,
    load_trainable_state_dict,
)
from src.components.component2_txv import Component2SoftDomainContext
from src.components.component4_lung import Component4MedSAM
from src.core.device import pick_device


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _walk_images(root: Path, limit: int | None = None) -> list[Path]:
    """Recursively find image files under root, with sane skips."""
    skip = {"mask", "manualmask", "leftmask", "rightmask", "segmentation"}
    images: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts_lower = {p.lower() for p in path.parts}
        if any(s in parts_lower for s in skip):
            continue
        images.append(path)
        if limit is not None and len(images) >= limit:
            break
    return images


def _load_image_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def cache_one(
    image_path: Path,
    dataset_id: str,
    *,
    component1: Component1DANNModel,
    component4: Component4MedSAM,
    component2: Component2SoftDomainContext,
    device: torch.device,
    output_dir: Path,
) -> Path | None:
    """Cache embedding + supporting tensors for a single image."""
    try:
        image = _load_image_grayscale(image_path)
    except Exception as e:
        print(f"  Skip {image_path}: {e}")
        return None

    sample = {
        "image": image,
        "image_id": image_path.stem,
        "dataset_id": dataset_id,
        "view": None,
        "pixel_spacing_cm": None,
        "path": str(image_path),
    }
    try:
        harmonised = harmonise_sample(sample)
    except Exception as e:
        print(f"  Skip {image_path}: {e}")
        return None

    with torch.no_grad():
        x3 = harmonised.x_3ch.unsqueeze(0).to(device)
        x224 = harmonised.x_224.unsqueeze(0).to(device)

        img_emb, _ = component1(x3, lambda_=0.0)
        lung_out = component4.predict_masks(x3)
        txv_out = component2.forward_features(x224)

    # Build a coarse lesion pseudo-label from TXV pathology probs over lung
    # (placeholder: real GT from TBX11K bboxes should override this).
    lesion_mask = torch.zeros(1, 256, 256)

    out_path = output_dir / f"{dataset_id}__{image_path.stem}.pt"
    torch.save(
        {
            "image_emb": img_emb.squeeze(0).cpu(),         # [256, 64, 64]
            "lesion_mask": lesion_mask,                       # [1, 256, 256]
            "lung_mask": lung_out.lung_mask_256.squeeze(0).cpu(),  # [1, 256, 256]
            "pathology_logits": txv_out.pathology_logits.squeeze(0).cpu(),  # [18]
            "image_1024": harmonised.x_1024.cpu(),           # [1, 1024, 1024]
            "dataset_id": dataset_id,
            "image_id": image_path.stem,
        },
        out_path,
    )
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-compute MoE training embeddings.")
    p.add_argument("--config", default="configs/moe.yaml")
    p.add_argument("--paths", default="configs/paths.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--datasets", nargs="+", default=["tbx11k", "shenzhen", "montgomery"])
    p.add_argument("--limit-per-domain", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    with open(args.paths, encoding="utf-8") as f:
        paths_cfg = yaml.safe_load(f) or {}
    dataset_paths = paths_cfg.get("datasets", {})

    device = pick_device(config.get("runtime", {}).get("device"))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build models once
    component1_cfg = config.get("component1", {})
    encoder = build_component1_encoder(
        Component1EncoderConfig(
            backend=str(component1_cfg.get("backend", "auto")),
            checkpoint_path=component1_cfg.get("checkpoint_path"),
            freeze_backbone=True,
        )
    )
    component1 = Component1DANNModel(encoder=encoder, head=DANNHead()).to(device).eval()
    if component1_cfg.get("adapter_path"):
        load_trainable_state_dict(component1, component1_cfg["adapter_path"])

    component4_cfg = config.get("component4", {})
    component4 = Component4MedSAM(
        backend=str(component4_cfg.get("backend", "auto")),
        checkpoint_path=component4_cfg.get("checkpoint_path"),
        model_type=str(component4_cfg.get("model_type", "vit_b")),
    ).to(device).eval()
    if component4_cfg.get("decoder_checkpoint_path") and component4.active_backend == "medsam":
        component4.load_trained_decoder(component4_cfg["decoder_checkpoint_path"])

    component2 = Component2SoftDomainContext(
        backend=str(config.get("component2", {}).get("backend", "auto")),
    ).to(device).eval()

    total = 0
    for dataset_id in args.datasets:
        ds_root = dataset_paths.get(dataset_id)
        if not ds_root or not Path(ds_root).exists():
            print(f"[skip] {dataset_id}: path not found ({ds_root})")
            continue

        print(f"[cache] {dataset_id}: scanning {ds_root}")
        images = _walk_images(Path(ds_root), limit=args.limit_per_domain)
        print(f"  found {len(images)} images")

        for i, img_path in enumerate(images):
            saved = cache_one(
                img_path,
                dataset_id,
                component1=component1,
                component4=component4,
                component2=component2,
                device=device,
                output_dir=output_dir,
            )
            if saved:
                total += 1
            if (i + 1) % 50 == 0:
                print(f"  {dataset_id}: cached {i + 1}/{len(images)}")

    print(f"\nTotal cached embeddings: {total}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
