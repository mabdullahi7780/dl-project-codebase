"""Pre-compute MoE training cache for the 3-dataset setup.

This script turns the frozen inference stack into a disk cache of:

    - Component 1 image embeddings
    - Component 2 domain context vectors
    - lesion supervision masks
    - per-expert masks
    - lung masks
    - low-level metadata used by Phase 1/2/3 MoE training

Supervision policy
------------------
    - ``TBX11K``: real coarse supervision from bounding boxes
    - ``Montgomery`` / ``Shenzhen``: weak pseudo-masks from TXV CAMs inside lungs
    - ``NIH``: intentionally excluded from the MoE cache path for now
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.components.baseline_lesion_proposer import (
    BaselineLesionProposer,
    BaselineLesionProposerConfig,
)
from src.components.component0_qc import harmonise_sample
from src.components.component1_dann import Component1DANNModel, DANNHead
from src.components.component1_encoder import (
    Component1EncoderConfig,
    build_component1_encoder,
    load_trainable_state_dict,
)
from src.components.component2_txv import Component2SoftDomainContext, TXV_CLASS_NAMES
from src.components.component4_lung import Component4MedSAM
from src.components.component5_experts import EXPERT_NAMES
from src.core.device import pick_device
from src.data.component4_lung_dataset import load_and_merge_binary_mask, resize_mask_to
from src.training.train_component1_dann import (
    Component1DomainDataset,
    DomainSampleRef,
    _build_generic_image_samples,
    _build_tbx11k_samples,
    load_yaml_config,
    maybe_limit_manifest,
)
from src.utils.morphology import binary_erode, otsu_threshold, postprocess_binary_mask


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MASK_DIR_TOKENS = ("mask", "manualmask", "leftmask", "rightmask", "segmentation", "segmask")
EXPERT_CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    "consolidation": ("consolidation", "infiltration", "pneumonia", "lung_opacity"),
    "cavity": ("lung_lesion",),
    "fibrosis": ("fibrosis", "pleural_thickening"),
    "nodule": ("nodule", "mass"),
}
PSEUDO_DATASET_WEIGHTS: dict[str, float] = {
    "tbx11k": 0.6,
    "montgomery": 0.45,
    "shenzhen": 0.45,
}
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_cache_name(sample: DomainSampleRef) -> str:
    raw = sample.source or sample.image_path or sample.member_name or "sample"
    safe = SAFE_NAME_RE.sub("_", raw).strip("._")
    return f"{sample.dataset_id}__{safe or 'sample'}.pt"


def _sample_identifier(sample: DomainSampleRef) -> str:
    if sample.image_path is not None:
        return Path(sample.image_path).stem
    if sample.member_name is not None:
        return Path(sample.member_name).stem
    return Path(sample.source).stem if sample.source else "sample"


def _load_dataset_samples(
    paths_config: str | Path,
    *,
    component1_config: str | Path,
    datasets: list[str],
    limit_per_domain: int | None,
) -> list[DomainSampleRef]:
    paths_yaml = load_yaml_config(paths_config)
    component1_yaml = load_yaml_config(component1_config).get("component1_dann", {})
    dataset_roots = paths_yaml["datasets"]
    tbx_list = component1_yaml.get("data", {}).get("tbx_list", "train.txt")

    samples: list[DomainSampleRef] = []
    if "montgomery" in datasets:
        samples.extend(_build_generic_image_samples("montgomery", Path(dataset_roots["montgomery"])))
    if "shenzhen" in datasets:
        samples.extend(_build_generic_image_samples("shenzhen", Path(dataset_roots["shenzhen"])))
    if "tbx11k" in datasets:
        samples.extend(_build_tbx11k_samples(Path(dataset_roots["tbx11k"]), str(tbx_list)))

    return maybe_limit_manifest(samples, limit_per_domain)


def _is_mask_path(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    return any(any(token in part for token in MASK_DIR_TOKENS) for part in lowered_parts)


def _build_lung_mask_index(root: Path) -> dict[str, tuple[Path, ...]]:
    index: dict[str, list[Path]] = defaultdict(list)
    if not root.exists():
        return {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not _is_mask_path(path):
            continue
        index[path.stem].append(path)

    return {
        stem: tuple(paths)
        for stem, paths in index.items()
    }


def _load_gt_lung_mask(
    sample: DomainSampleRef,
    *,
    mask_index: dict[str, tuple[Path, ...]] | None,
) -> torch.Tensor | None:
    if mask_index is None or sample.image_path is None:
        return None

    image_stem = Path(sample.image_path).stem
    mask_paths = mask_index.get(image_stem)
    if not mask_paths:
        return None

    try:
        mask_np = load_and_merge_binary_mask(mask_paths)
    except Exception as exc:
        print(f"  Warning: failed to load GT lung mask for {sample.image_path}: {exc}")
        return None

    return resize_mask_to(mask_np, 256).cpu()


def _load_tbx11k_annotation_index(
    tbx_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    anno_dir = tbx_root / "annotations" / "json"
    if not anno_dir.is_dir():
        return {}

    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for json_path in sorted(anno_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue

        id_to_name = {
            int(img["id"]): Path(str(img["file_name"])).name
            for img in data.get("images") or []
            if "id" in img and "file_name" in img
        }
        category_to_name = {
            int(cat["id"]): str(cat.get("name", ""))
            for cat in data.get("categories") or []
            if "id" in cat
        }

        for ann in data.get("annotations") or []:
            image_id = ann.get("image_id")
            bbox = ann.get("bbox")
            if image_id is None or not bbox or len(bbox) != 4:
                continue
            image_name = id_to_name.get(int(image_id))
            if image_name is None:
                continue
            x, y, w, h = bbox
            payload = {
                "bbox": (
                    int(round(x)),
                    int(round(y)),
                    int(round(x + w)),
                    int(round(y + h)),
                ),
                "category": category_to_name.get(int(ann.get("category_id", -1)), ""),
            }
            result[image_name].append(payload)
            result[Path(image_name).stem].append(payload)

    return dict(result)


def _rasterise_boxes_to_mask(
    boxes: list[tuple[int, int, int, int]],
    original_size: tuple[int, int],
    *,
    size: int = 256,
) -> torch.Tensor:
    width, height = original_size
    mask = np.zeros((size, size), dtype=np.uint8)
    if width <= 0 or height <= 0:
        return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

    sx = size / float(width)
    sy = size / float(height)
    for x1, y1, x2, y2 in boxes:
        tx1 = max(0, min(size, int(round(x1 * sx))))
        ty1 = max(0, min(size, int(round(y1 * sy))))
        tx2 = max(0, min(size, int(round(x2 * sx))))
        ty2 = max(0, min(size, int(round(y2 * sy))))
        if tx2 > tx1 and ty2 > ty1:
            mask[ty1:ty2, tx1:tx2] = 1

    return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)


def _map_tbx_category_to_experts(category_name: str) -> list[str]:
    name = category_name.lower().replace("-", " ").replace("_", " ")
    matches: list[str] = []
    if any(token in name for token in ("consolid", "infiltrat", "opacity", "pneumon")):
        matches.append("consolidation")
    if any(token in name for token in ("cavit", "cavity")):
        matches.append("cavity")
    if any(token in name for token in ("fibro", "scar", "pleural")):
        matches.append("fibrosis")
    if any(token in name for token in ("nodul", "mass", "granuloma", "lesion")):
        matches.append("nodule")
    return matches


def _class_group_confidence(
    pathology_probs: torch.Tensor,
    *,
    class_to_index: dict[str, int],
    group_names: tuple[str, ...],
) -> float:
    values = [
        float(pathology_probs[class_to_index[name]].item())
        for name in group_names
        if name in class_to_index
    ]
    return max(values) if values else 0.0


def _class_group_cam(
    *,
    features_7x7: torch.Tensor,
    pathology_probs: torch.Tensor,
    classifier_weight: torch.Tensor | None,
    class_to_index: dict[str, int],
    group_names: tuple[str, ...],
) -> torch.Tensor:
    selected_indices = [class_to_index[name] for name in group_names if name in class_to_index]
    if not selected_indices:
        return torch.zeros((7, 7), dtype=torch.float32, device=features_7x7.device)

    if classifier_weight is None:
        base = features_7x7.abs().mean(dim=0)
        scale = sum(float(pathology_probs[idx].item()) for idx in selected_indices) / max(len(selected_indices), 1)
        return base * scale

    maps: list[torch.Tensor] = []
    for class_index in selected_indices:
        weight = classifier_weight[class_index].to(features_7x7.device).view(-1, 1, 1)
        cam = torch.relu((features_7x7 * weight).sum(dim=0))
        maps.append(cam * pathology_probs[class_index])

    if not maps:
        return torch.zeros((7, 7), dtype=torch.float32, device=features_7x7.device)
    return torch.stack(maps, dim=0).sum(dim=0)


def _score_map_to_mask(
    score_map_256: torch.Tensor,
    prior_mask_256: torch.Tensor,
    *,
    min_region_px: int,
) -> torch.Tensor:
    score_np = score_map_256.detach().cpu().numpy().astype(np.float32)
    prior_np = (prior_mask_256.detach().cpu().numpy() > 0.5)
    score_np = score_np * prior_np.astype(np.float32)

    max_value = float(score_np.max())
    if max_value <= 0.0 or not prior_np.any():
        return torch.zeros((1, score_np.shape[0], score_np.shape[1]), dtype=torch.float32)

    score_np = score_np / max_value
    threshold = otsu_threshold(score_np[prior_np])
    cleaned = postprocess_binary_mask(
        score_np >= threshold,
        min_area=min_region_px,
        opening_iters=1,
        closing_iters=1,
    )
    cleaned &= prior_np
    return torch.from_numpy(cleaned.astype(np.float32)).unsqueeze(0)


def _ringify_mask(mask_256: torch.Tensor) -> torch.Tensor:
    mask_np = (mask_256.squeeze(0).detach().cpu().numpy() > 0.5)
    if not mask_np.any():
        return mask_256
    eroded = binary_erode(mask_np, iterations=1)
    ring = mask_np & (~eroded)
    if not ring.any():
        return mask_256
    return torch.from_numpy(ring.astype(np.float32)).unsqueeze(0)


def _build_pseudo_targets(
    *,
    dataset_id: str,
    x_224: torch.Tensor,
    txv_features: torch.Tensor,
    pathology_logits: torch.Tensor,
    classifier_weight: torch.Tensor | None,
    lung_mask_256: torch.Tensor,
    proposer: BaselineLesionProposer,
    pseudo_threshold: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float], dict[str, float], str]:
    class_to_index = {name: idx for idx, name in enumerate(TXV_CLASS_NAMES)}
    pathology_probs = torch.sigmoid(pathology_logits.squeeze(0)).detach().cpu()

    suspicious_classes = {
        "consolidation",
        "infiltration",
        "fibrosis",
        "pleural_thickening",
        "effusion",
        "mass",
        "nodule",
        "pneumonia",
        "lung_opacity",
        "lung_lesion",
    }
    suspicious_max = max(
        (
            float(pathology_probs[class_to_index[name]].item())
            for name in suspicious_classes
            if name in class_to_index
        ),
        default=0.0,
    )

    zero_mask = torch.zeros_like(lung_mask_256.cpu())
    expert_confidences = {
        name: _class_group_confidence(
            pathology_probs,
            class_to_index=class_to_index,
            group_names=EXPERT_CLASS_GROUPS[name],
        )
        for name in EXPERT_NAMES
    }

    if suspicious_max < pseudo_threshold:
        expert_masks = {name: zero_mask.clone() for name in EXPERT_NAMES}
        expert_weights = {
            name: PSEUDO_DATASET_WEIGHTS[dataset_id] * (0.1 if name == "cavity" else 0.2)
            for name in EXPERT_NAMES
        }
        return zero_mask.clone(), expert_masks, expert_weights, expert_confidences, "weak_negative"

    proposal = proposer.propose(
        x_224=x_224,
        features_7x7=txv_features,
        pathology_logits=pathology_logits,
        lung_mask_256=lung_mask_256.unsqueeze(0).to(txv_features.device),
        classifier_weight=classifier_weight,
    )
    lesion_mask = proposal.lesion_mask_coarse_256[0].cpu()
    prior_mask = lesion_mask if float(lesion_mask.sum().item()) > 0 else lung_mask_256.cpu()

    expert_masks: dict[str, torch.Tensor] = {}
    expert_weights: dict[str, float] = {}
    for name in EXPERT_NAMES:
        confidence = expert_confidences[name]
        cam_7x7 = _class_group_cam(
            features_7x7=txv_features.squeeze(0),
            pathology_probs=pathology_probs.to(txv_features.device),
            classifier_weight=classifier_weight,
            class_to_index=class_to_index,
            group_names=EXPERT_CLASS_GROUPS[name],
        )
        cam_256 = F.interpolate(
            cam_7x7.unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        expert_mask = _score_map_to_mask(
            cam_256,
            prior_mask.squeeze(0),
            min_region_px=12 if name == "cavity" else 24,
        )
        if name == "cavity":
            expert_mask = _ringify_mask(expert_mask)
        expert_masks[name] = expert_mask.cpu()

        floor = 0.1 if name == "cavity" else 0.25
        expert_weights[name] = PSEUDO_DATASET_WEIGHTS[dataset_id] * max(floor, min(confidence, 1.0))

    return lesion_mask, expert_masks, expert_weights, expert_confidences, "pseudo_cam"


def _build_tbx_targets(
    *,
    sample: DomainSampleRef,
    x_224: torch.Tensor,
    original_size: tuple[int, int],
    tbx_annotations: dict[str, list[dict[str, Any]]],
    txv_features: torch.Tensor,
    pathology_logits: torch.Tensor,
    classifier_weight: torch.Tensor | None,
    lung_mask_256: torch.Tensor,
    proposer: BaselineLesionProposer,
    pseudo_threshold: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float], dict[str, float], float, str]:
    image_name = Path(sample.image_path).name if sample.image_path else sample.source
    annotations = tbx_annotations.get(str(image_name), []) or tbx_annotations.get(_sample_identifier(sample), [])
    box_list = [tuple(ann["bbox"]) for ann in annotations if "bbox" in ann]

    if not box_list:
        lesion_mask, expert_masks, expert_weights, confidences, supervision_type = _build_pseudo_targets(
            dataset_id="tbx11k",
            x_224=x_224,
            txv_features=txv_features,
            pathology_logits=pathology_logits,
            classifier_weight=classifier_weight,
            lung_mask_256=lung_mask_256,
            proposer=proposer,
            pseudo_threshold=pseudo_threshold,
        )
        return lesion_mask, expert_masks, expert_weights, confidences, PSEUDO_DATASET_WEIGHTS["tbx11k"], f"tbx_{supervision_type}"

    lesion_mask = _rasterise_boxes_to_mask(box_list, original_size, size=256)
    expert_box_map: dict[str, list[tuple[int, int, int, int]]] = {name: [] for name in EXPERT_NAMES}
    for ann in annotations:
        bbox = ann.get("bbox")
        if bbox is None:
            continue
        for expert_name in _map_tbx_category_to_experts(str(ann.get("category", ""))):
            expert_box_map[expert_name].append(tuple(bbox))

    class_to_index = {name: idx for idx, name in enumerate(TXV_CLASS_NAMES)}
    pathology_probs = torch.sigmoid(pathology_logits.squeeze(0)).detach().cpu()
    expert_confidences = {
        name: _class_group_confidence(
            pathology_probs,
            class_to_index=class_to_index,
            group_names=EXPERT_CLASS_GROUPS[name],
        )
        for name in EXPERT_NAMES
    }

    expert_masks: dict[str, torch.Tensor] = {}
    expert_weights: dict[str, float] = {}
    for name in EXPERT_NAMES:
        if expert_box_map[name]:
            expert_masks[name] = _rasterise_boxes_to_mask(expert_box_map[name], original_size, size=256)
            expert_weights[name] = 1.0
            continue

        cam_7x7 = _class_group_cam(
            features_7x7=txv_features.squeeze(0),
            pathology_probs=pathology_probs.to(txv_features.device),
            classifier_weight=classifier_weight,
            class_to_index=class_to_index,
            group_names=EXPERT_CLASS_GROUPS[name],
        )
        cam_256 = F.interpolate(
            cam_7x7.unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        expert_mask = _score_map_to_mask(
            cam_256,
            lesion_mask.squeeze(0),
            min_region_px=12 if name == "cavity" else 24,
        )
        if name == "cavity":
            expert_mask = _ringify_mask(expert_mask)
        expert_masks[name] = expert_mask.cpu()
        expert_weights[name] = 0.75 if float(expert_mask.sum().item()) > 0 else 0.35

    return lesion_mask.cpu(), expert_masks, expert_weights, expert_confidences, 1.0, "tbx_bbox"


def cache_one(
    sample: DomainSampleRef,
    *,
    image_loader: Component1DomainDataset,
    component1: Component1DANNModel,
    component4: Component4MedSAM,
    component2: Component2SoftDomainContext,
    device: torch.device,
    output_dir: Path,
    proposer: BaselineLesionProposer,
    pseudo_threshold: float,
    lung_mask_index: dict[str, dict[str, tuple[Path, ...]]],
    tbx_annotations: dict[str, list[dict[str, Any]]],
) -> Path | None:
    try:
        image = image_loader._load_image(sample)
    except Exception as exc:
        print(f"  Skip {sample.source}: {exc}")
        return None

    sample_meta = {
        "image": image,
        "image_id": _sample_identifier(sample),
        "dataset_id": sample.dataset_id,
        "view": None,
        "pixel_spacing_cm": None,
        "path": sample.image_path or sample.source,
        "source": sample.source,
    }
    try:
        harmonised = harmonise_sample(sample_meta)
    except Exception as exc:
        print(f"  Skip {sample.source}: {exc}")
        return None

    original_size = (int(image.shape[1]), int(image.shape[0]))

    with torch.no_grad():
        x3 = harmonised.x_3ch.unsqueeze(0).to(device)
        x224 = harmonised.x_224.unsqueeze(0).to(device)

        img_emb, _ = component1(x3, lambda_=0.0)
        txv_out = component2.forward_features(x224)
        lung_out = component4.predict_masks(x3)

    gt_lung_mask = _load_gt_lung_mask(
        sample,
        mask_index=lung_mask_index.get(sample.dataset_id),
    )
    if gt_lung_mask is not None:
        lung_mask_256 = gt_lung_mask
        lung_mask_source = "gt"
    else:
        lung_mask_256 = lung_out.lung_mask_256.squeeze(0).cpu()
        lung_mask_source = "component4"

    if sample.dataset_id == "tbx11k":
        lesion_mask, expert_masks, expert_weights, expert_confidences, supervision_weight, supervision_type = _build_tbx_targets(
            sample=sample,
            x_224=x224,
            original_size=original_size,
            tbx_annotations=tbx_annotations,
            txv_features=txv_out.features_7x7,
            pathology_logits=txv_out.pathology_logits,
            classifier_weight=txv_out.classifier_weight,
            lung_mask_256=lung_mask_256,
            proposer=proposer,
            pseudo_threshold=pseudo_threshold,
        )
    else:
        lesion_mask, expert_masks, expert_weights, expert_confidences, supervision_type = _build_pseudo_targets(
            dataset_id=sample.dataset_id,
            x_224=x224,
            txv_features=txv_out.features_7x7,
            pathology_logits=txv_out.pathology_logits,
            classifier_weight=txv_out.classifier_weight,
            lung_mask_256=lung_mask_256,
            proposer=proposer,
            pseudo_threshold=pseudo_threshold,
        )
        supervision_weight = PSEUDO_DATASET_WEIGHTS[sample.dataset_id]

    out_path = output_dir / _safe_cache_name(sample)
    torch.save(
        {
            "image_emb": img_emb.squeeze(0).cpu(),
            "domain_ctx": txv_out.domain_ctx.squeeze(0).cpu(),
            "lesion_mask": lesion_mask.cpu(),
            "expert_masks": {
                name: expert_masks[name].cpu()
                for name in EXPERT_NAMES
            },
            "supervision_weight": float(supervision_weight),
            "expert_supervision_weights": {
                name: float(expert_weights[name])
                for name in EXPERT_NAMES
            },
            "expert_confidences": {
                name: float(expert_confidences[name])
                for name in EXPERT_NAMES
            },
            "lung_mask": lung_mask_256.cpu(),
            "lung_mask_source": lung_mask_source,
            "pathology_logits": txv_out.pathology_logits.squeeze(0).cpu(),
            "pathology_probs": torch.sigmoid(txv_out.pathology_logits.squeeze(0)).cpu(),
            "image_1024": harmonised.x_1024.cpu(),
            "dataset_id": sample.dataset_id,
            "image_id": _sample_identifier(sample),
            "source": sample.source,
            "path": sample.image_path or sample.source,
            "supervision_type": supervision_type,
            "original_size": original_size,
        },
        out_path,
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute 3-dataset MoE cache.")
    parser.add_argument("--config", default="configs/moe.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--component1-config", default="configs/component1_dann.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", default=["tbx11k", "shenzhen", "montgomery"])
    parser.add_argument("--limit-per-domain", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [dataset for dataset in args.datasets if dataset != "nih_cxr14"]

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    paths_cfg = load_yaml_config(args.paths)
    dataset_paths = paths_cfg.get("datasets", {})

    device = pick_device(config.get("runtime", {}).get("device"))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_dataset_samples(
        args.paths,
        component1_config=args.component1_config,
        datasets=datasets,
        limit_per_domain=args.limit_per_domain,
    )
    image_loader = Component1DomainDataset(samples, apply_augmentation=False)

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
        mask_threshold=float(component4_cfg.get("mask_threshold", 0.5)),
    ).to(device).eval()
    if component4_cfg.get("decoder_checkpoint_path") and component4.active_backend == "medsam":
        component4.load_trained_decoder(component4_cfg["decoder_checkpoint_path"])

    component2_cfg = config.get("component2", {})
    component2 = Component2SoftDomainContext(
        backend=str(component2_cfg.get("backend", "auto")),
        weights=str(component2_cfg.get("weights", "densenet121-res224-all")),
    ).to(device).eval()
    routing_head_path = component2_cfg.get("routing_head_path")
    if routing_head_path:
        loaded = component2.load_trained_routing_head(routing_head_path)
        print(f"Component 2: loaded routing head from {loaded}")

    proposer = BaselineLesionProposer(
        BaselineLesionProposerConfig(
            suspicious_class_threshold=float(
                config.get("baseline_lesion_proposer", {}).get("suspicious_class_threshold", 0.55)
            ),
            fixed_binary_threshold=config.get("baseline_lesion_proposer", {}).get("fixed_binary_threshold"),
            min_region_px=int(config.get("baseline_lesion_proposer", {}).get("min_region_px", 48)),
            opening_iters=int(config.get("baseline_lesion_proposer", {}).get("opening_iters", 1)),
            closing_iters=int(config.get("baseline_lesion_proposer", {}).get("closing_iters", 1)),
            fallback_blend=float(config.get("baseline_lesion_proposer", {}).get("fallback_blend", 0.35)),
        )
    )
    pseudo_threshold = float(config.get("moe_cache", {}).get("pseudo_threshold", 0.45))

    lung_mask_index = {
        dataset_id: _build_lung_mask_index(Path(dataset_paths[dataset_id]))
        for dataset_id in ("montgomery", "shenzhen")
        if dataset_id in datasets and dataset_id in dataset_paths
    }
    tbx_annotations = (
        _load_tbx11k_annotation_index(Path(dataset_paths["tbx11k"]))
        if "tbx11k" in datasets and "tbx11k" in dataset_paths
        else {}
    )

    print(f"[cache] datasets={datasets} samples={len(samples)} output={output_dir}")
    counters: dict[str, int] = defaultdict(int)
    supervision_counters: dict[str, int] = defaultdict(int)

    for index, sample in enumerate(samples, start=1):
        saved = cache_one(
            sample,
            image_loader=image_loader,
            component1=component1,
            component4=component4,
            component2=component2,
            device=device,
            output_dir=output_dir,
            proposer=proposer,
            pseudo_threshold=pseudo_threshold,
            lung_mask_index=lung_mask_index,
            tbx_annotations=tbx_annotations,
        )
        if saved is None:
            continue

        counters[sample.dataset_id] += 1
        try:
            payload = torch.load(saved, map_location="cpu", weights_only=False)
            supervision_counters[str(payload.get("supervision_type", "unknown"))] += 1
        except Exception:
            supervision_counters["unknown"] += 1

        if index % 50 == 0:
            print(f"  cached {index}/{len(samples)}")

    print("\nCache complete.")
    print(f"  output_dir: {output_dir}")
    print(f"  per_dataset: {dict(counters)}")
    print(f"  supervision: {dict(supervision_counters)}")


if __name__ == "__main__":
    main()
