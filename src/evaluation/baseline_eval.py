"""Baseline TB CXR pipeline evaluation.

Runs the full baseline pipeline (C0 → C1 → C2 → C4 → baseline_lesion_proposer
→ C7 → C8) on held-out test splits from the four canonical domains and
reports both component-level sanity metrics and system-level numbers. The
output CSVs + summary JSON are the frozen "baseline" that MoE runs will be
compared against later.

Invoked by ``scripts/kaggle_baseline_eval.py`` (Kaggle bootstrap) or directly
via ``python -m src.evaluation.baseline_eval --paths ... --config ...``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import tarfile
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.app.infer import build_models
from src.components.baseline_lesion_proposer import (
    BaselineLesionProposer,
    BaselineLesionProposerConfig,
)
from src.components.component0_qc import harmonise_sample
from src.components.component2_txv import TXV_CLASS_NAMES
from src.components.component7_boundary import score_boundary_quality
from src.components.component7_fp_auditor import estimate_fp_probability
from src.components.component7_refine import BaselineRefineConfig, refine_mask
from src.components.component8_timika import compute_baseline_timika
from src.core.device import describe_device, pick_device
from src.core.seed import seed_everything
from src.training.train_component1_dann import (
    DOMAIN_TO_ID,
    DomainSampleRef,
    _build_generic_image_samples,
    _build_nih_samples,
    _build_tbx11k_samples,
)


DOMAIN_IDS: tuple[str, ...] = ("montgomery", "shenzhen", "tbx11k", "nih_cxr14")
ID_TO_DOMAIN = {v: k for k, v in DOMAIN_TO_ID.items()}

# 8 TB-mimic classes TXV can predict; used for NIH pathology AUROC.
TB_MIMIC_CLASSES: tuple[str, ...] = (
    "consolidation",
    "infiltration",
    "fibrosis",
    "pleural_thickening",
    "effusion",
    "mass",
    "nodule",
    "pneumonia",
)

DEFAULT_SEED = 42
DEFAULT_HOLDOUT_FRAC = 0.2
DEFAULT_LIMIT_PER_DOMAIN = 200


@dataclass(slots=True)
class SplitEntry:
    dataset_id: str
    image_path: str
    archive_path: str | None = None
    member_name: str | None = None
    tb_label: int | None = None  # 1=TB, 0=non-TB, None=unknown


@dataclass(slots=True)
class ComponentRow:
    metric: str
    dataset: str
    value: float | None
    n: int
    notes: str = ""


@dataclass(slots=True)
class SystemRow:
    metric: str
    dataset: str
    value: float | None
    n: int
    notes: str = ""


@dataclass(slots=True)
class PerImageResult:
    dataset_id: str
    image_path: str
    image_basename: str
    tb_label: int | None
    original_size: tuple[int, int]  # (W, H)
    dom_pred: int
    dom_true: int
    pathology_probs: np.ndarray  # [18]
    lung_mask_1024: np.ndarray  # [1024, 1024] uint8
    lung_mask_256: np.ndarray  # [256, 256] uint8
    lesion_mask_coarse_256: np.ndarray  # [256, 256] uint8
    lesion_mask_refined_1024: np.ndarray  # [1024, 1024] uint8
    alp: float
    timika_score: float
    severity: str


# ---------- small helpers ----------


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _config_hash(config: dict[str, Any]) -> str:
    serialised = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(serialised.encode("utf-8")).hexdigest()[:12]


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    denom = int(p.sum()) + int(g.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * int(np.logical_and(p, g).sum()) / denom)


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return 1.0
    return float(int(np.logical_and(p, g).sum()) / union)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


# ---------- TB label conventions ----------


def _montgomery_or_shenzhen_tb_label(image_path: Path) -> int | None:
    """MCUCXR_0001_0.png → 0, MCUCXR_0001_1.png → 1."""
    parts = image_path.stem.split("_")
    if len(parts) >= 2 and parts[-1] in {"0", "1"}:
        return int(parts[-1])
    return None


def _tbx11k_tb_label(image_path: Path) -> int | None:
    """TBX11K layout: imgs/{health,sick,tb}/*.png. tb → 1, others → 0."""
    for part in image_path.parts:
        lowered = part.lower()
        if lowered in {"tb", "atb", "ltb"}:
            return 1
        if lowered in {"health", "sick", "non-tb", "non_tb"}:
            return 0
    return None


def _assign_tb_label(dataset_id: str, image_path: Path) -> int | None:
    if dataset_id in {"montgomery", "shenzhen"}:
        return _montgomery_or_shenzhen_tb_label(image_path)
    if dataset_id == "tbx11k":
        return _tbx11k_tb_label(image_path)
    return None


# ---------- manifest / splits ----------


def build_eval_manifest(
    *,
    montgomery_root: Path,
    shenzhen_root: Path,
    tbx11k_root: Path,
    nih_root: Path | None,
    tbx_list_name: str | None,
    nih_cache_path: Path,
) -> dict[str, list[DomainSampleRef]]:
    out: dict[str, list[DomainSampleRef]] = {d: [] for d in DOMAIN_IDS}
    out["montgomery"] = _build_generic_image_samples("montgomery", montgomery_root)
    out["shenzhen"] = _build_generic_image_samples("shenzhen", shenzhen_root)
    if tbx_list_name:
        out["tbx11k"] = _build_tbx11k_samples(tbx11k_root, tbx_list_name)
    else:
        out["tbx11k"] = _build_generic_image_samples("tbx11k", tbx11k_root / "imgs")
    if nih_root is not None and nih_root.exists():
        out["nih_cxr14"] = _build_nih_samples(
            nih_root, None, cache_path=nih_cache_path, metadata_csv=None,
        )
    return out


def make_test_splits(
    samples_by_domain: dict[str, list[DomainSampleRef]],
    *,
    seed: int = DEFAULT_SEED,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    limit_per_domain: int | None = DEFAULT_LIMIT_PER_DOMAIN,
    cache_path: Path | None = None,
) -> dict[str, list[SplitEntry]]:
    if cache_path is not None and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        loaded = {
            dom: [SplitEntry(**entry) for entry in entries]
            for dom, entries in payload.items()
        }
        # Cache is the full held-out split; slice to the current limit
        # so smoke / full / dry can share one cache file.
        if limit_per_domain is not None:
            loaded = {dom: entries[:limit_per_domain] for dom, entries in loaded.items()}
        return loaded

    splits: dict[str, list[SplitEntry]] = {}
    for dataset_id in DOMAIN_IDS:
        items = sorted(
            samples_by_domain.get(dataset_id, []),
            key=lambda ref: (ref.image_path or "", ref.source),
        )
        rng = np.random.default_rng(seed + DOMAIN_TO_ID[dataset_id])
        n = len(items)
        if n == 0:
            splits[dataset_id] = []
            continue
        n_holdout = max(1, int(round(n * holdout_frac)))
        indices = rng.choice(n, size=n_holdout, replace=False)
        indices.sort()
        picked = [items[i] for i in indices]

        entries: list[SplitEntry] = []
        for ref in picked:
            tb_label = None
            if ref.image_path:
                tb_label = _assign_tb_label(dataset_id, Path(ref.image_path))
            entries.append(
                SplitEntry(
                    dataset_id=dataset_id,
                    image_path=ref.image_path or "",
                    archive_path=ref.archive_path,
                    member_name=ref.member_name,
                    tb_label=tb_label,
                )
            )
        splits[dataset_id] = entries

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {dom: [asdict(e) for e in entries] for dom, entries in splits.items()},
                handle,
                indent=2,
            )
    if limit_per_domain is not None:
        splits = {dom: entries[:limit_per_domain] for dom, entries in splits.items()}
    return splits


# ---------- ground truth loaders ----------


def _load_montgomery_lung_gt_1024(image_path: Path) -> np.ndarray | None:
    """Combine Montgomery's leftMask + rightMask into a single [1024,1024] lung GT."""
    filename = image_path.name
    for ancestor in image_path.parents:
        left = ancestor / "ManualMask" / "leftMask" / filename
        right = ancestor / "ManualMask" / "rightMask" / filename
        if left.is_file() and right.is_file():
            with Image.open(left) as im:
                l = np.asarray(im.convert("L")) > 0
            with Image.open(right) as im:
                r = np.asarray(im.convert("L")) > 0
            combined = np.logical_or(l, r).astype(np.uint8)
            return _resize_mask_np(combined, 1024)
        if (ancestor / "MontgomerySet").is_dir() or ancestor.name == "MontgomerySet":
            break
    return None


def _load_shenzhen_lung_gt_1024(image_path: Path) -> np.ndarray | None:
    """Shenzhen (Stirenko): {root}/mask/{stem}_mask.png, if shipped."""
    stem = image_path.stem
    for ancestor in image_path.parents:
        for mask_dir_name in ("mask", "masks"):
            candidate = ancestor / mask_dir_name / f"{stem}_mask.png"
            if candidate.is_file():
                with Image.open(candidate) as im:
                    m = (np.asarray(im.convert("L")) > 0).astype(np.uint8)
                return _resize_mask_np(m, 1024)
        if (ancestor / "ChinaSet_AllFiles").is_dir() or ancestor.name == "ChinaSet_AllFiles":
            break
    return None


def _resize_mask_np(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize a 2-D 0/1 mask to size × size."""
    if mask.shape == (size, size):
        return mask.astype(np.uint8)
    t = torch.from_numpy(mask.astype(np.uint8)).unsqueeze(0).unsqueeze(0).float()
    r = F.interpolate(t, size=(size, size), mode="nearest")
    return r.squeeze().to(torch.uint8).numpy()


def load_tbx11k_bbox_index(tbx_root: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """Parse TBX11K COCO-style annotations into {basename → [(x1,y1,x2,y2), …]}.

    Scans any ``*.json`` under ``{tbx_root}/annotations/json``. Returns an
    empty dict if annotations aren't present — the lesion-proposer sanity
    metric will then be reported as N/A rather than crashing.
    """
    anno_dir = tbx_root / "annotations" / "json"
    if not anno_dir.is_dir():
        return {}
    result: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for json_path in sorted(anno_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        images = data.get("images") or []
        if not images:
            continue
        id_to_name = {int(img["id"]): Path(img["file_name"]).name for img in images if "id" in img and "file_name" in img}
        for ann in data.get("annotations") or []:
            image_id = ann.get("image_id")
            if image_id is None:
                continue
            name = id_to_name.get(int(image_id))
            if name is None:
                continue
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            result[name].append(
                (int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h)))
            )
    return dict(result)


def _rasterise_boxes_to_256(
    boxes: list[tuple[int, int, int, int]],
    original_size: tuple[int, int],
) -> np.ndarray:
    """Convert TBX bboxes (in original image coords) to a [256, 256] binary mask."""
    W, H = original_size
    mask = np.zeros((256, 256), dtype=np.uint8)
    if W <= 0 or H <= 0:
        return mask
    sx = 256.0 / float(W)
    sy = 256.0 / float(H)
    for x1, y1, x2, y2 in boxes:
        tx1 = max(0, min(256, int(round(x1 * sx))))
        ty1 = max(0, min(256, int(round(y1 * sy))))
        tx2 = max(0, min(256, int(round(x2 * sx))))
        ty2 = max(0, min(256, int(round(y2 * sy))))
        if tx2 > tx1 and ty2 > ty1:
            mask[ty1:ty2, tx1:tx2] = 1
    return mask


def load_nih_multilabels(nih_root: Path) -> dict[str, np.ndarray]:
    """Read Data_Entry_2017*.csv → {image_basename: multi-hot [len(TB_MIMIC_CLASSES)]}."""
    csv_path: Path | None = None
    for name in ("Data_Entry_2017_v2020.csv", "Data_Entry_2017.csv"):
        candidate = nih_root / name
        if candidate.is_file():
            csv_path = candidate
            break
    if csv_path is None:
        return {}

    def _norm(tag: str) -> str:
        return tag.strip().lower().replace(" ", "_").replace("-", "_")

    class_index = {name: idx for idx, name in enumerate(TB_MIMIC_CLASSES)}
    labels: dict[str, np.ndarray] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_index = (row.get("Image Index") or "").strip()
            if not image_index:
                continue
            vec = np.zeros(len(TB_MIMIC_CLASSES), dtype=np.float32)
            for tag in (row.get("Finding Labels") or "").split("|"):
                norm = _norm(tag)
                if norm in class_index:
                    vec[class_index[norm]] = 1.0
            labels[image_index] = vec
    return labels


# ---------- per-image pipeline forward ----------


def _load_image_for_entry(entry: SplitEntry) -> tuple[np.ndarray, tuple[int, int]]:
    """Load grayscale image + original ``(W, H)``."""
    if entry.image_path:
        with Image.open(entry.image_path) as image:
            return np.asarray(image.convert("L")), image.size
    if not (entry.archive_path and entry.member_name):
        raise ValueError(f"Cannot load image for entry {entry}")

    lower = entry.archive_path.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(entry.archive_path, mode="r") as archive:
            with archive.open(entry.member_name) as member:
                with Image.open(io.BytesIO(member.read())) as image:
                    return np.asarray(image.convert("L")), image.size
    mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r:"
    with tarfile.open(entry.archive_path, mode=mode) as archive:
        member = archive.extractfile(entry.member_name)
        if member is None:
            raise FileNotFoundError(
                f"Missing tar member {entry.member_name!r} in {entry.archive_path}"
            )
        with Image.open(io.BytesIO(member.read())) as image:
            return np.asarray(image.convert("L")), image.size


def _pipeline_forward(
    entry: SplitEntry,
    *,
    component1_model,
    component2_model,
    component4_model,
    proposer: BaselineLesionProposer,
    refine_cfg: BaselineRefineConfig,
    device: torch.device,
) -> PerImageResult:
    image_np, original_size = _load_image_for_entry(entry)
    harmonised = harmonise_sample(
        {"image": image_np, "dataset_id": entry.dataset_id}
    )
    x3 = harmonised.x_3ch.unsqueeze(0).to(device)
    x224 = harmonised.x_224.unsqueeze(0).to(device)
    x1024 = harmonised.x_1024.to(device)

    _, dom_logits = component1_model(x3, lambda_=0.0)
    txv = component2_model.forward_features(x224)
    lung = component4_model.predict_masks(x3)

    proposal = proposer.propose(
        x_224=x224,
        features_7x7=txv.features_7x7,
        pathology_logits=txv.pathology_logits,
        lung_mask_256=lung.lung_mask_256,
        classifier_weight=txv.classifier_weight,
    )

    coarse_256 = proposal.lesion_mask_coarse_256[0]  # [1, 256, 256]
    lung_256 = lung.lung_mask_256[0]  # [1, 256, 256]
    boundary = score_boundary_quality(coarse_256, lung_256, x1024)
    fp_audit = estimate_fp_probability(
        coarse_256, lung_256, txv.pathology_logits[0], class_names=txv.class_names,
    )
    refined_256 = refine_mask(
        coarse_256, lung_256, x1024,
        boundary.boundary_score, fp_audit.fp_probability, refine_cfg,
    )
    refined_1024 = F.interpolate(
        refined_256.unsqueeze(0), size=(1024, 1024), mode="nearest"
    ).squeeze(0)
    timika = compute_baseline_timika(refined_1024, lung.lung_mask_1024[0])

    image_basename = Path(entry.image_path or entry.member_name or "").name

    return PerImageResult(
        dataset_id=entry.dataset_id,
        image_path=entry.image_path or "",
        image_basename=image_basename,
        tb_label=entry.tb_label,
        original_size=original_size,
        dom_pred=int(dom_logits.argmax(dim=1).item()),
        dom_true=DOMAIN_TO_ID[entry.dataset_id],
        pathology_probs=torch.sigmoid(txv.pathology_logits).squeeze(0).detach().cpu().numpy(),
        lung_mask_1024=lung.lung_mask_1024[0, 0].detach().cpu().numpy().astype(np.uint8),
        lung_mask_256=lung.lung_mask_256[0, 0].detach().cpu().numpy().astype(np.uint8),
        lesion_mask_coarse_256=coarse_256.squeeze(0).detach().cpu().numpy().astype(np.uint8),
        lesion_mask_refined_1024=refined_1024[0].detach().cpu().numpy().astype(np.uint8),
        alp=float(timika.ALP),
        timika_score=float(timika.timika_score),
        severity=str(timika.severity),
    )


# ---------- metric aggregation ----------


def compute_component_metrics(
    results: list[PerImageResult],
    *,
    nih_labels: dict[str, np.ndarray],
    tbx_boxes: dict[str, list[tuple[int, int, int, int]]],
    shenzhen_mask_available: bool,
) -> list[ComponentRow]:
    rows: list[ComponentRow] = []

    # -- C1 domain-classifier accuracy (target ≈ 0.25 after DANN training) --
    correct_total, n_total = 0, 0
    for dom in DOMAIN_IDS:
        vals = [int(r.dom_pred == r.dom_true) for r in results if r.dataset_id == dom]
        if not vals:
            rows.append(ComponentRow("c1_domain_accuracy", dom, None, 0, "no samples"))
            continue
        rows.append(
            ComponentRow("c1_domain_accuracy", dom, sum(vals) / len(vals), len(vals))
        )
        correct_total += sum(vals)
        n_total += len(vals)
    rows.append(
        ComponentRow(
            "c1_domain_accuracy",
            "overall",
            (correct_total / n_total) if n_total else None,
            n_total,
            "target=0.25 (chance) after DANN; >0.25 means encoder is still domain-biased",
        )
    )

    # -- C4 lung Dice / IoU vs manual GT (Montgomery, Shenzhen if available) --
    for dom, loader in (
        ("montgomery", _load_montgomery_lung_gt_1024),
        ("shenzhen", _load_shenzhen_lung_gt_1024),
    ):
        if dom == "shenzhen" and not shenzhen_mask_available:
            rows.append(
                ComponentRow("c4_lung_dice", dom, None, 0, "Shenzhen GT masks not present")
            )
            rows.append(
                ComponentRow("c4_lung_iou", dom, None, 0, "Shenzhen GT masks not present")
            )
            continue
        dices: list[float] = []
        ious: list[float] = []
        for r in results:
            if r.dataset_id != dom or not r.image_path:
                continue
            gt = loader(Path(r.image_path))
            if gt is None:
                continue
            dices.append(_dice(r.lung_mask_1024, gt))
            ious.append(_iou(r.lung_mask_1024, gt))
        if dices:
            rows.append(ComponentRow("c4_lung_dice", dom, float(np.mean(dices)), len(dices)))
            rows.append(ComponentRow("c4_lung_iou", dom, float(np.mean(ious)), len(ious)))
        else:
            rows.append(ComponentRow("c4_lung_dice", dom, None, 0, "no GT masks resolved"))
            rows.append(ComponentRow("c4_lung_iou", dom, None, 0, "no GT masks resolved"))

    # -- C2 pathology AUROC on NIH (8 TB-mimic classes, macro) --
    if nih_labels:
        y_true_matrix: list[np.ndarray] = []
        y_score_matrix: list[np.ndarray] = []
        class_index_in_txv = {
            name: TXV_CLASS_NAMES.index(name)
            for name in TB_MIMIC_CLASSES
            if name in TXV_CLASS_NAMES
        }
        for r in results:
            if r.dataset_id != "nih_cxr14":
                continue
            gt_vec = nih_labels.get(r.image_basename)
            if gt_vec is None:
                continue
            y_true_matrix.append(gt_vec)
            y_score_matrix.append(
                np.array(
                    [r.pathology_probs[class_index_in_txv[name]] for name in TB_MIMIC_CLASSES],
                    dtype=np.float32,
                )
            )
        if y_true_matrix:
            y_true_arr = np.stack(y_true_matrix, axis=0)
            y_score_arr = np.stack(y_score_matrix, axis=0)
            per_class = []
            for idx in range(len(TB_MIMIC_CLASSES)):
                auc = _safe_auroc(y_true_arr[:, idx], y_score_arr[:, idx])
                if auc is not None:
                    per_class.append(auc)
            macro = float(np.mean(per_class)) if per_class else None
            rows.append(
                ComponentRow(
                    "c2_pathology_macro_auroc",
                    "nih_cxr14",
                    macro,
                    len(y_true_matrix),
                    f"macro over {len(per_class)} TB-mimic classes",
                )
            )
        else:
            rows.append(
                ComponentRow(
                    "c2_pathology_macro_auroc",
                    "nih_cxr14",
                    None,
                    0,
                    "no NIH labels matched sample basenames",
                )
            )
    else:
        rows.append(
            ComponentRow(
                "c2_pathology_macro_auroc",
                "nih_cxr14",
                None,
                0,
                "NIH metadata CSV not found",
            )
        )

    # -- Lesion proposer sanity vs TBX11K bbox pseudo-masks --
    if tbx_boxes:
        dices: list[float] = []
        ious: list[float] = []
        matched = 0
        for r in results:
            if r.dataset_id != "tbx11k":
                continue
            boxes = tbx_boxes.get(r.image_basename)
            if not boxes:
                continue
            gt_256 = _rasterise_boxes_to_256(boxes, r.original_size)
            pred_256 = (r.lesion_mask_coarse_256 > 0).astype(np.uint8)
            dices.append(_dice(pred_256, gt_256))
            ious.append(_iou(pred_256, gt_256))
            matched += 1
        if dices:
            rows.append(
                ComponentRow(
                    "lesion_proposer_dice",
                    "tbx11k",
                    float(np.mean(dices)),
                    matched,
                    "vs. bbox-rasterised pseudo-masks — baseline ceiling for MoE",
                )
            )
            rows.append(
                ComponentRow(
                    "lesion_proposer_iou", "tbx11k", float(np.mean(ious)), matched,
                )
            )
        else:
            rows.append(
                ComponentRow(
                    "lesion_proposer_dice",
                    "tbx11k",
                    None,
                    0,
                    "no TBX11K bbox GT matched test samples",
                )
            )
    else:
        rows.append(
            ComponentRow(
                "lesion_proposer_dice",
                "tbx11k",
                None,
                0,
                "TBX11K annotations/json not found",
            )
        )

    return rows


def compute_system_metrics(results: list[PerImageResult]) -> list[SystemRow]:
    rows: list[SystemRow] = []

    # -- ALP distribution per domain --
    for dom in DOMAIN_IDS:
        vals = np.array([r.alp for r in results if r.dataset_id == dom], dtype=np.float64)
        if len(vals) == 0:
            rows.append(SystemRow("alp_mean", dom, None, 0, "no samples"))
            continue
        rows.append(SystemRow("alp_mean", dom, float(vals.mean()), len(vals)))
        rows.append(SystemRow("alp_std", dom, float(vals.std()), len(vals)))
        rows.append(SystemRow("alp_p50", dom, float(np.percentile(vals, 50)), len(vals)))
        rows.append(SystemRow("alp_p95", dom, float(np.percentile(vals, 95)), len(vals)))

    # -- Timika-AUROC per domain (domains with TB labels only) --
    for dom in DOMAIN_IDS:
        labelled = [
            (r.timika_score, r.tb_label)
            for r in results
            if r.dataset_id == dom and r.tb_label is not None
        ]
        if not labelled:
            rows.append(
                SystemRow(
                    "timika_auroc",
                    dom,
                    None,
                    0,
                    "no TB/non-TB labels (expected for nih_cxr14)",
                )
            )
            continue
        y_score = np.array([x for x, _ in labelled], dtype=np.float64)
        y_true = np.array([y for _, y in labelled], dtype=np.int32)
        auc = _safe_auroc(y_true, y_score)
        note = "" if auc is not None else "only one class present in test split"
        rows.append(SystemRow("timika_auroc", dom, auc, len(labelled), note))

    # -- Report faithfulness (deterministic template → 1.0 by construction) --
    rows.append(
        SystemRow(
            "report_faithfulness",
            "all",
            1.0,
            len(results),
            "deterministic template floor; MoE will replace with learned generator",
        )
    )

    return rows


# ---------- top-level entrypoint ----------


def _write_rows_csv(rows: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "dataset", "value", "n", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_baseline_evaluation(
    *,
    baseline_config_path: Path,
    paths_config_path: Path,
    output_dir: Path,
    limit_per_domain: int = DEFAULT_LIMIT_PER_DOMAIN,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    seed: int = DEFAULT_SEED,
    tbx_list_name: str | None = "all_trainval.txt",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run the full baseline evaluation; write CSVs + summary JSON to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    with baseline_config_path.open("r", encoding="utf-8") as handle:
        baseline_config = yaml.safe_load(handle) or {}
    with paths_config_path.open("r", encoding="utf-8") as handle:
        paths_config = yaml.safe_load(handle) or {}

    device = pick_device(baseline_config.get("runtime", {}).get("device"))
    print(f"Baseline eval device: {describe_device(device)}")

    dataset_roots = paths_config.get("datasets", {})
    montgomery_root = Path(dataset_roots["montgomery"])
    shenzhen_root = Path(dataset_roots["shenzhen"])
    tbx11k_root = Path(dataset_roots["tbx11k"])
    nih_root_raw = dataset_roots.get("nih_cxr14")
    nih_root = Path(nih_root_raw) if nih_root_raw and Path(nih_root_raw).exists() else None

    print("Building manifests …")
    nih_cache = output_dir / "_nih_index_cache.json"
    samples_by_domain = build_eval_manifest(
        montgomery_root=montgomery_root,
        shenzhen_root=shenzhen_root,
        tbx11k_root=tbx11k_root,
        nih_root=nih_root,
        tbx_list_name=tbx_list_name,
        nih_cache_path=nih_cache,
    )
    for dom in DOMAIN_IDS:
        print(f"  {dom:<11}: {len(samples_by_domain.get(dom, []))} total samples")

    split_cache = output_dir / "test_splits.json"
    splits = make_test_splits(
        samples_by_domain,
        seed=seed,
        holdout_frac=holdout_frac,
        limit_per_domain=limit_per_domain,
        cache_path=split_cache,
    )
    for dom in DOMAIN_IDS:
        print(f"  held-out {dom:<11}: {len(splits.get(dom, []))} images")

    print("Loading ground truth (NIH metadata, TBX11K bboxes) …")
    nih_labels = load_nih_multilabels(nih_root) if nih_root else {}
    print(f"  NIH multilabels: {len(nih_labels)} images")
    tbx_boxes = load_tbx11k_bbox_index(tbx11k_root)
    print(f"  TBX11K bbox GT : {len(tbx_boxes)} images")

    # Check for Shenzhen masks by probing the first held-out sample.
    shenzhen_mask_available = False
    for entry in splits.get("shenzhen", []):
        if entry.image_path and _load_shenzhen_lung_gt_1024(Path(entry.image_path)) is not None:
            shenzhen_mask_available = True
            break

    print("Building pipeline …")
    component1_model, component2_model, component4_model = build_models(baseline_config, device)
    component1_model.eval()
    component2_model.eval()
    component4_model.eval()

    proposer = BaselineLesionProposer(
        BaselineLesionProposerConfig(
            suspicious_class_threshold=float(
                baseline_config.get("baseline_lesion_proposer", {}).get("suspicious_class_threshold", 0.55)
            ),
            fixed_binary_threshold=baseline_config.get("baseline_lesion_proposer", {}).get("fixed_binary_threshold"),
            min_region_px=int(baseline_config.get("baseline_lesion_proposer", {}).get("min_region_px", 48)),
            opening_iters=int(baseline_config.get("baseline_lesion_proposer", {}).get("opening_iters", 1)),
            closing_iters=int(baseline_config.get("baseline_lesion_proposer", {}).get("closing_iters", 1)),
            fallback_blend=float(baseline_config.get("baseline_lesion_proposer", {}).get("fallback_blend", 0.35)),
        )
    )
    refine_cfg = BaselineRefineConfig(
        min_area_px=int(baseline_config.get("component7_refine", {}).get("min_area_px", 48)),
        opening_iters=int(baseline_config.get("component7_refine", {}).get("opening_iters", 1)),
        closing_iters=int(baseline_config.get("component7_refine", {}).get("closing_iters", 1)),
        weak_boundary_threshold=float(baseline_config.get("component7_refine", {}).get("weak_boundary_threshold", 0.45)),
        suppress_fp_threshold=float(baseline_config.get("component7_refine", {}).get("suppress_fp_threshold", 0.85)),
        caution_fp_threshold=float(baseline_config.get("component7_refine", {}).get("caution_fp_threshold", 0.65)),
    )

    print("Running pipeline on held-out images …")
    total = sum(len(v) for v in splits.values())
    results: list[PerImageResult] = []
    skipped: list[dict[str, str]] = []
    start_time = time.time()
    processed = 0
    for dom in DOMAIN_IDS:
        for entry in splits.get(dom, []):
            processed += 1
            try:
                with torch.no_grad():
                    result = _pipeline_forward(
                        entry,
                        component1_model=component1_model,
                        component2_model=component2_model,
                        component4_model=component4_model,
                        proposer=proposer,
                        refine_cfg=refine_cfg,
                        device=device,
                    )
                results.append(result)
            except Exception as exc:  # noqa: BLE001 — we never want one bad image to kill the run
                skipped.append(
                    {
                        "dataset_id": dom,
                        "image_path": entry.image_path,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if processed % 20 == 0 or processed == total:
                elapsed = time.time() - start_time
                rate = processed / max(elapsed, 1e-6)
                eta = (total - processed) / max(rate, 1e-6)
                print(
                    f"  [{processed:>4}/{total}] elapsed={elapsed:6.1f}s "
                    f"rate={rate:4.2f} img/s eta={eta:6.1f}s"
                )

    print(f"Finished pipeline. {len(results)} ok, {len(skipped)} skipped.")

    print("Computing component-level metrics …")
    component_rows = compute_component_metrics(
        results,
        nih_labels=nih_labels,
        tbx_boxes=tbx_boxes,
        shenzhen_mask_available=shenzhen_mask_available,
    )
    print("Computing system-level metrics …")
    system_rows = compute_system_metrics(results)

    components_csv = output_dir / "baseline_components.csv"
    system_csv = output_dir / "baseline_system.csv"
    _write_rows_csv(component_rows, components_csv)
    _write_rows_csv(system_rows, system_csv)

    # Per-image results (so paired bootstrap vs. MoE is possible later).
    per_image_csv = output_dir / "baseline_per_image.csv"
    per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    with per_image_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_id",
                "image_basename",
                "tb_label",
                "dom_pred",
                "dom_true",
                "alp",
                "timika_score",
                "severity",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.dataset_id,
                    r.image_basename,
                    r.tb_label if r.tb_label is not None else "",
                    r.dom_pred,
                    r.dom_true,
                    f"{r.alp:.6f}",
                    f"{r.timika_score:.6f}",
                    r.severity,
                ]
            )

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(repo_root) if repo_root is not None else None,
        "baseline_config_hash": _config_hash(baseline_config),
        "paths_config_hash": _config_hash(paths_config),
        "device": describe_device(device),
        "seed": seed,
        "holdout_frac": holdout_frac,
        "limit_per_domain": limit_per_domain,
        "components": {
            "component1_backend": getattr(component1_model.encoder, "active_backend", None),
            "component1_adapter_path": getattr(component1_model, "loaded_adapter_path", None),
            "component2_backend": component2_model.active_backend,
            "component4_backend": component4_model.active_backend,
            "component4_decoder_ckpt": getattr(component4_model, "loaded_decoder_checkpoint", None),
        },
        "counts": {
            "manifest": {dom: len(samples_by_domain.get(dom, [])) for dom in DOMAIN_IDS},
            "test_split": {dom: len(splits.get(dom, [])) for dom in DOMAIN_IDS},
            "ran_ok": len(results),
            "skipped": len(skipped),
        },
        "ground_truth_coverage": {
            "nih_multilabels_indexed": len(nih_labels),
            "tbx_bbox_images_indexed": len(tbx_boxes),
            "shenzhen_masks_found": shenzhen_mask_available,
        },
        "skipped_samples": skipped[:50],  # cap so the json stays small
        "output_files": {
            "components_csv": str(components_csv),
            "system_csv": str(system_csv),
            "per_image_csv": str(per_image_csv),
            "test_splits": str(split_cache),
        },
    }
    summary_path = output_dir / "baseline_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    # Pretty-print the key numbers so the notebook output itself tells the story.
    print("\n" + "=" * 70)
    print("COMPONENT-LEVEL METRICS")
    print("=" * 70)
    for row in component_rows:
        value_str = f"{row.value:.4f}" if isinstance(row.value, float) else "   N/A"
        print(f"  {row.metric:<28} {row.dataset:<13} {value_str}   n={row.n:<5} {row.notes}")
    print("\n" + "=" * 70)
    print("SYSTEM-LEVEL METRICS")
    print("=" * 70)
    for row in system_rows:
        value_str = f"{row.value:.4f}" if isinstance(row.value, float) else "   N/A"
        print(f"  {row.metric:<28} {row.dataset:<13} {value_str}   n={row.n:<5} {row.notes}")
    print("=" * 70)
    print(f"Wrote: {components_csv}")
    print(f"Wrote: {system_csv}")
    print(f"Wrote: {per_image_csv}")
    print(f"Wrote: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline TB CXR pipeline evaluation.")
    parser.add_argument("--config", required=True, help="Path to baseline.yaml.")
    parser.add_argument("--paths", required=True, help="Path to paths.yaml.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-per-domain", type=int, default=DEFAULT_LIMIT_PER_DOMAIN)
    parser.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tbx-list", default="all_trainval.txt")
    parser.add_argument("--repo-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baseline_evaluation(
        baseline_config_path=Path(args.config),
        paths_config_path=Path(args.paths),
        output_dir=Path(args.output_dir),
        limit_per_domain=args.limit_per_domain,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        tbx_list_name=args.tbx_list,
        repo_root=Path(args.repo_root) if args.repo_root else None,
    )


if __name__ == "__main__":
    main()
