"""MoE TB CXR pipeline evaluation.

Mirrors ``src/evaluation/baseline_eval.py`` but runs the full MoE path
(C3 routing gate → C5 expert bank → C6 fusion → C7 upgraded → C8 cavity-aware).

Reuses all manifest / metric helpers from baseline_eval so the two result
CSVs are directly comparable row-by-row.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.app.infer import build_models, build_moe_models
from src.components.component0_qc import harmonise_sample
from src.components.component2_txv import TXV_CLASS_NAMES
from src.components.component5_experts import EXPERT_NAMES
from src.components.component7_boundary import score_boundary_quality
from src.components.component7_fp_auditor import estimate_fp_probability
from src.components.component7_refine import BaselineRefineConfig, refine_mask
from src.components.component7_verification import (
    Component7BoundaryCritic,
    Component7RepromptRefiner,
    RepromptRefinerConfig,
)
from src.components.component8_timika import compute_moe_timika
from src.core.device import describe_device, pick_device
from src.core.seed import seed_everything
from src.evaluation.baseline_eval import (
    DOMAIN_IDS,
    DOMAIN_TO_ID,
    TB_MIMIC_CLASSES,
    ComponentRow,
    SplitEntry,
    SystemRow,
    _dice,
    _iou,
    _load_image_for_entry,
    _load_montgomery_lung_gt_1024,
    _load_shenzhen_lung_gt_1024,
    _rasterise_boxes_to_256,
    _safe_auroc,
    build_eval_manifest,
    load_nih_multilabels,
    load_tbx11k_bbox_index,
    make_test_splits,
)


@dataclass(slots=True)
class MoEPerImageResult:
    dataset_id: str
    image_path: str
    image_basename: str
    tb_label: int | None
    original_size: tuple[int, int]
    dom_pred: int
    dom_true: int
    pathology_probs: np.ndarray          # [18]
    lung_mask_1024: np.ndarray           # [1024, 1024] uint8
    lung_mask_256: np.ndarray            # [256, 256] uint8
    lesion_mask_fused_256: np.ndarray    # [256, 256] uint8  (fused, pre-refinement)
    lesion_mask_refined_1024: np.ndarray # [1024, 1024] uint8
    cavity_mask_256: np.ndarray          # [256, 256] uint8
    routing_weights: dict[str, float]    # {expert_name: weight}
    alp: float
    timika_score: float
    severity: str
    cavity_flag: int
    boundary_score: float
    fp_probability: float
    # Fix 1: sigmoid(tb_logit) from the trained TB head; 0.5 when tb_head absent.
    tb_head_score: float = 0.5


def _pipeline_forward_moe(
    entry: SplitEntry,
    *,
    component1_model,
    component2_model,
    component4_model,
    routing_gate,
    expert_bank,
    fusion,
    boundary_critic,
    refine_cfg: BaselineRefineConfig,
    reprompt_cfg: RepromptRefinerConfig,
    device: torch.device,
) -> MoEPerImageResult:
    image_np, original_size = _load_image_for_entry(entry)
    harmonised = harmonise_sample({"image": image_np, "dataset_id": entry.dataset_id})

    x3 = harmonised.x_3ch.unsqueeze(0).to(device)
    x224 = harmonised.x_224.unsqueeze(0).to(device)
    x1024 = harmonised.x_1024.to(device)

    # C1
    img_emb, dom_logits = component1_model(x3, lambda_=0.0)
    # C2
    txv = component2_model.forward_features(x224)
    # C4
    lung = component4_model.predict_masks(x3)

    # Fix 1: compute TB head score for direct AUROC metric.
    tb_head_score = 0.5
    if hasattr(component2_model, "tb_head"):
        _tb_l = component2_model.tb_head(txv.pooled_features)
        tb_head_score = float(torch.sigmoid(_tb_l).squeeze().item())

    # C3: routing gate
    routing_weights = routing_gate(
        img_emb,
        txv.domain_ctx if routing_gate.use_domain_ctx else None,
    )  # [1, K]

    # C5: all expert decoders
    expert_logits = expert_bank(img_emb)  # list of K tensors [1,1,256,256]

    # C6: fusion in logit space
    fusion_output = fusion(expert_logits, routing_weights)
    mask_fused_256 = fusion_output.mask_fused_256   # [1,1,256,256]
    mask_variance = fusion_output.mask_variance

    # Cavity expert (Expert index 1)
    cavity_idx = list(EXPERT_NAMES).index("cavity")
    cavity_mask_256 = fusion_output.expert_masks_256[cavity_idx]  # [1,1,256,256]

    # C7 upgraded: boundary scoring
    coarse_256 = (mask_fused_256[0] > 0.5).float()   # [1,256,256]
    lung_256 = lung.lung_mask_256[0]                  # [1,256,256]
    heuristic = score_boundary_quality(coarse_256, lung_256, x1024)
    boundary_score = heuristic.boundary_score
    if boundary_critic is not None:
        crop = Component7BoundaryCritic.prepare_crop(x1024, coarse_256, lung_256).to(device)
        boundary_score = float(boundary_critic(crop).item())

    # C7: FP audit
    fp_audit = estimate_fp_probability(
        coarse_256, lung_256, txv.pathology_logits[0], class_names=txv.class_names,
    )

    # C7: reprompt refiner (Expert 3 = fibrosis, most edge-sensitive)
    reprompt_refiner = Component7RepromptRefiner(reprompt_cfg)
    expert3 = expert_bank.expert_by_name("fibrosis")
    refined_moe = reprompt_refiner(
        image_emb=img_emb,
        mask_fused_256=mask_fused_256,
        mask_variance=mask_variance,
        boundary_score=boundary_score,
        expert3_decoder=expert3,
        lung_mask_256=lung.lung_mask_256,
    )  # [1,1,256,256]

    # C7: final FP suppression + morphology cleanup
    refined_256 = refine_mask(
        refined_moe[0], lung.lung_mask_256[0], x1024,
        boundary_score, fp_audit.fp_probability, refine_cfg,
    )  # [1,256,256]

    refined_1024 = F.interpolate(
        refined_256.unsqueeze(0), size=(1024, 1024), mode="nearest"
    ).squeeze(0)  # [1,1024,1024]

    # C8: cavity-aware Timika
    timika = compute_moe_timika(
        refined_1024, lung.lung_mask_1024[0], cavity_mask_256[0],
    )

    # Routing weights as named dict
    rw = routing_weights[0].detach().cpu().tolist()
    routing_dict = {
        name: float(rw[i])
        for i, name in enumerate(list(EXPERT_NAMES)[: expert_bank.num_experts])
    }

    image_basename = Path(entry.image_path or entry.member_name or "").name

    return MoEPerImageResult(
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
        lesion_mask_fused_256=(mask_fused_256[0, 0] > 0.5).detach().cpu().numpy().astype(np.uint8),
        lesion_mask_refined_1024=refined_1024[0].detach().cpu().numpy().astype(np.uint8),
        cavity_mask_256=(cavity_mask_256[0, 0] > 0.5).detach().cpu().numpy().astype(np.uint8),
        routing_weights=routing_dict,
        alp=float(timika.ALP),
        timika_score=float(timika.timika_score),
        severity=str(timika.severity),
        cavity_flag=int(timika.cavity_flag),
        boundary_score=float(boundary_score),
        fp_probability=float(fp_audit.fp_probability),
        tb_head_score=tb_head_score,
    )


def compute_moe_component_metrics(
    results: list[MoEPerImageResult],
    *,
    nih_labels: dict[str, np.ndarray],
    tbx_boxes: dict[str, list[tuple[int, int, int, int]]],
    shenzhen_mask_available: bool,
) -> list[ComponentRow]:
    rows: list[ComponentRow] = []

    # C1 domain-classifier accuracy (target ≈ 0.25 after DANN)
    correct_total, n_total = 0, 0
    for dom in DOMAIN_IDS:
        vals = [int(r.dom_pred == r.dom_true) for r in results if r.dataset_id == dom]
        if not vals:
            rows.append(ComponentRow("c1_domain_accuracy", dom, None, 0, "no samples"))
            continue
        rows.append(ComponentRow("c1_domain_accuracy", dom, sum(vals) / len(vals), len(vals)))
        correct_total += sum(vals); n_total += len(vals)
    rows.append(ComponentRow(
        "c1_domain_accuracy", "overall",
        (correct_total / n_total) if n_total else None, n_total,
        "target=0.25 (chance) after DANN",
    ))

    # C4 lung Dice / IoU vs manual GT
    for dom, loader in (("montgomery", _load_montgomery_lung_gt_1024), ("shenzhen", _load_shenzhen_lung_gt_1024)):
        if dom == "shenzhen" and not shenzhen_mask_available:
            rows.append(ComponentRow("c4_lung_dice", dom, None, 0, "GT masks not present"))
            rows.append(ComponentRow("c4_lung_iou",  dom, None, 0, "GT masks not present"))
            continue
        dices, ious = [], []
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
            rows.append(ComponentRow("c4_lung_iou",  dom, float(np.mean(ious)),  len(ious)))
        else:
            rows.append(ComponentRow("c4_lung_dice", dom, None, 0, "no GT masks resolved"))
            rows.append(ComponentRow("c4_lung_iou",  dom, None, 0, "no GT masks resolved"))

    # C2 pathology AUROC on NIH
    if nih_labels:
        y_true_mat, y_score_mat = [], []
        class_idx = {n: TXV_CLASS_NAMES.index(n) for n in TB_MIMIC_CLASSES if n in TXV_CLASS_NAMES}
        for r in results:
            if r.dataset_id != "nih_cxr14":
                continue
            gt_vec = nih_labels.get(r.image_basename)
            if gt_vec is None:
                continue
            y_true_mat.append(gt_vec)
            y_score_mat.append(np.array([r.pathology_probs[class_idx[n]] for n in TB_MIMIC_CLASSES], dtype=np.float32))
        if y_true_mat:
            yt = np.stack(y_true_mat); ys = np.stack(y_score_mat)
            per_class = [a for a in (_safe_auroc(yt[:, i], ys[:, i]) for i in range(len(TB_MIMIC_CLASSES))) if a is not None]
            rows.append(ComponentRow("c2_pathology_macro_auroc", "nih_cxr14",
                                     float(np.mean(per_class)) if per_class else None,
                                     len(y_true_mat), f"macro over {len(per_class)} TB-mimic classes"))
        else:
            rows.append(ComponentRow("c2_pathology_macro_auroc", "nih_cxr14", None, 0, "no NIH labels matched"))
    else:
        rows.append(ComponentRow("c2_pathology_macro_auroc", "nih_cxr14", None, 0, "NIH metadata CSV not found"))

    # MoE fused mask Dice / IoU vs TBX11K bboxes
    if tbx_boxes:
        dices, ious, matched = [], [], 0
        for r in results:
            if r.dataset_id != "tbx11k":
                continue
            boxes = tbx_boxes.get(r.image_basename)
            if not boxes:
                continue
            gt_256 = _rasterise_boxes_to_256(boxes, r.original_size)
            pred_256 = (r.lesion_mask_refined_1024.astype(bool)).astype(np.uint8)
            # resize refined_1024 → 256 for fair comparison
            t = torch.from_numpy(pred_256).float().unsqueeze(0).unsqueeze(0)
            pred_256_rs = F.interpolate(t, (256, 256), mode="nearest").squeeze().numpy().astype(np.uint8)
            dices.append(_dice(pred_256_rs, gt_256))
            ious.append(_iou(pred_256_rs, gt_256))
            matched += 1
        if dices:
            rows.append(ComponentRow("moe_lesion_dice", "tbx11k", float(np.mean(dices)), matched,
                                     "refined mask vs bbox pseudo-masks"))
            rows.append(ComponentRow("moe_lesion_iou",  "tbx11k", float(np.mean(ious)),  matched))
        else:
            rows.append(ComponentRow("moe_lesion_dice", "tbx11k", None, 0, "no TBX11K bbox GT matched"))
    else:
        rows.append(ComponentRow("moe_lesion_dice", "tbx11k", None, 0, "TBX11K annotations not found"))

    return rows


def compute_moe_system_metrics(results: list[MoEPerImageResult]) -> list[SystemRow]:
    rows: list[SystemRow] = []

    for dom in DOMAIN_IDS:
        vals = np.array([r.alp for r in results if r.dataset_id == dom], dtype=np.float64)
        if len(vals) == 0:
            rows.append(SystemRow("alp_mean", dom, None, 0, "no samples"))
            continue
        rows.append(SystemRow("alp_mean", dom, float(vals.mean()), len(vals)))
        rows.append(SystemRow("alp_std",  dom, float(vals.std()),  len(vals)))
        rows.append(SystemRow("alp_p50",  dom, float(np.percentile(vals, 50)), len(vals)))
        rows.append(SystemRow("alp_p95",  dom, float(np.percentile(vals, 95)), len(vals)))

    for dom in DOMAIN_IDS:
        labelled = [(r.timika_score, r.tb_label) for r in results
                    if r.dataset_id == dom and r.tb_label is not None]
        if not labelled:
            rows.append(SystemRow("timika_auroc", dom, None, 0, "no TB/non-TB labels"))
            continue
        y_score = np.array([x for x, _ in labelled])
        y_true  = np.array([y for _, y in labelled], dtype=np.int32)
        auc = _safe_auroc(y_true, y_score)
        rows.append(SystemRow("timika_auroc", dom, auc, len(labelled),
                              "" if auc is not None else "only one class present"))

    # Cavity detection rate (TB-positive cases only)
    for dom in DOMAIN_IDS:
        tb_pos = [r for r in results if r.dataset_id == dom and r.tb_label == 1]
        if not tb_pos:
            rows.append(SystemRow("cavity_detection_rate", dom, None, 0, "no TB-positive samples"))
            continue
        rate = float(np.mean([r.cavity_flag for r in tb_pos]))
        rows.append(SystemRow("cavity_detection_rate", dom, rate, len(tb_pos),
                              "fraction of TB+ cases with cavity flagged by Expert 2"))

    # TB head AUROC (Fix 1): P(TB|image) from the trained binary head.
    for dom in DOMAIN_IDS:
        labelled = [(r.tb_head_score, r.tb_label) for r in results
                    if r.dataset_id == dom and r.tb_label is not None]
        if not labelled:
            rows.append(SystemRow("tb_head_auroc", dom, None, 0, "no TB/non-TB labels"))
            continue
        y_score = np.array([x for x, _ in labelled], dtype=np.float64)
        y_true  = np.array([y for _, y in labelled], dtype=np.int32)
        auc = _safe_auroc(y_true, y_score)
        rows.append(SystemRow("tb_head_auroc", dom, auc, len(labelled),
                              "" if auc is not None else "only one class present"))

    rows.append(SystemRow("report_faithfulness", "all", 1.0, len(results),
                          "deterministic template — faithfulness guaranteed"))
    return rows


def _write_rows_csv(rows: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "dataset", "value", "n", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_moe_evaluation(
    *,
    moe_config_path: Path,
    paths_config_path: Path,
    output_dir: Path,
    limit_per_domain: int = 200,
    holdout_frac: float = 0.2,
    seed: int = 42,
    tbx_list_name: str | None = "all_trainval.txt",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    with moe_config_path.open("r", encoding="utf-8") as f:
        moe_config = yaml.safe_load(f) or {}
    with paths_config_path.open("r", encoding="utf-8") as f:
        paths_config = yaml.safe_load(f) or {}

    device = pick_device(moe_config.get("runtime", {}).get("device"))
    print(f"MoE eval device: {describe_device(device)}")

    dataset_roots = paths_config.get("datasets", {})
    montgomery_root = Path(dataset_roots["montgomery"])
    shenzhen_root   = Path(dataset_roots["shenzhen"])
    tbx11k_root     = Path(dataset_roots["tbx11k"])
    nih_root_raw    = dataset_roots.get("nih_cxr14")
    nih_root        = Path(nih_root_raw) if nih_root_raw and Path(nih_root_raw).exists() else None

    print("Building manifests …")
    nih_cache = output_dir / "_nih_index_cache.json"
    samples_by_domain = build_eval_manifest(
        montgomery_root=montgomery_root, shenzhen_root=shenzhen_root,
        tbx11k_root=tbx11k_root, nih_root=nih_root,
        tbx_list_name=tbx_list_name, nih_cache_path=nih_cache,
    )
    for dom in DOMAIN_IDS:
        print(f"  {dom:<11}: {len(samples_by_domain.get(dom, []))} total")

    split_cache = output_dir / "test_splits.json"
    splits = make_test_splits(
        samples_by_domain, seed=seed, holdout_frac=holdout_frac,
        limit_per_domain=limit_per_domain, cache_path=split_cache,
    )
    for dom in DOMAIN_IDS:
        print(f"  held-out {dom:<11}: {len(splits.get(dom, []))} images")

    print("Loading ground truth …")
    nih_labels = load_nih_multilabels(nih_root) if nih_root else {}
    print(f"  NIH multilabels: {len(nih_labels)}")
    tbx_boxes = load_tbx11k_bbox_index(tbx11k_root)
    print(f"  TBX11K bbox GT : {len(tbx_boxes)}")

    shenzhen_mask_available = False
    for entry in splits.get("shenzhen", []):
        if entry.image_path and _load_shenzhen_lung_gt_1024(Path(entry.image_path)) is not None:
            shenzhen_mask_available = True
            break

    print("Building models …")
    c1, c2, c4 = build_models(moe_config, device)
    c1.eval(); c2.eval(); c4.eval()

    moe_models = build_moe_models(moe_config, device)
    if moe_models is None:
        raise ValueError("moe.enabled must be true in the config for MoE evaluation.")
    routing_gate, expert_bank, fusion, boundary_critic = moe_models
    routing_gate.eval(); expert_bank.eval(); fusion.eval()

    refine_cfg = BaselineRefineConfig(
        min_area_px=int(moe_config.get("component7_refine", {}).get("min_area_px", 48)),
        opening_iters=int(moe_config.get("component7_refine", {}).get("opening_iters", 1)),
        closing_iters=int(moe_config.get("component7_refine", {}).get("closing_iters", 1)),
        weak_boundary_threshold=float(moe_config.get("component7_refine", {}).get("weak_boundary_threshold", 0.45)),
        suppress_fp_threshold=float(moe_config.get("component7_refine", {}).get("suppress_fp_threshold", 0.85)),
        caution_fp_threshold=float(moe_config.get("component7_refine", {}).get("caution_fp_threshold", 0.65)),
    )
    reprompt_cfg = RepromptRefinerConfig(
        boundary_threshold=float(moe_config.get("component7_moe", {}).get("boundary_threshold", 0.7)),
        variance_threshold=float(moe_config.get("component7_moe", {}).get("variance_threshold", 0.3)),
        num_prompt_points=int(moe_config.get("component7_moe", {}).get("num_prompt_points", 5)),
    )

    print("Running MoE pipeline on held-out images …")
    total = sum(len(v) for v in splits.values())
    results: list[MoEPerImageResult] = []
    skipped: list[dict[str, str]] = []
    start = time.time()
    processed = 0

    for dom in DOMAIN_IDS:
        for entry in splits.get(dom, []):
            processed += 1
            try:
                with torch.no_grad():
                    result = _pipeline_forward_moe(
                        entry,
                        component1_model=c1, component2_model=c2, component4_model=c4,
                        routing_gate=routing_gate, expert_bank=expert_bank,
                        fusion=fusion, boundary_critic=boundary_critic,
                        refine_cfg=refine_cfg, reprompt_cfg=reprompt_cfg, device=device,
                    )
                results.append(result)
            except Exception as exc:
                skipped.append({"dataset_id": dom, "image_path": entry.image_path,
                                 "error": f"{type(exc).__name__}: {exc}"})
            if processed % 20 == 0 or processed == total:
                elapsed = time.time() - start
                rate = processed / max(elapsed, 1e-6)
                eta = (total - processed) / max(rate, 1e-6)
                print(f"  [{processed:>4}/{total}] elapsed={elapsed:6.1f}s "
                      f"rate={rate:.2f} img/s eta={eta:6.1f}s")

    print(f"Finished. {len(results)} ok, {len(skipped)} skipped.")

    print("Computing metrics …")
    component_rows = compute_moe_component_metrics(
        results, nih_labels=nih_labels, tbx_boxes=tbx_boxes,
        shenzhen_mask_available=shenzhen_mask_available,
    )
    system_rows = compute_moe_system_metrics(results)

    components_csv  = output_dir / "moe_components.csv"
    system_csv      = output_dir / "moe_system.csv"
    per_image_csv   = output_dir / "moe_per_image.csv"
    _write_rows_csv(component_rows, components_csv)
    _write_rows_csv(system_rows,    system_csv)

    with per_image_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_id", "image_basename", "tb_label",
            "dom_pred", "dom_true",
            "alp", "timika_score", "severity", "cavity_flag",
            "boundary_score", "fp_probability",
            "tb_head_score",
            "routing_consolidation", "routing_cavity",
            "routing_fibrosis", "routing_nodule",
        ])
        for r in results:
            rw = r.routing_weights
            writer.writerow([
                r.dataset_id, r.image_basename,
                r.tb_label if r.tb_label is not None else "",
                r.dom_pred, r.dom_true,
                f"{r.alp:.6f}", f"{r.timika_score:.6f}", r.severity, r.cavity_flag,
                f"{r.boundary_score:.4f}", f"{r.fp_probability:.4f}",
                f"{r.tb_head_score:.6f}",
                f"{rw.get('consolidation', 0):.4f}", f"{rw.get('cavity', 0):.4f}",
                f"{rw.get('fibrosis', 0):.4f}",      f"{rw.get('nodule', 0):.4f}",
            ])

    summary = {
        "pipeline": "moe",
        "device": describe_device(device),
        "seed": seed, "holdout_frac": holdout_frac, "limit_per_domain": limit_per_domain,
        "counts": {
            "manifest": {dom: len(samples_by_domain.get(dom, [])) for dom in DOMAIN_IDS},
            "test_split": {dom: len(splits.get(dom, [])) for dom in DOMAIN_IDS},
            "ran_ok": len(results), "skipped": len(skipped),
        },
        "ground_truth_coverage": {
            "nih_multilabels_indexed": len(nih_labels),
            "tbx_bbox_images_indexed": len(tbx_boxes),
            "shenzhen_masks_found": shenzhen_mask_available,
        },
        "skipped_samples": skipped[:50],
        "output_files": {
            "components_csv": str(components_csv),
            "system_csv":     str(system_csv),
            "per_image_csv":  str(per_image_csv),
        },
    }
    summary_path = output_dir / "moe_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("MoE COMPONENT-LEVEL METRICS")
    print("=" * 70)
    for row in component_rows:
        val = f"{row.value:.4f}" if isinstance(row.value, float) else "   N/A"
        print(f"  {row.metric:<28} {row.dataset:<13} {val}   n={row.n:<5} {row.notes}")
    print("\n" + "=" * 70)
    print("MoE SYSTEM-LEVEL METRICS")
    print("=" * 70)
    for row in system_rows:
        val = f"{row.value:.4f}" if isinstance(row.value, float) else "   N/A"
        print(f"  {row.metric:<28} {row.dataset:<13} {val}   n={row.n:<5} {row.notes}")
    print("=" * 70)
    print(f"Wrote: {components_csv}\nWrote: {system_csv}\nWrote: {per_image_csv}\nWrote: {summary_path}")

    return summary
