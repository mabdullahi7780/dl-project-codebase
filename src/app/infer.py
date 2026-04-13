from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.components.baseline_lesion_proposer import (
    BaselineLesionProposer,
    BaselineLesionProposerConfig,
)
from src.components.component0_qc import harmonise_sample
from src.components.component10_report import generate_baseline_report
from src.components.component1_dann import Component1DANNModel, DANNHead
from src.components.component1_encoder import Component1EncoderConfig, build_component1_encoder
from src.components.component2_txv import Component2SoftDomainContext
from src.components.component4_lung import Component4MedSAM
from src.components.component7_boundary import score_boundary_quality
from src.components.component7_fp_auditor import estimate_fp_probability
from src.components.component7_refine import BaselineRefineConfig, refine_mask
from src.components.component8_timika import compute_baseline_timika
from src.components.component9_json_output import generate_structured_json, save_structured_json
from src.core.device import describe_device, pick_device
from src.core.seed import seed_everything
from src.core.types import BaselineInferenceBundle
from src.utils.morphology import connected_component_stats
from src.utils.visualization import save_mask_png, save_overlay_png


def load_baseline_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_grayscale_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def pathology_flags_from_logits(
    pathology_logits: torch.Tensor,
    class_names: tuple[str, ...],
    *,
    threshold: float = 0.5,
) -> dict[str, bool]:
    probs = torch.sigmoid(pathology_logits.detach().cpu()).squeeze(0)
    return {
        name: bool(float(probs[index].item()) > threshold)
        for index, name in enumerate(class_names)
    }


def compute_lesion_area_cm2(mask_1024: torch.Tensor, pixel_spacing_cm: float | None) -> float:
    if pixel_spacing_cm is None:
        return 0.0
    lesion_pixels = int((mask_1024 > 0.5).sum().item())
    return float(lesion_pixels * (pixel_spacing_cm ** 2))


def count_connected_regions(mask_256: torch.Tensor) -> int:
    array = (mask_256.detach().cpu().numpy() > 0.5).astype(np.uint8)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return len(connected_component_stats(array))


def build_models(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[Component1DANNModel, Component2SoftDomainContext, Component4MedSAM]:
    component1_cfg = config.get("component1", {})
    encoder = build_component1_encoder(
        Component1EncoderConfig(
            backend=str(component1_cfg.get("backend", "auto")),
            checkpoint_path=component1_cfg.get("checkpoint_path"),
            freeze_backbone=True,
        )
    )
    component1_model = Component1DANNModel(encoder=encoder, head=DANNHead()).to(device)
    component2_model = Component2SoftDomainContext(
        backend=str(config.get("component2", {}).get("backend", "auto")),
        weights=str(config.get("component2", {}).get("weights", "densenet121-res224-all")),
    ).to(device)
    component4_model = Component4MedSAM(
        backend=str(config.get("component4", {}).get("backend", "auto"))
    ).to(device)
    component1_model.eval()
    component2_model.eval()
    component4_model.eval()
    return component1_model, component2_model, component4_model


def run_single_image_inference(
    *,
    image_path: str | Path,
    dataset: str,
    outdir: str | Path,
    config_path: str | Path = "configs/baseline.yaml",
    view: str | None = None,
    pixel_spacing_cm: float | None = None,
    seed: int = 1337,
) -> BaselineInferenceBundle:
    config = load_baseline_config(config_path)
    seed_everything(seed)
    device = pick_device(config.get("runtime", {}).get("device"))

    image_path = Path(image_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample = {
        "image": load_grayscale_image(image_path),
        "image_id": image_path.stem,
        "dataset_id": dataset,
        "view": view,
        "pixel_spacing_cm": pixel_spacing_cm,
        "path": str(image_path),
    }
    harmonised = harmonise_sample(
        sample,
        apply_clahe=config.get("component0", {}).get("apply_clahe"),
    )

    component1_model, component2_model, component4_model = build_models(config, device)

    proposer = BaselineLesionProposer(
        BaselineLesionProposerConfig(
            suspicious_class_threshold=float(config.get("baseline_lesion_proposer", {}).get("suspicious_class_threshold", 0.25)),
            fixed_binary_threshold=config.get("baseline_lesion_proposer", {}).get("fixed_binary_threshold"),
            min_region_px=int(config.get("baseline_lesion_proposer", {}).get("min_region_px", 48)),
            opening_iters=int(config.get("baseline_lesion_proposer", {}).get("opening_iters", 1)),
            closing_iters=int(config.get("baseline_lesion_proposer", {}).get("closing_iters", 1)),
            fallback_blend=float(config.get("baseline_lesion_proposer", {}).get("fallback_blend", 0.35)),
        )
    )
    refine_cfg = BaselineRefineConfig(
        min_area_px=int(config.get("component7_refine", {}).get("min_area_px", 48)),
        opening_iters=int(config.get("component7_refine", {}).get("opening_iters", 1)),
        closing_iters=int(config.get("component7_refine", {}).get("closing_iters", 1)),
        weak_boundary_threshold=float(config.get("component7_refine", {}).get("weak_boundary_threshold", 0.45)),
        suppress_fp_threshold=float(config.get("component7_refine", {}).get("suppress_fp_threshold", 0.85)),
        caution_fp_threshold=float(config.get("component7_refine", {}).get("caution_fp_threshold", 0.65)),
    )

    with torch.no_grad():
        x3 = harmonised.x_3ch.unsqueeze(0).to(device)
        x224 = harmonised.x_224.unsqueeze(0).to(device)
        x1024 = harmonised.x_1024.to(device)

        img_emb, dom_logits = component1_model(x3, lambda_=0.0)
        txv_output = component2_model.forward_features(x224)
        lung_output = component4_model.predict_masks(x3)

        lesion_proposal = proposer.propose(
            x_224=x224,
            features_7x7=txv_output.features_7x7,
            pathology_logits=txv_output.pathology_logits,
            lung_mask_256=lung_output.lung_mask_256,
            classifier_weight=txv_output.classifier_weight,
        )

        coarse_mask_256 = lesion_proposal.lesion_mask_coarse_256[0]
        lung_mask_256 = lung_output.lung_mask_256[0]
        boundary = score_boundary_quality(coarse_mask_256, lung_mask_256, x1024)
        fp_audit = estimate_fp_probability(
            coarse_mask_256,
            lung_mask_256,
            txv_output.pathology_logits[0],
            class_names=txv_output.class_names,
        )
        refined_mask_256 = refine_mask(
            coarse_mask_256,
            lung_mask_256,
            x1024,
            boundary.boundary_score,
            fp_audit.fp_probability,
            refine_cfg,
        )
        refined_mask_1024 = F.interpolate(
            refined_mask_256.unsqueeze(0),
            size=(1024, 1024),
            mode="nearest",
        ).squeeze(0)

        timika = compute_baseline_timika(refined_mask_1024, lung_output.lung_mask_1024[0])
        pathology_flags = pathology_flags_from_logits(
            txv_output.pathology_logits,
            txv_output.class_names,
            threshold=float(config.get("component2", {}).get("pathology_flag_threshold", 0.5)),
        )

        evidence_json = generate_structured_json(
            patient_id=sample["image_id"],
            modality="CXR-PA" if view == "PA" else "CXR",
            scanner_domain=str(harmonised.meta.get("scanner_domain", dataset)),
            n_distinct_lesions=count_connected_regions(refined_mask_256),
            lesion_area_cm2=compute_lesion_area_cm2(refined_mask_1024, pixel_spacing_cm),
            expert_routing=None,
            boundary_quality_score=boundary.boundary_score,
            fp_probability=fp_audit.fp_probability,
            alp=timika.ALP,
            cavity_flag=timika.cavity_flag,
            timika_score=timika.timika_score,
            severity=timika.severity,
            pathology_flags=pathology_flags,
            cavitation_confidence=timika.cavitation_confidence,
        )
        report_text = generate_baseline_report(evidence_json)

    bundle = BaselineInferenceBundle(
        harmonised=harmonised,
        img_emb=img_emb.detach().cpu(),
        dom_logits=dom_logits.detach().cpu(),
        domain_ctx=txv_output.domain_ctx.detach().cpu(),
        pathology_logits=txv_output.pathology_logits.detach().cpu(),
        lung_mask_256=lung_output.lung_mask_256.detach().cpu(),
        lung_mask_1024=lung_output.lung_mask_1024.detach().cpu(),
        lesion_mask_coarse_256=lesion_proposal.lesion_mask_coarse_256.detach().cpu(),
        lesion_mask_refined_256=refined_mask_256.unsqueeze(0).detach().cpu(),
        lesion_mask_refined_1024=refined_mask_1024.unsqueeze(0).detach().cpu(),
        boundary_score=boundary.boundary_score,
        fp_prob=fp_audit.fp_probability,
        alp=timika.ALP,
        cavity_flag=timika.cavity_flag,
        timika_score=timika.timika_score,
        severity=timika.severity,
        evidence_json=evidence_json,
        report_text=report_text,
    )

    save_mask_png(bundle.lung_mask_1024[0], outdir / "lung_mask_1024.png")
    save_mask_png(bundle.lesion_mask_coarse_256[0], outdir / "lesion_mask_coarse_256.png")
    save_mask_png(bundle.lesion_mask_refined_1024[0], outdir / "lesion_mask_refined_1024.png")
    save_overlay_png(
        harmonised.x_1024.squeeze(0),
        bundle.lesion_mask_refined_1024[0],
        bundle.lung_mask_1024[0],
        outdir / "overlay.png",
    )
    save_structured_json(evidence_json, str(outdir / "evidence.json"))
    (outdir / "report.txt").write_text(report_text, encoding="utf-8")
    (outdir / "run_summary.json").write_text(
        json.dumps(
            {
                "image": str(image_path),
                "dataset": dataset,
                "device": describe_device(device),
                "selected_classes": lesion_proposal.selected_classes[0],
                "boundary_score": boundary.boundary_score,
                "fp_probability": fp_audit.fp_probability,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline TB CXR pipeline on a single image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--view", default=None)
    parser.add_argument("--pixel-spacing-cm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = run_single_image_inference(
        image_path=args.image,
        dataset=args.dataset,
        outdir=args.outdir,
        config_path=args.config,
        view=args.view,
        pixel_spacing_cm=args.pixel_spacing_cm,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "report_path": str(Path(args.outdir) / "report.txt"),
                "json_path": str(Path(args.outdir) / "evidence.json"),
                "overlay_path": str(Path(args.outdir) / "overlay.png"),
                "timika_score": bundle.timika_score,
                "severity": bundle.severity,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
