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
from src.components.component10_biogpt import BioGPTConfig, build_report_generator
from src.components.component10_report import generate_baseline_report
from src.components.component1_dann import Component1DANNModel, DANNHead
from src.components.component1_encoder import (
    Component1EncoderConfig,
    build_component1_encoder,
    load_trainable_state_dict,
)
from src.components.component2_txv import Component2SoftDomainContext
from src.components.component3_routing import Component3RoutingGate, RoutingGateConfig
from src.components.component4_lung import Component4MedSAM
from src.components.component5_experts import ExpertBank, ExpertDecoderConfig, EXPERT_NAMES
from src.components.component6_fusion import Component6ExpertFusion, FusionConfig
from src.components.component7_boundary import BoundaryScoreBreakdown, score_boundary_quality
from src.components.component7_fp_auditor import estimate_fp_probability
from src.components.component7_refine import BaselineRefineConfig, refine_mask
from src.components.component7_verification import (
    Component7BoundaryCritic,
    Component7RepromptRefiner,
    RepromptRefinerConfig,
)
from src.components.component8_timika import compute_baseline_timika, compute_moe_timika
from src.components.component9_json_output import generate_structured_json, save_structured_json
from src.core.device import describe_device, pick_device
from src.core.seed import seed_everything
from src.core.types import BaselineInferenceBundle
from src.utils.checkpoints import compute_file_sha256
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


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_models(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[Component1DANNModel, Component2SoftDomainContext, Component4MedSAM]:
    component1_cfg = config.get("component1", {})
    component2_cfg = config.get("component2", {})
    component4_cfg = config.get("component4", {})
    encoder = build_component1_encoder(
        Component1EncoderConfig(
            backend=str(component1_cfg.get("backend", "auto")),
            checkpoint_path=component1_cfg.get("checkpoint_path"),
            freeze_backbone=True,
        )
    )
    component1_model = Component1DANNModel(encoder=encoder, head=DANNHead()).to(device)
    component1_model.loaded_adapter_path = None  # type: ignore[attr-defined]
    adapter_path = component1_cfg.get("adapter_path")
    if adapter_path:
        if encoder.active_backend != "segment_anything":
            # Adapters are trained against the real MedSAM ViT-B backbone;
            # their parameter names won't match the mock fallback's module tree.
            print(
                f"WARNING: adapter_path set but Component 1 backend is "
                f"{encoder.active_backend!r}; LoRA+DANN adapters NOT loaded."
            )
        else:
            try:
                loaded_adapter = load_trainable_state_dict(component1_model, adapter_path)
                component1_model.loaded_adapter_path = str(loaded_adapter)  # type: ignore[attr-defined]
                print(f"Component 1: loaded LoRA+DANN adapters from {loaded_adapter}")
            except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
                print(
                    "WARNING: failed to load Component 1 adapters from "
                    f"{adapter_path!r}; continuing with frozen backbone only. Error: {exc}"
                )
    component2_model = Component2SoftDomainContext(
        backend=str(component2_cfg.get("backend", "auto")),
        weights=str(component2_cfg.get("weights", "densenet121-res224-all")),
    ).to(device)
    component2_model.loaded_routing_head_path = None  # type: ignore[attr-defined]
    routing_head_path = component2_cfg.get("routing_head_path")
    if routing_head_path:
        try:
            loaded_routing_head = component2_model.load_trained_routing_head(routing_head_path)
            component2_model.loaded_routing_head_path = str(loaded_routing_head)  # type: ignore[attr-defined]
            print(f"Component 2: loaded routing head from {loaded_routing_head}")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(
                "WARNING: failed to load Component 2 routing head from "
                f"{routing_head_path!r}; continuing with default routing head. Error: {exc}"
            )
    component4_model = Component4MedSAM(
        backend=str(component4_cfg.get("backend", "auto")),
        checkpoint_path=component4_cfg.get("checkpoint_path"),
        model_type=str(component4_cfg.get("model_type", "vit_b")),
        mask_threshold=float(component4_cfg.get("mask_threshold", 0.5)),
    ).to(device)
    decoder_ckpt = component4_cfg.get("decoder_checkpoint_path")
    component4_model.loaded_decoder_checkpoint = None  # type: ignore[attr-defined]
    if decoder_ckpt:
        if component4_model.active_backend != "medsam":
            print(
                f"WARNING: decoder_checkpoint_path set but Component 4 backend is "
                f"{component4_model.active_backend!r}; fine-tuned decoder NOT loaded."
            )
        else:
            try:
                loaded = component4_model.load_trained_decoder(decoder_ckpt)
                component4_model.loaded_decoder_checkpoint = str(loaded)  # type: ignore[attr-defined]
                print(f"Component 4: loaded fine-tuned decoder from {loaded}")
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(
                    "WARNING: failed to load Component 4 fine-tuned decoder from "
                    f"{decoder_ckpt!r}; continuing with base decoder. Error: {exc}"
                )
    component1_model.eval()
    component2_model.eval()
    component4_model.eval()
    return component1_model, component2_model, component4_model


def build_moe_models(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[Component3RoutingGate, ExpertBank, Component6ExpertFusion, Component7BoundaryCritic | None] | None:
    """Build MoE components (C3, C5, C6) if configured.

    Returns ``None`` if the MoE path is not enabled in the config, allowing
    the pipeline to fall back to the baseline lesion proposer.
    """
    moe_cfg = config.get("moe", {})
    if not moe_cfg.get("enabled", False):
        return None

    num_experts = int(moe_cfg.get("num_experts", 4))

    # Component 3 — Routing Gate
    routing_gate = Component3RoutingGate(
        RoutingGateConfig(
            num_experts=num_experts,
            use_domain_ctx=bool(moe_cfg.get("use_domain_ctx", False)),
            domain_ctx_dim=int(moe_cfg.get("domain_ctx_dim", 256)),
            temperature=float(moe_cfg.get("routing_temperature", 1.0)),
            top_k=moe_cfg.get("routing_top_k"),
        )
    ).to(device)

    # Component 5 — Expert Bank
    expert_bank = ExpertBank(
        ExpertDecoderConfig(
            num_experts=num_experts,
            dropout=float(moe_cfg.get("expert_dropout", 0.1)),
        )
    ).to(device)

    # Component 6 — Fusion
    fusion = Component6ExpertFusion(
        FusionConfig(
            num_experts=num_experts,
            fusion_mode=str(moe_cfg.get("fusion_mode", "weighted_logit")),
            learnable_bias=bool(moe_cfg.get("learnable_fusion_bias", False)),
        )
    ).to(device)

    # Load checkpoints if specified
    moe_ckpt = moe_cfg.get("checkpoint_path")
    if moe_ckpt and Path(moe_ckpt).is_file():
        state = torch.load(moe_ckpt, map_location=device, weights_only=False)
        if "routing_gate" in state:
            routing_gate.load_state_dict(state["routing_gate"], strict=False)
        if "expert_bank" in state:
            expert_bank.load_state_dict(state["expert_bank"])
        if "fusion" in state:
            fusion.load_state_dict(state["fusion"])
        print(f"MoE: loaded checkpoint from {moe_ckpt}")

    component7_cfg = config.get("component7_moe", {})
    boundary_critic: Component7BoundaryCritic | None = None
    critic_ckpt = component7_cfg.get("boundary_critic_checkpoint")
    if critic_ckpt:
        critic_path = Path(critic_ckpt)
        if critic_path.is_file():
            boundary_critic = Component7BoundaryCritic().to(device)
            boundary_critic.load_state_dict(
                torch.load(critic_path, map_location=device, weights_only=False)
            )
            boundary_critic.eval()
            print(f"Component 7: loaded boundary critic from {critic_path}")
        else:
            print(f"WARNING: boundary_critic_checkpoint not found at {critic_path}; using heuristic boundary scorer.")

    routing_gate.eval()
    expert_bank.eval()
    fusion.eval()
    return routing_gate, expert_bank, fusion, boundary_critic


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_single_image_inference(
    *,
    image_path: str | Path,
    dataset: str,
    outdir: str | Path,
    config_path: str | Path = "configs/baseline.yaml",
    view: str | None = None,
    pixel_spacing_cm: float | None = None,
    seed: int = 1337,
    component4_decoder_ckpt: str | Path | None = None,
    component1_adapter_path: str | Path | None = None,
    prebuilt_models: tuple | None = None,
    prebuilt_moe_models: tuple | None = None,
) -> BaselineInferenceBundle:
    config = load_baseline_config(config_path)
    if component4_decoder_ckpt is not None:
        config.setdefault("component4", {})["decoder_checkpoint_path"] = str(component4_decoder_ckpt)
    if component1_adapter_path is not None:
        config.setdefault("component1", {})["adapter_path"] = str(component1_adapter_path)
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

    if prebuilt_models is not None:
        component1_model, component2_model, component4_model = prebuilt_models
    else:
        component1_model, component2_model, component4_model = build_models(config, device)

    # Try to build MoE models — returns None if not configured
    if prebuilt_moe_models is not None:
        moe_models = prebuilt_moe_models
    else:
        moe_models = build_moe_models(config, device)
    use_moe = moe_models is not None

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

        # --- MoE fields (populated only in MoE mode) ---
        routing_weights_out = None
        expert_masks_out = None
        mask_fused_out = None
        mask_variance_out = None
        cavity_mask_out = None
        expert_routing_dict = None

        if use_moe:
            # ===== MoE PATH: C3 → C5 → C6 → C7(upgraded) → C8(upgraded) =====
            routing_gate, expert_bank, fusion, boundary_critic = moe_models

            # C3: Routing gate
            routing_weights = routing_gate(
                img_emb,
                txv_output.domain_ctx if routing_gate.use_domain_ctx else None,
            )  # [B, K]

            # C5: All experts
            expert_logits = expert_bank(img_emb)  # list of K × [B, 1, 256, 256]

            # C6: Fusion
            fusion_output = fusion(expert_logits, routing_weights)
            mask_fused_256 = fusion_output.mask_fused_256
            mask_variance = fusion_output.mask_variance

            # Extract cavity mask from Expert 2 for C8
            cavity_expert_idx = EXPERT_NAMES.index("cavity")
            cavity_mask_256 = fusion_output.expert_masks_256[cavity_expert_idx]  # [B, 1, 256, 256]

            # C7 upgraded: keep heuristic diagnostics, but use the trained
            # critic score when a boundary critic checkpoint is available.
            coarse_mask_256 = (mask_fused_256[0] > 0.5).float()
            lung_mask_256 = lung_output.lung_mask_256[0]
            heuristic_boundary = score_boundary_quality(coarse_mask_256, lung_mask_256, x1024)
            boundary_score = heuristic_boundary.boundary_score
            if boundary_critic is not None:
                critic_crop = Component7BoundaryCritic.prepare_crop(
                    x1024,
                    coarse_mask_256,
                    lung_mask_256,
                ).to(device)
                boundary_score = float(boundary_critic(critic_crop).item())
            boundary = BoundaryScoreBreakdown(
                boundary_score=boundary_score,
                lesion_fraction=heuristic_boundary.lesion_fraction,
                spill_fraction=heuristic_boundary.spill_fraction,
                n_components=heuristic_boundary.n_components,
                compactness=heuristic_boundary.compactness,
                edge_alignment=heuristic_boundary.edge_alignment,
            )

            # C7 upgraded: reprompt refinement using Expert 3
            reprompt_refiner = Component7RepromptRefiner(
                RepromptRefinerConfig(
                    boundary_threshold=float(
                        config.get("component7_moe", {}).get("boundary_threshold", 0.7)
                    ),
                    variance_threshold=float(
                        config.get("component7_moe", {}).get("variance_threshold", 0.3)
                    ),
                    num_prompt_points=int(
                        config.get("component7_moe", {}).get("num_prompt_points", 5)
                    ),
                )
            )
            expert3 = expert_bank.expert_by_name("fibrosis")
            refined_mask_256_moe = reprompt_refiner(
                image_emb=img_emb,
                mask_fused_256=mask_fused_256,
                mask_variance=mask_variance,
                boundary_score=boundary.boundary_score,
                expert3_decoder=expert3,
                lung_mask_256=lung_output.lung_mask_256,
            )

            # FP audit (same as baseline — reuses TXV features)
            fp_audit = estimate_fp_probability(
                coarse_mask_256,
                lung_mask_256,
                txv_output.pathology_logits[0],
                class_names=txv_output.class_names,
            )

            # Apply FP suppression on the MoE-refined mask
            refined_mask_256 = refine_mask(
                refined_mask_256_moe[0],
                lung_output.lung_mask_256[0],
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

            # C8 upgraded: real cavity detection from Expert 2
            timika = compute_moe_timika(
                refined_mask_1024,
                lung_output.lung_mask_1024[0],
                cavity_mask_256[0],
            )

            # Build expert routing dict for JSON output
            rw = routing_weights[0].detach().cpu().tolist()
            expert_routing_dict = {
                name: rw[i]
                for i, name in enumerate(EXPERT_NAMES[: expert_bank.num_experts])
            }

            # Store MoE outputs for the bundle
            routing_weights_out = routing_weights.detach().cpu()
            expert_masks_out = [m.detach().cpu() for m in fusion_output.expert_masks_256]
            mask_fused_out = mask_fused_256.detach().cpu()
            mask_variance_out = mask_variance.detach().cpu()
            cavity_mask_out = cavity_mask_256.detach().cpu()

            # Coarse mask for the bundle = fused mask (before refinement)
            coarse_mask_for_bundle = mask_fused_256.detach().cpu()
            selected_classes_for_summary = list(EXPERT_NAMES[: expert_bank.num_experts])
            pipeline_mode = "moe"

        else:
            # ===== BASELINE PATH (unchanged) =====
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

            coarse_mask_for_bundle = lesion_proposal.lesion_mask_coarse_256.detach().cpu()
            selected_classes_for_summary = lesion_proposal.selected_classes[0]
            pipeline_mode = "baseline"

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
            expert_routing=expert_routing_dict,
            boundary_quality_score=boundary.boundary_score,
            fp_probability=fp_audit.fp_probability,
            alp=timika.ALP,
            cavity_flag=timika.cavity_flag,
            timika_score=timika.timika_score,
            severity=timika.severity,
            pathology_flags=pathology_flags,
            cavitation_confidence=timika.cavitation_confidence,
        )
        report_cfg = config.get("component10", {}) or {}
        report_backend = str(report_cfg.get("backend", "template"))
        if report_backend == "biogpt":
            biogpt_cfg = BioGPTConfig(
                model_name=str(report_cfg.get("model_name", "microsoft/BioGPT-Large")),
                use_mock=bool(report_cfg.get("use_mock", False)),
                fall_back_on_unfaithful=bool(report_cfg.get("fall_back_on_unfaithful", True)),
                fall_back_on_load_error=bool(report_cfg.get("fall_back_on_load_error", True)),
            )
            report_gen = build_report_generator(backend="biogpt", biogpt_config=biogpt_cfg)
            report_text = report_gen.generate(evidence_json)
        else:
            report_text = generate_baseline_report(evidence_json)

    bundle = BaselineInferenceBundle(
        harmonised=harmonised,
        img_emb=img_emb.detach().cpu(),
        dom_logits=dom_logits.detach().cpu(),
        domain_ctx=txv_output.domain_ctx.detach().cpu(),
        pathology_logits=txv_output.pathology_logits.detach().cpu(),
        lung_mask_256=lung_output.lung_mask_256.detach().cpu(),
        lung_mask_1024=lung_output.lung_mask_1024.detach().cpu(),
        lesion_mask_coarse_256=coarse_mask_for_bundle,
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
        # MoE fields
        pipeline_mode=pipeline_mode,
        routing_weights=routing_weights_out,
        expert_masks_256=expert_masks_out,
        mask_fused_256=mask_fused_out,
        mask_variance_256=mask_variance_out,
        cavity_mask_256=cavity_mask_out,
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

    # --- Checkpoint provenance: compute SHA-256 for every loaded checkpoint ---
    _provenance_candidates: dict[str, str | None] = {
        "component1_backbone": config.get("component1", {}).get("checkpoint_path"),
        "component1_adapter": getattr(component1_model, "loaded_adapter_path", None),
        "component2_routing_head": getattr(component2_model, "loaded_routing_head_path", None),
        "component4_backbone": config.get("component4", {}).get("checkpoint_path"),
        "component4_decoder": getattr(component4_model, "loaded_decoder_checkpoint", None),
    }
    if use_moe:
        _provenance_candidates["moe_checkpoint"] = config.get("moe", {}).get("checkpoint_path")
        _provenance_candidates["boundary_critic"] = config.get("component7_moe", {}).get("boundary_critic_checkpoint")

    checkpoint_provenance: dict[str, str] = {}
    for ckpt_name, ckpt_path in _provenance_candidates.items():
        if ckpt_path is None:
            continue
        try:
            checkpoint_provenance[ckpt_name] = compute_file_sha256(ckpt_path)
        except FileNotFoundError:
            checkpoint_provenance[ckpt_name] = "file-not-found"

    run_summary: dict[str, Any] = {
        "image": str(image_path),
        "dataset": dataset,
        "device": describe_device(device),
        "pipeline_mode": pipeline_mode,
        "component1_backend": component1_model.encoder.active_backend,
        "component2_backend": component2_model.active_backend,
        "component2_routing_head_path": getattr(component2_model, "loaded_routing_head_path", None),
        "component4_backend": component4_model.active_backend,
        "component4_decoder_ckpt": getattr(component4_model, "loaded_decoder_checkpoint", None),
        "component1_adapter_path": getattr(component1_model, "loaded_adapter_path", None),
        "boundary_score": boundary.boundary_score,
        "fp_probability": fp_audit.fp_probability,
        "checkpoint_provenance": checkpoint_provenance,
    }
    if use_moe:
        run_summary["expert_routing"] = expert_routing_dict
        run_summary["moe_checkpoint"] = config.get("moe", {}).get("checkpoint_path")
        run_summary["boundary_critic_checkpoint"] = config.get("component7_moe", {}).get("boundary_critic_checkpoint")
    else:
        run_summary["selected_classes"] = selected_classes_for_summary

    (outdir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )

    # Save MoE-specific masks if in MoE mode
    if use_moe and bundle.mask_fused_256 is not None:
        save_mask_png(bundle.mask_fused_256[0, 0], outdir / "mask_fused_256.png")
        if bundle.cavity_mask_256 is not None:
            save_mask_png(bundle.cavity_mask_256[0, 0], outdir / "cavity_mask_256.png")
        if bundle.expert_masks_256 is not None:
            for i, emask in enumerate(bundle.expert_masks_256):
                name = EXPERT_NAMES[i] if i < len(EXPERT_NAMES) else f"expert{i}"
                save_mask_png(emask[0, 0], outdir / f"expert_{name}_256.png")

    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TB CXR pipeline on a single image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--view", default=None)
    parser.add_argument("--pixel-spacing-cm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--component4-decoder-ckpt",
        default=None,
        help="Path to a fine-tuned Component 4 mask decoder checkpoint (overrides config).",
    )
    parser.add_argument(
        "--component1-adapter",
        default=None,
        help="Path to Component 1 LoRA+DANN adapters (overrides config).",
    )
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
        component4_decoder_ckpt=args.component4_decoder_ckpt,
        component1_adapter_path=args.component1_adapter,
    )
    run_summary = json.loads((Path(args.outdir) / "run_summary.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "report_path": str(Path(args.outdir) / "report.txt"),
                "json_path": str(Path(args.outdir) / "evidence.json"),
                "overlay_path": str(Path(args.outdir) / "overlay.png"),
                "pipeline_mode": bundle.pipeline_mode,
                "component1_backend": run_summary.get("component1_backend"),
                "component1_adapter_path": run_summary.get("component1_adapter_path"),
                "component2_backend": run_summary.get("component2_backend"),
                "component4_backend": run_summary.get("component4_backend"),
                "component4_decoder_ckpt": run_summary.get("component4_decoder_ckpt"),
                "timika_score": bundle.timika_score,
                "severity": bundle.severity,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
