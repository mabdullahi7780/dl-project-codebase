"""End-to-end contract test for MoE inference via run_single_image_inference.

Creates a synthetic 1024x1024 image and a minimal dummy moe_best.pt checkpoint
in a temp directory, then runs the full pipeline with moe.enabled: true.

All model backends are forced to "mock" so no real checkpoints are needed.
Runs on CPU only.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _make_dummy_moe_checkpoint(path: Path) -> None:
    """Create a minimal moe_best.pt with the three expected state-dict keys.

    The routing gate, expert bank, and fusion weights produced here are
    random but structurally compatible with the default configs (4 experts,
    256-channel embeddings) so load_state_dict(strict=False) succeeds without
    shape errors at inference time.
    """
    from src.components.component3_routing import Component3RoutingGate, RoutingGateConfig
    from src.components.component5_experts import ExpertBank, ExpertDecoderConfig
    from src.components.component6_fusion import Component6ExpertFusion, FusionConfig

    gate = Component3RoutingGate(RoutingGateConfig(num_experts=4, use_domain_ctx=False))
    bank = ExpertBank(ExpertDecoderConfig(num_experts=4))
    fusion = Component6ExpertFusion(FusionConfig(num_experts=4))

    state = {
        "routing_gate": gate.state_dict(),
        "expert_bank": bank.state_dict(),
        "fusion": fusion.state_dict(),
    }
    torch.save(state, path)


def _write_moe_config(config_path: Path, moe_ckpt_path: Path) -> None:
    """Write a minimal YAML config that enables the MoE path with mock backends."""
    import yaml

    cfg = {
        "runtime": {"device": "cpu"},
        "component0": {},
        "component1": {
            "backend": "mock",
            "checkpoint_path": None,
            "adapter_path": None,
        },
        "component2": {
            "backend": "mock",
            "weights": "densenet121-res224-all",
            "pathology_flag_threshold": 0.5,
        },
        "component4": {
            "backend": "mock",
            "checkpoint_path": None,
            "model_type": "vit_b",
            "mask_threshold": 0.5,
        },
        "moe": {
            "enabled": True,
            "num_experts": 4,
            "checkpoint_path": str(moe_ckpt_path),
            "use_domain_ctx": False,
            "domain_ctx_dim": 256,
            "routing_temperature": 1.0,
            "routing_top_k": None,
            "expert_dropout": 0.0,
            "fusion_mode": "weighted_logit",
            "learnable_fusion_bias": False,
        },
        "component7_moe": {
            "boundary_threshold": 0.7,
            "variance_threshold": 0.3,
            "num_prompt_points": 5,
            "boundary_critic_checkpoint": None,
        },
        "component7_refine": {
            "min_area_px": 48,
            "opening_iters": 1,
            "closing_iters": 1,
            "weak_boundary_threshold": 0.45,
            "suppress_fp_threshold": 0.85,
            "caution_fp_threshold": 0.65,
        },
        "baseline_lesion_proposer": {
            "suspicious_class_threshold": 0.25,
            "fixed_binary_threshold": None,
            "min_region_px": 48,
            "opening_iters": 1,
            "closing_iters": 1,
            "fallback_blend": 0.35,
        },
        "component10": {
            "backend": "template",
        },
    }
    config_path.write_text(__import__("yaml").dump(cfg), encoding="utf-8")


def test_moe_inference_contract(tmp_path: Path) -> None:
    """Run the full pipeline in MoE mode and assert all contract fields."""
    # --- synthesise a 1024x1024 grayscale image ---
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, size=(1024, 1024), dtype=np.uint8)
    image_path = tmp_path / "demo_1024.png"
    Image.fromarray(pixels, mode="L").save(image_path)

    # --- write dummy MoE checkpoint ---
    moe_ckpt = tmp_path / "moe_best.pt"
    _make_dummy_moe_checkpoint(moe_ckpt)

    # --- write config ---
    config_path = tmp_path / "moe_test.yaml"
    _write_moe_config(config_path, moe_ckpt)

    outdir = tmp_path / "outputs"

    # --- run inference ---
    from src.app.infer import run_single_image_inference

    bundle = run_single_image_inference(
        image_path=image_path,
        dataset="montgomery",
        outdir=outdir,
        config_path=config_path,
        view="PA",
        seed=0,
    )

    # --- pipeline_mode must be "moe" ---
    assert bundle.pipeline_mode == "moe", (
        f"Expected pipeline_mode=='moe', got {bundle.pipeline_mode!r}"
    )

    # --- routing_weights: non-empty tensor, shape [1, 4] ---
    assert bundle.routing_weights is not None, "routing_weights must not be None in MoE mode"
    assert bundle.routing_weights.numel() > 0, "routing_weights must be non-empty"
    assert bundle.routing_weights.shape == (1, 4), (
        f"Expected routing_weights shape (1, 4), got {tuple(bundle.routing_weights.shape)}"
    )

    # --- expert_masks_256: exactly 4 tensors, each [1, 1, 256, 256] ---
    assert bundle.expert_masks_256 is not None, "expert_masks_256 must not be None in MoE mode"
    assert len(bundle.expert_masks_256) == 4, (
        f"Expected 4 expert masks, got {len(bundle.expert_masks_256)}"
    )
    for i, emask in enumerate(bundle.expert_masks_256):
        assert emask.shape == (1, 1, 256, 256), (
            f"expert_masks_256[{i}] expected shape (1,1,256,256), got {tuple(emask.shape)}"
        )

    # --- mask_fused_256: present ---
    assert bundle.mask_fused_256 is not None, "mask_fused_256 must be set in MoE mode"
    assert bundle.mask_fused_256.shape == (1, 1, 256, 256), (
        f"mask_fused_256 expected (1,1,256,256), got {tuple(bundle.mask_fused_256.shape)}"
    )

    # --- cavitation_confidence from Expert 2 ---
    assert bundle.evidence_json is not None
    scoring = bundle.evidence_json.get("scoring", {})
    assert scoring.get("cavitation_confidence") == "expert2-radiographic", (
        f"Expected cavitation_confidence='expert2-radiographic', "
        f"got {scoring.get('cavitation_confidence')!r}"
    )

    # --- standard output artifacts exist ---
    assert (outdir / "evidence.json").exists()
    assert (outdir / "report.txt").exists()
    assert (outdir / "overlay.png").exists()
    assert (outdir / "run_summary.json").exists()

    # --- run_summary confirms MoE mode ---
    summary = json.loads((outdir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary.get("pipeline_mode") == "moe"
    assert "expert_routing" in summary
