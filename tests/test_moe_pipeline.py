"""End-to-end smoke test for the MoE inference path.

Wires together C3 (routing) → C5 (experts) → C6 (fusion) → upgraded C8
(real cavity from Expert 2) and verifies the shapes line up.  Does NOT
exercise infer.py directly because that requires loading C1/C2/C4 which
in turn need on-disk checkpoints; this test focuses on the new MoE
pieces.
"""

from __future__ import annotations

import torch

from src.components.component3_routing import Component3RoutingGate, RoutingGateConfig
from src.components.component5_experts import (
    EXPERT_NAMES,
    ExpertBank,
    ExpertDecoderConfig,
)
from src.components.component6_fusion import Component6ExpertFusion, FusionConfig
from src.components.component7_verification import (
    Component7BoundaryCritic,
    Component7RepromptRefiner,
    RepromptRefinerConfig,
)
from src.components.component8_timika import compute_moe_timika


def test_moe_pipeline_runs_end_to_end() -> None:
    img_emb = torch.randn(1, 256, 64, 64)
    lung_mask_256 = torch.zeros(1, 1, 256, 256)
    lung_mask_256[:, :, 50:200, 50:200] = 1.0  # square lung region
    lung_mask_1024 = torch.zeros(1, 1024, 1024)
    lung_mask_1024[:, 200:800, 200:800] = 1.0

    # C3
    routing_gate = Component3RoutingGate(RoutingGateConfig(num_experts=4))
    weights = routing_gate(img_emb)
    assert weights.shape == (1, 4)

    # C5
    bank = ExpertBank(ExpertDecoderConfig(num_experts=4))
    expert_logits = bank(img_emb)
    assert len(expert_logits) == 4

    # C6
    fusion = Component6ExpertFusion(FusionConfig(num_experts=4))
    fused = fusion(expert_logits, weights)
    assert fused.mask_fused_256.shape == (1, 1, 256, 256)
    assert fused.mask_variance.shape == (1, 1, 256, 256)

    # Cavity mask from Expert 2
    cavity_idx = EXPERT_NAMES.index("cavity")
    cavity_mask = fused.expert_masks_256[cavity_idx]
    assert cavity_mask.shape == (1, 1, 256, 256)

    # C7 reprompt refiner — should run without error
    refiner = Component7RepromptRefiner(RepromptRefinerConfig(boundary_threshold=0.7))
    expert3 = bank.expert_by_name("fibrosis")
    refined = refiner(
        image_emb=img_emb,
        mask_fused_256=fused.mask_fused_256,
        mask_variance=fused.mask_variance,
        boundary_score=0.4,                # below threshold → triggers refinement
        expert3_decoder=expert3,
        lung_mask_256=lung_mask_256,
    )
    assert refined.shape == (1, 1, 256, 256)

    # Upgrade C8 — real cavity flag
    refined_1024 = torch.nn.functional.interpolate(
        refined, size=(1024, 1024), mode="nearest"
    ).squeeze(0)
    timika = compute_moe_timika(refined_1024, lung_mask_1024, cavity_mask[0])
    assert timika.cavitation_confidence == "expert2-radiographic"
    assert timika.cavity_flag in (0, 1)
    assert isinstance(timika.ALP, float)
    assert isinstance(timika.timika_score, float)


def test_moe_pipeline_skips_refinement_when_boundary_strong() -> None:
    img_emb = torch.randn(1, 256, 64, 64)
    bank = ExpertBank()
    fusion = Component6ExpertFusion(FusionConfig(num_experts=4))
    weights = torch.softmax(torch.randn(1, 4), dim=-1)

    fused = fusion(bank(img_emb), weights)

    refiner = Component7RepromptRefiner(RepromptRefinerConfig(boundary_threshold=0.5))
    refined = refiner(
        image_emb=img_emb,
        mask_fused_256=fused.mask_fused_256,
        mask_variance=fused.mask_variance,
        boundary_score=0.9,                # above threshold → returns input
        expert3_decoder=bank.expert_by_name("fibrosis"),
    )
    # When boundary is strong, refiner returns mask_fused unchanged
    assert torch.equal(refined, fused.mask_fused_256)


def test_compute_moe_timika_falls_back_when_no_cavity() -> None:
    refined_1024 = torch.zeros(1, 1024, 1024)
    refined_1024[0, 200:600, 200:600] = 1.0
    lung_1024 = torch.zeros(1, 1024, 1024)
    lung_1024[0, 100:900, 100:900] = 1.0

    result = compute_moe_timika(refined_1024, lung_1024, cavity_mask_256=None)
    assert result.cavitation_confidence == "not-assessed-baseline"
    assert result.cavity_flag == 0


def test_boundary_critic_smoke() -> None:
    """The ResNet18 boundary critic should accept a 3-channel 224x224 crop."""
    critic = Component7BoundaryCritic()
    crop = torch.randn(1, 3, 224, 224)
    score = critic(crop)
    assert score.shape == (1, 1)
    assert (0 <= score).all() and (score <= 1).all()
