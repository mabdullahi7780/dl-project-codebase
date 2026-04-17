"""Tests for Component 6 — Expert Fusion."""

from __future__ import annotations

import pytest
import torch

from src.components.component6_fusion import Component6ExpertFusion, FusionConfig


def _make_expert_logits(batch: int, num: int) -> list[torch.Tensor]:
    return [torch.randn(batch, 1, 256, 256) for _ in range(num)]


def test_fusion_output_shapes() -> None:
    fusion = Component6ExpertFusion(FusionConfig(num_experts=4))
    expert_logits = _make_expert_logits(2, 4)
    weights = torch.softmax(torch.randn(2, 4), dim=-1)

    out = fusion(expert_logits, weights)
    assert out.mask_fused_logits.shape == (2, 1, 256, 256)
    assert out.mask_fused_256.shape == (2, 1, 256, 256)
    assert out.mask_variance.shape == (2, 1, 256, 256)
    assert len(out.expert_masks_256) == 4


def test_fusion_probs_in_unit_range() -> None:
    fusion = Component6ExpertFusion(FusionConfig(num_experts=4))
    expert_logits = _make_expert_logits(1, 4)
    weights = torch.softmax(torch.randn(1, 4), dim=-1)

    out = fusion(expert_logits, weights)
    assert out.mask_fused_256.min() >= 0
    assert out.mask_fused_256.max() <= 1


def test_fusion_variance_zero_when_experts_agree() -> None:
    fusion = Component6ExpertFusion(FusionConfig(num_experts=4))
    # All experts produce the SAME logits → zero variance
    base = torch.randn(1, 1, 256, 256)
    expert_logits = [base.clone() for _ in range(4)]
    weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

    out = fusion(expert_logits, weights)
    assert out.mask_variance.max().item() < 1e-5


def test_fusion_variance_nonzero_when_experts_disagree() -> None:
    fusion = Component6ExpertFusion(FusionConfig(num_experts=4))
    expert_logits = [
        torch.full((1, 1, 256, 256), -10.0),
        torch.full((1, 1, 256, 256), 10.0),
        torch.full((1, 1, 256, 256), -10.0),
        torch.full((1, 1, 256, 256), 10.0),
    ]
    weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

    out = fusion(expert_logits, weights)
    assert out.mask_variance.mean().item() > 0.1


def test_fusion_rejects_weight_shape_mismatch() -> None:
    fusion = Component6ExpertFusion(FusionConfig(num_experts=4))
    expert_logits = _make_expert_logits(2, 4)
    bad_weights = torch.softmax(torch.randn(2, 5), dim=-1)  # 5 weights, 4 experts

    with pytest.raises(ValueError, match="routing_weights shape"):
        fusion(expert_logits, bad_weights)


def test_fusion_modes_both_run() -> None:
    expert_logits = _make_expert_logits(1, 4)
    weights = torch.softmax(torch.randn(1, 4), dim=-1)

    for mode in ("weighted_logit", "weighted_prob"):
        fusion = Component6ExpertFusion(FusionConfig(num_experts=4, fusion_mode=mode))
        out = fusion(expert_logits, weights)
        assert out.mask_fused_256.shape == (1, 1, 256, 256)


def test_fusion_unknown_mode_raises() -> None:
    fusion = Component6ExpertFusion(FusionConfig(num_experts=4, fusion_mode="bogus"))
    with pytest.raises(ValueError, match="Unknown fusion_mode"):
        fusion(_make_expert_logits(1, 4), torch.softmax(torch.randn(1, 4), dim=-1))
