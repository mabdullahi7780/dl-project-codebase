"""Tests for Component 5 — Expert Decoders."""

from __future__ import annotations

import pytest
import torch

from src.components.component5_experts import (
    EXPERT_NAMES,
    ExpertBank,
    ExpertDecoderConfig,
    LightweightExpertDecoder,
)


def test_expert_decoder_shape_contract() -> None:
    expert = LightweightExpertDecoder(expert_name="consolidation")
    img_emb = torch.randn(2, 256, 64, 64)
    logits = expert(img_emb)
    assert logits.shape == (2, 1, 256, 256)
    # Logits — not bounded to [0,1]; sigmoid converts to probabilities
    probs = torch.sigmoid(logits)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_expert_decoder_with_prompts() -> None:
    expert = LightweightExpertDecoder(expert_name="cavity")
    img_emb = torch.randn(1, 256, 64, 64)
    prompts = torch.randn(1, 5, 256)              # 5 point prompts
    dense = torch.randn(1, 1, 64, 64)              # dense mask prior
    logits = expert(img_emb, prompts=prompts, dense_prompt=dense)
    assert logits.shape == (1, 1, 256, 256)


def test_expert_decoder_rejects_wrong_input_shape() -> None:
    expert = LightweightExpertDecoder(expert_name="nodule")
    bad = torch.randn(1, 128, 64, 64)  # wrong channel count
    with pytest.raises(ValueError, match="image_emb"):
        expert(bad)


def test_expert_bank_default_size() -> None:
    bank = ExpertBank()
    assert bank.num_experts == 4
    assert bank.expert_names == EXPERT_NAMES


def test_expert_bank_runs_all_experts() -> None:
    bank = ExpertBank(ExpertDecoderConfig(num_experts=4))
    img_emb = torch.randn(2, 256, 64, 64)
    masks = bank(img_emb)
    assert len(masks) == 4
    for m in masks:
        assert m.shape == (2, 1, 256, 256)


def test_expert_bank_runs_subset() -> None:
    bank = ExpertBank()
    img_emb = torch.randn(1, 256, 64, 64)
    masks = bank(img_emb, indices=[0, 2])
    assert len(masks) == 2


def test_expert_bank_lookup_by_name() -> None:
    bank = ExpertBank()
    expert = bank.expert_by_name("fibrosis")
    assert expert.expert_name == "fibrosis"


def test_expert_bank_lookup_unknown_raises() -> None:
    bank = ExpertBank()
    with pytest.raises(KeyError):
        bank.expert_by_name("does_not_exist")
