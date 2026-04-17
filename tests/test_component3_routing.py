"""Tests for Component 3 — Routing Gate."""

from __future__ import annotations

import torch

from src.components.component3_routing import (
    Component3RoutingGate,
    RoutingGateConfig,
    routing_load_balance_loss,
)


def test_routing_gate_shape_and_softmax() -> None:
    gate = Component3RoutingGate(RoutingGateConfig(num_experts=4))
    img_emb = torch.randn(3, 256, 64, 64)
    weights = gate(img_emb)

    assert weights.shape == (3, 4)
    # Softmax → each row sums to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-5)
    # All non-negative
    assert (weights >= 0).all()


def test_routing_gate_topk_sparsity() -> None:
    gate = Component3RoutingGate(RoutingGateConfig(num_experts=4, top_k=2))
    img_emb = torch.randn(2, 256, 64, 64)
    weights = gate(img_emb)

    assert weights.shape == (2, 4)
    # Each row has at most top_k non-zero entries
    nonzero_counts = (weights > 1e-6).sum(dim=-1)
    assert (nonzero_counts <= 2).all()
    # Each row still sums to 1 after re-normalisation
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_routing_gate_temperature_changes_entropy() -> None:
    """Higher temperature → flatter distribution → higher entropy."""
    img_emb = torch.randn(4, 256, 64, 64)

    cold = Component3RoutingGate(RoutingGateConfig(num_experts=4, temperature=0.1))
    hot = Component3RoutingGate(RoutingGateConfig(num_experts=4, temperature=5.0))
    # Use the same gate weights so only temperature varies
    hot.load_state_dict(cold.state_dict())

    cold_w = cold(img_emb)
    hot_w = hot(img_emb)

    cold_entropy = -(cold_w * torch.log(cold_w + 1e-9)).sum(dim=-1).mean()
    hot_entropy = -(hot_w * torch.log(hot_w + 1e-9)).sum(dim=-1).mean()
    assert hot_entropy > cold_entropy


def test_load_balance_loss_min_when_uniform() -> None:
    # Perfectly uniform → loss = K * (1/K) * (1/K) * K = 1
    uniform = torch.ones(8, 4) / 4.0
    loss = routing_load_balance_loss(uniform)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-3)


def test_load_balance_loss_high_when_collapsed() -> None:
    # All routed to expert 0 → loss = K * 1 * (1/K) * K = K
    collapsed = torch.zeros(8, 4)
    collapsed[:, 0] = 1.0
    loss = routing_load_balance_loss(collapsed)
    # K=4: f=[1,0,0,0], p=[1,0,0,0], sum(f*p)=1, K*sum=4
    assert loss.item() > 1.5
