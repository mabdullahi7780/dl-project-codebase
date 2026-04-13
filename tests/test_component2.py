from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.components.component2_txv import Component2SoftDomainContext, supervised_contrastive_loss


def test_component2_tensor_contract() -> None:
    model = Component2SoftDomainContext(backend="mock")
    
    # Simulate TXV-normalised batch: B=2, 1-channel, 224x224
    x_224 = torch.rand(2, 1, 224, 224) * 2048.0 - 1024.0
    
    domain_ctx, pathology_logits = model(x_224)
    
    # 1. Check Output Shapes
    assert tuple(domain_ctx.shape) == (2, 256)
    assert tuple(pathology_logits.shape) == (2, 18)
    
    # 2. Check L2 Normalization constraint on domain_ctx
    norms = torch.norm(domain_ctx, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms)), "domain_ctx must be L2-normalized"

    # 3. Check trainability boundaries
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen_params = [name for name, p in model.named_parameters() if not p.requires_grad]
    
    assert all("domain_routing_head" in name for name in trainable_params), "Only routing head should be trainable"
    assert all("txv_model" in name for name in frozen_params), "Backbone & pathology head must be totally frozen"


def test_supervised_contrastive_loss() -> None:
    features = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],  # same as index 0 (domain 0)
        [0.0, 1.0],  # different (domain 1)
        [0.0, 1.0]   # same as index 2 (domain 1)
    ])
    # L2 normalize just to be safe
    features = F.normalize(features, p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    
    loss = supervised_contrastive_loss(features, labels, temperature=0.1)
    
    assert loss.ndim == 0
    assert float(loss.item()) > 0.0
