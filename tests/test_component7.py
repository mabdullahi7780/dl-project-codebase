import torch
import pytest
from src.components.component7_verification import Component7aBoundaryCritic, Component7bFPAuditor

def test_component7a():
    model = Component7aBoundaryCritic()
    x = torch.randn(2, 3, 224, 224)
    
    # Check frozen layers
    assert not next(model.features.parameters()).requires_grad
    # Check trainable layers
    assert next(model.fc.parameters()).requires_grad
    
    out = model(x)
    assert out.shape == (2, 1)
    assert torch.all(out >= 0) and torch.all(out <= 1)

def test_component7b():
    model = Component7bFPAuditor(txh_fallback=True)
    x = torch.randn(2, 1, 224, 224)
    
    # Check frozen backbone
    assert not next(model.backbone.parameters()).requires_grad
    # Check trainable head
    assert next(model.fp_head.parameters()).requires_grad
    
    fp_prob, concat_feats = model(x)
    assert concat_feats.shape == (2, 1042)
    assert fp_prob.shape == (2, 1)
    assert torch.all(fp_prob >= 0) and torch.all(fp_prob <= 1)
