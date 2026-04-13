import pytest
import torch
from src.components.component4_lung import (
    Component4MedSAM,
    bce_dice_loss,
    resolve_component4_backend,
)

def test_component4_tensor_contract() -> None:
    model = Component4MedSAM(backend="mock")
    
    # Simulate batch: B=2, 3-channel, 1024x1024
    x_3ch = torch.rand(2, 3, 1024, 1024)
    
    mask_1024 = model(x_3ch)
    
    assert tuple(mask_1024.shape) == (2, 1, 1024, 1024)
    
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen_params = [name for name, p in model.named_parameters() if not p.requires_grad]
    
    assert all("decoder" in name for name in trainable_params), "Only mask decoder should be trainable"
    assert all("encoder" in name for name in frozen_params), "ViT-B backbone must be totally frozen"
    assert model.active_backend == "mock"


def test_component4_auto_backend_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.components.component4_lung._has_segment_anything", lambda: True)
    monkeypatch.setattr("src.components.component4_lung._checkpoint_available", lambda path: path == "demo.pth")

    assert resolve_component4_backend("auto", "demo.pth") == "medsam"
    assert resolve_component4_backend("auto", None) == "mock"

def test_bce_dice_loss() -> None:
    logits = torch.randn(2, 1, 256, 256)
    targets = torch.zeros(2, 1, 256, 256)
    targets[0, 0, 10:20, 10:20] = 1.0
    
    loss = bce_dice_loss(logits, targets)
    assert loss.ndim == 0
    assert float(loss.item()) > 0.0
