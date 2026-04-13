from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class LungMaskOutput:
    lung_mask_256: torch.Tensor
    lung_mask_1024: torch.Tensor
    lung_logits_256: torch.Tensor

class MockMedSAMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 256, 64, 64), device=x.device)

class MockMedSAMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1)
    def forward(self, image_embeddings: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        mask = self.conv(image_embeddings)
        return F.interpolate(mask, size=(256, 256), mode="bilinear", align_corners=False)

class Component4MedSAM(nn.Module):
    def __init__(self, backend: str = "mock"):
        super().__init__()
        self.backend = backend
        if backend == "auto":
            self.backend = "mock"
            
        if self.backend == "mock":
            self.encoder = MockMedSAMEncoder()
            self.decoder = MockMedSAMDecoder()
            
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        for param in self.decoder.parameters():
            param.requires_grad = True

    def predict_masks(self, x_3ch: torch.Tensor) -> LungMaskOutput:
        # x_3ch: [B, 3, 1024, 1024]
        if x_3ch.ndim != 4 or x_3ch.shape[1] != 3 or tuple(x_3ch.shape[2:]) != (1024, 1024):
            raise ValueError(f"Expected input [B, 3, 1024, 1024], got {tuple(x_3ch.shape)}")
            
        img_emb = self.encoder(x_3ch)
        
        B = x_3ch.shape[0]
        bbox = torch.tensor([[0.0, 0.0, 1024.0, 1024.0]], device=x_3ch.device).repeat(B, 1)
        
        mask_logits = self.decoder(img_emb, bbox)
        
        mask_prob = torch.sigmoid(mask_logits)
        mask_256 = (mask_prob > 0.5).float()
        
        mask_1024 = F.interpolate(mask_256, size=(1024, 1024), mode="nearest")
        
        return LungMaskOutput(
            lung_mask_256=mask_256,
            lung_mask_1024=mask_1024,
            lung_logits_256=mask_logits,
        )

    def forward(self, x_3ch: torch.Tensor) -> torch.Tensor:
        return self.predict_masks(x_3ch).lung_mask_1024

def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    
    return 0.5 * bce + 0.5 * dice.mean()
