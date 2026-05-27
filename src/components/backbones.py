"""Frozen image-feature backbones for the agentic Timika pipeline.

Primary backbone = **RAD-DINO** (``microsoft/rad-dino``), a DINOv2 ViT pretrained
self-supervised on ~900k chest X-rays. We use it FROZEN and cache its 768-d CLS
embedding per image (``scripts/cache_features.py``), so every downstream head
trains on cached vectors in seconds — the whole experimental ladder costs minutes,
not hours. RAD-DINO's authors report fine-tuning is usually unnecessary.

Fallbacks / controls:
  - ``txrv``     : TorchXRayVision DenseNet121 (multi-dataset CXR), 1024-d. Friction-free.
  - ``densenet`` : torchvision ImageNet DenseNet121, 1024-d. Matches the Kantipudi backbone.

All backbones expose ``embed(list[PIL.Image]) -> np.ndarray [N, dim]`` and ``dim``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class RadDinoBackbone:
    name = "rad-dino"

    def __init__(self, device: torch.device, model_id: str = "microsoft/rad-dino") -> None:
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.dim = int(self.model.config.hidden_size)  # 768 for base

    @torch.inference_mode()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        emb = getattr(out, "pooler_output", None)
        if emb is None:  # CLS token of the last hidden state
            emb = out.last_hidden_state[:, 0]
        return emb.float().cpu().numpy()


class TorchXRVBackbone:
    name = "txrv"
    dim = 1024

    def __init__(self, device: torch.device, weights: str = "densenet121-res224-all") -> None:
        import torchxrayvision as xrv
        self.xrv = xrv
        self.model = xrv.models.DenseNet(weights=weights).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        batch = []
        for im in images:
            arr = np.asarray(im.convert("L"), dtype=np.float32)
            arr = self.xrv.datasets.normalize(arr, 255)  # -> roughly [-1024, 1024]
            batch.append(torch.from_numpy(arr)[None])  # [1, H, W]
        x = torch.stack(batch).to(self.device)  # [N, 1, H, W]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feat = self.model.features(x)           # [N, 1024, 7, 7]
        feat = F.relu(feat, inplace=True)
        return F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1).float().cpu().numpy()


class DenseNetBackbone:
    name = "densenet"
    dim = 1024

    def __init__(self, device: torch.device, pretrained: bool = True) -> None:
        from src.components.baseline_paper import _DenseNet121Features
        self.model = _DenseNet121Features(pretrained).to(device).eval()
        self.device = device
        from torchvision import transforms
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.tf(im.convert("RGB")) for im in images]).to(self.device)
        return self.model(x).float().cpu().numpy()


def build_backbone(name: str, device: torch.device, model_id: str | None = None):
    if name == "rad-dino":
        return RadDinoBackbone(device, model_id or "microsoft/rad-dino")
    if name == "txrv":
        return TorchXRVBackbone(device)
    if name == "densenet":
        return DenseNetBackbone(device)
    raise ValueError(f"unknown backbone {name!r} (use rad-dino | txrv | densenet)")
