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


def _pool_tokens_to_grid(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    """[N, P, D] ViT patch tokens -> [N, grid*grid, D] via 2-D adaptive pooling.

    If P is a perfect square the tokens are reshaped to their native HpxWp grid
    before pooling (preserves 2-D layout); otherwise we fall back to 1-D adaptive
    pooling over the sequence so the function never fails on non-square inputs.
    """
    n, p, d = tokens.shape
    side = int(round(p ** 0.5))
    if side * side == p:
        g = tokens.reshape(n, side, side, d).permute(0, 3, 1, 2)   # [N, D, side, side]
        g = F.adaptive_avg_pool2d(g, (grid, grid))                 # [N, D, grid, grid]
        return g.reshape(n, d, grid * grid).permute(0, 2, 1)       # [N, grid*grid, D]
    seq = tokens.permute(0, 2, 1)                                   # [N, D, P]
    seq = F.adaptive_avg_pool1d(seq, grid * grid)                   # [N, D, grid*grid]
    return seq.permute(0, 2, 1)


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

    @torch.inference_mode()
    def embed_grid(self, images: list[Image.Image], grid: int = 7) -> np.ndarray:
        """Pooled patch-token grid [N, grid*grid, D] (A1 spatial features)."""
        inputs = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        tokens = out.last_hidden_state[:, 1:]  # drop CLS -> [N, P, D]
        return _pool_tokens_to_grid(tokens, grid).float().cpu().numpy()


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

    @torch.inference_mode()
    def embed_grid(self, images: list[Image.Image], grid: int = 7) -> np.ndarray:
        """Pooled conv-feature grid [N, grid*grid, 1024] (A1 spatial features)."""
        batch = []
        for im in images:
            arr = np.asarray(im.convert("L"), dtype=np.float32)
            arr = self.xrv.datasets.normalize(arr, 255)
            batch.append(torch.from_numpy(arr)[None])
        x = torch.stack(batch).to(self.device)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feat = F.relu(self.model.features(x), inplace=True)         # [N, 1024, 7, 7]
        feat = F.adaptive_avg_pool2d(feat, (grid, grid))            # [N, 1024, g, g]
        N, C = feat.shape[0], feat.shape[1]
        return feat.reshape(N, C, grid * grid).permute(0, 2, 1).float().cpu().numpy()


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


class BioMedCLIPBackbone:
    """BioMedCLIP visual encoder (frozen). Loaded via open_clip from HF Hub."""
    name = "biomedclip"

    def __init__(self, device: torch.device,
                 model_id: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        from open_clip import create_model_from_pretrained
        model, preprocess = create_model_from_pretrained(model_id)
        self.model = model.visual.to(device).eval()
        self.preprocess = preprocess
        self.device = device
        # ViT-B/16: width 768
        self.dim = 768

    @torch.inference_mode()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        out = self.model(x)
        # open_clip visual returns a single pooled feature [N, D]
        if isinstance(out, tuple):
            out = out[0]
        return out.float().cpu().numpy()

    @torch.inference_mode()
    def embed_grid(self, images: list[Image.Image], grid: int = 7) -> np.ndarray:
        """Pooled patch-token grid via hook on the transformer."""
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        # Use forward pre-final-norm patch tokens.
        # open_clip ViT exposes .trunk or .transformer; we monkey-patch a hook.
        feats: dict = {}
        def _hook(module, inp, out):
            feats["tokens"] = out
        # last block's residual stream
        try:
            handle = self.model.transformer.resblocks[-1].register_forward_hook(_hook)
        except AttributeError:
            handle = self.model.trunk.blocks[-1].register_forward_hook(_hook)
        _ = self.model(x)
        handle.remove()
        toks = feats["tokens"]                              # [seq, N, D] or [N, seq, D]
        if toks.shape[0] != x.shape[0]:                     # open_clip default: [seq, N, D]
            toks = toks.permute(1, 0, 2)                    # -> [N, seq, D]
        toks = toks[:, 1:, :]                               # drop CLS
        return _pool_tokens_to_grid(toks, grid).float().cpu().numpy()


class Dinov2NaturalBackbone(RadDinoBackbone):
    """DINOv2-natural (Facebook). Same HF interface as RAD-DINO; different weights."""
    name = "dinov2-natural"

    def __init__(self, device: torch.device, model_id: str = "facebook/dinov2-base"):
        super().__init__(device, model_id=model_id)


def build_backbone(name: str, device: torch.device, model_id: str | None = None):
    if name == "rad-dino":
        return RadDinoBackbone(device, model_id or "microsoft/rad-dino")
    if name == "biomedclip":
        return BioMedCLIPBackbone(device,
                                  model_id or "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    if name == "dinov2-natural":
        return Dinov2NaturalBackbone(device, model_id or "facebook/dinov2-base")
    if name == "txrv":
        return TorchXRVBackbone(device)
    if name == "densenet":
        return DenseNetBackbone(device)
    raise ValueError(f"unknown backbone {name!r} "
                     f"(use rad-dino | biomedclip | dinov2-natural | txrv | densenet)")
