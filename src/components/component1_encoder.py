from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.checkpoints import load_checkpoint_into_module


SAM_INPUT_SIZE = 1024
SAM_PATCH_SIZE = 16
SAM_EMBED_SIZE = 64
SAM_NECK_CHANNELS = 256


@dataclass(slots=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("qkv",)


@dataclass(slots=True)
class Component1EncoderConfig:
    backend: str = "auto"
    checkpoint_path: str | None = None
    freeze_backbone: bool = True
    input_channels: int = 3
    input_size: int = SAM_INPUT_SIZE
    patch_size: int = SAM_PATCH_SIZE
    embed_dim: int = SAM_NECK_CHANNELS
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    lora: LoRAConfig = field(default_factory=LoRAConfig)


class LoRALinear(nn.Module):
    """Low-rank adapter wrapper for an existing Linear layer."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_down = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(self.dropout(x)))
        return base_out + (lora_out * self.scale)


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _module_matches(name: str, target_modules: Iterable[str]) -> bool:
    return any(target in name for target in target_modules)


def inject_lora_modules(module: nn.Module, config: LoRAConfig) -> list[str]:
    """Replace matching Linear submodules with LoRA-wrapped versions."""

    replaced: list[str] = []
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and _module_matches(child_name, config.target_modules):
            setattr(
                module,
                child_name,
                LoRALinear(
                    child,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                ),
            )
            replaced.append(child_name)
            continue

        nested = inject_lora_modules(child, config)
        replaced.extend(f"{child_name}.{name}" for name in nested)
    return replaced


def trainable_parameter_names(module: nn.Module) -> list[str]:
    return [name for name, param in module.named_parameters() if param.requires_grad]


class MockAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed dim {dim} must be divisible by num_heads {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out)


class MockMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MockTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MockAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MockMLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MockSAMViTHImageEncoder(nn.Module):
    """Development fallback that preserves the SAM ViT-H tensor contract."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        patch_size: int = SAM_PATCH_SIZE,
        embed_dim: int = SAM_NECK_CHANNELS,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.blocks = nn.ModuleList(
            MockTransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        )
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, 3, 1024, 1024], got {tuple(x.shape)}.")
        if x.shape[1] != 3:
            raise ValueError(f"Component 1 expects 3-channel SAM input, got {tuple(x.shape)}.")
        if tuple(x.shape[-2:]) != (SAM_INPUT_SIZE, SAM_INPUT_SIZE):
            raise ValueError(
                f"Component 1 expects spatial size {(SAM_INPUT_SIZE, SAM_INPUT_SIZE)}, got {tuple(x.shape[-2:])}."
            )

        x = self.patch_embed(x)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        x = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return self.neck(x)


class SegmentAnythingViTHEncoder(nn.Module):
    """Thin wrapper around the official Segment Anything ViT-H image encoder."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        super().__init__()
        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "segment_anything is not installed. Install it or switch config.encoder.backend to 'mock'."
            ) from exc

        sam = sam_model_registry["vit_h"](checkpoint=None)
        if checkpoint_path is not None:
            load_checkpoint_into_module(sam, checkpoint_path)
        self.image_encoder = sam.image_encoder
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(1, 3, 1, 1) / 255.0,
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(1, 3, 1, 1) / 255.0,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected SAM input [B, 3, 1024, 1024], got {tuple(x.shape)}.")
        x = (x - self.pixel_mean) / self.pixel_std
        return self.image_encoder(x)


class Component1Encoder(nn.Module):
    """Shared image encoder that returns SAM-style image embeddings."""

    def __init__(self, backbone: nn.Module, *, lora_targets: list[str], active_backend: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.lora_targets = tuple(lora_targets)
        self.active_backend = active_backend

    def forward(self, x_3ch: torch.Tensor) -> torch.Tensor:
        img_emb = self.backbone(x_3ch)
        expected_shape = (SAM_NECK_CHANNELS, SAM_EMBED_SIZE, SAM_EMBED_SIZE)
        if img_emb.ndim != 4 or tuple(img_emb.shape[1:]) != expected_shape:
            raise ValueError(
                "Component 1 encoder must return [B, 256, 64, 64]. "
                f"Got {tuple(img_emb.shape)}."
            )
        return img_emb


def _has_segment_anything() -> bool:
    return importlib.util.find_spec("segment_anything") is not None


def _checkpoint_available(checkpoint_path: str | None) -> bool:
    return checkpoint_path is not None and Path(checkpoint_path).is_file()


def resolve_component1_backend(config: Component1EncoderConfig) -> str:
    if config.backend in {"mock", "segment_anything"}:
        return config.backend
    if config.backend != "auto":
        raise ValueError(f"Unsupported Component 1 backend {config.backend!r}.")
    if _has_segment_anything() and _checkpoint_available(config.checkpoint_path):
        return "segment_anything"
    return "mock"


def extract_trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def save_trainable_state_dict(module: nn.Module, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = extract_trainable_state_dict(module)
    if destination.suffix == ".safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "Saving `.safetensors` requires the `safetensors` package."
            ) from exc
        save_file(payload, str(destination))
    else:
        torch.save(payload, destination)
    return destination


def build_component1_encoder(config: Component1EncoderConfig | None = None) -> Component1Encoder:
    cfg = config or Component1EncoderConfig()
    backend = resolve_component1_backend(cfg)

    if backend == "segment_anything":
        backbone: nn.Module = SegmentAnythingViTHEncoder(checkpoint_path=cfg.checkpoint_path)
    elif backend == "mock":
        backbone = MockSAMViTHImageEncoder(
            input_channels=cfg.input_channels,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
        )
    else:
        raise ValueError(f"Unsupported Component 1 backend {cfg.backend!r}.")

    if cfg.freeze_backbone:
        freeze_module(backbone)

    lora_targets = inject_lora_modules(backbone, cfg.lora)
    return Component1Encoder(backbone, lora_targets=lora_targets, active_backend=backend)
