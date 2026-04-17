"""Component 5 — MoE Expert Decoders.

Four lightweight expert decoders, each specialised for a different TB
pathology pattern:

    Expert 1: Consolidation — dense opacity regions
    Expert 2: Cavity         — ring-enhancing lesions (used by C8 for Timika)
    Expert 3: Fibrosis       — linear / reticular patterns (also used by C7 refiner)
    Expert 4: Nodule         — small focal opacities

Every expert shares the same ``ExpertDecoder`` interface so they are
interchangeable from the perspective of the routing gate (C3) and the
fusion module (C6).

Shape contract
--------------
    input:  image_emb  [B, 256, 64, 64]   — from Component 1 shared encoder
    output: mask_logits [B, 1, 256, 256]   — raw logits (pre-sigmoid)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Expert interface
# ---------------------------------------------------------------------------

class ExpertDecoder(Protocol):
    """Shared interface that every expert must satisfy (plan.md §13)."""

    expert_name: str

    def forward(
        self,
        image_emb: torch.Tensor,
        prompts: torch.Tensor | None = None,
        dense_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return mask logits ``[B, 1, 256, 256]``."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERT_NAMES: tuple[str, ...] = (
    "consolidation",
    "cavity",
    "fibrosis",
    "nodule",
)

NUM_EXPERTS = len(EXPERT_NAMES)


@dataclass(slots=True)
class ExpertDecoderConfig:
    in_channels: int = 256
    mid_channels: int = 128
    out_channels: int = 1
    num_experts: int = NUM_EXPERTS
    dropout: float = 0.1
    use_skip: bool = True


# ---------------------------------------------------------------------------
# Lightweight CNN expert decoder
# ---------------------------------------------------------------------------

class _UpsampleBlock(nn.Module):
    """Conv → BN → GELU → 2x Upsample."""

    def __init__(self, in_ch: int, out_ch: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.drop(self.act(self.bn(self.conv(x))))


class LightweightExpertDecoder(nn.Module):
    """Small CNN decoder: 64x64 → 128x128 → 256x256.

    Each expert gets its own instance with independent weights.  The
    architecture is deliberately small (~0.5 M params per expert) so four
    experts together are still lighter than a single SAM mask decoder.

    Optionally accepts ``prompts`` (point embeddings) and ``dense_prompt``
    (mask prior) to match the ``ExpertDecoder`` protocol, but neither is
    required — the decoder works in a fully automatic mode when they are
    ``None``.
    """

    def __init__(
        self,
        expert_name: str,
        *,
        in_channels: int = 256,
        mid_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.expert_name = expert_name

        # Bottleneck 1x1 to reduce channel count
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

        # Optional prompt injection: project sparse prompts into a spatial
        # bias that is added to the bottleneck features.
        self.prompt_proj = nn.Linear(256, mid_channels, bias=False)

        # Two upsample stages: 64 → 128 → 256
        self.up1 = _UpsampleBlock(mid_channels, mid_channels // 2, dropout=dropout)
        self.up2 = _UpsampleBlock(mid_channels // 2, mid_channels // 4, dropout=dropout)

        # Final 1x1 head → logits
        self.head = nn.Conv2d(mid_channels // 4, 1, 1)

    def forward(
        self,
        image_emb: torch.Tensor,
        prompts: torch.Tensor | None = None,
        dense_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            image_emb:    [B, 256, 64, 64] from Component 1 encoder.
            prompts:      [B, N, 256] sparse point embeddings (optional).
            dense_prompt: [B, 1, 64, 64] dense mask prior (optional).

        Returns:
            mask_logits [B, 1, 256, 256] — raw logits, apply sigmoid for prob.
        """
        if image_emb.ndim != 4 or image_emb.shape[1] != 256:
            raise ValueError(
                f"Expert {self.expert_name!r} expects image_emb [B, 256, 64, 64], "
                f"got {tuple(image_emb.shape)}."
            )

        x = self.bottleneck(image_emb)  # [B, mid, 64, 64]

        # Inject sparse prompts as a global spatial bias
        if prompts is not None:
            # prompts: [B, N, 256] → project and pool → [B, mid, 1, 1]
            prompt_feat = self.prompt_proj(prompts).mean(dim=1)  # [B, mid]
            x = x + prompt_feat.unsqueeze(-1).unsqueeze(-1)

        # Inject dense prompt as an additive prior
        if dense_prompt is not None:
            # dense_prompt: [B, 1, 64, 64] → broadcast-add via learned scale
            x = x + dense_prompt.expand_as(x[:, :1, :, :]).repeat(1, x.shape[1], 1, 1) * 0.1

        x = self.up1(x)   # [B, mid/2, 128, 128]
        x = self.up2(x)   # [B, mid/4, 256, 256]
        return self.head(x)  # [B, 1, 256, 256]


# ---------------------------------------------------------------------------
# Expert bank — the full set of 4 experts
# ---------------------------------------------------------------------------

class ExpertBank(nn.Module):
    """Holds all four expert decoders and dispatches by index or name.

    Usage::

        bank = ExpertBank()
        masks = bank(image_emb)          # list of 4 × [B, 1, 256, 256]
        masks = bank(image_emb, indices=[0, 2])  # only experts 0 and 2
    """

    def __init__(self, config: ExpertDecoderConfig | None = None) -> None:
        super().__init__()
        cfg = config or ExpertDecoderConfig()
        self.expert_names = EXPERT_NAMES[: cfg.num_experts]

        self.experts = nn.ModuleList(
            LightweightExpertDecoder(
                expert_name=name,
                in_channels=cfg.in_channels,
                mid_channels=cfg.mid_channels,
                dropout=cfg.dropout,
            )
            for name in self.expert_names
        )

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def forward_single(
        self,
        index: int,
        image_emb: torch.Tensor,
        prompts: torch.Tensor | None = None,
        dense_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.experts[index](image_emb, prompts, dense_prompt)

    def forward(
        self,
        image_emb: torch.Tensor,
        *,
        indices: list[int] | None = None,
        prompts_per_expert: list[torch.Tensor | None] | None = None,
        dense_prompts_per_expert: list[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        """Run selected (or all) experts and return their mask logits.

        Args:
            image_emb: [B, 256, 64, 64].
            indices:   which experts to run (default: all).
            prompts_per_expert: list of prompt tensors per expert.
            dense_prompts_per_expert: list of dense prompts per expert.

        Returns:
            List of [B, 1, 256, 256] logit tensors, one per selected expert.
        """
        run_indices = indices if indices is not None else list(range(self.num_experts))

        results: list[torch.Tensor] = []
        for i, idx in enumerate(run_indices):
            prompts = None
            if prompts_per_expert is not None and i < len(prompts_per_expert):
                prompts = prompts_per_expert[i]
            dense = None
            if dense_prompts_per_expert is not None and i < len(dense_prompts_per_expert):
                dense = dense_prompts_per_expert[i]

            results.append(self.experts[idx](image_emb, prompts, dense))
        return results

    def expert_by_name(self, name: str) -> LightweightExpertDecoder:
        for expert in self.experts:
            if expert.expert_name == name:
                return expert
        raise KeyError(f"No expert named {name!r}. Available: {self.expert_names}")
