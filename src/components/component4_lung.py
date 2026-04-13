from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.checkpoints import load_checkpoint_into_module


MEDSAM_IMAGE_SIZE = 1024
LOW_RES_MASK_SIZE = 256


@dataclass(slots=True)
class LungMaskOutput:
    lung_mask_256: torch.Tensor
    lung_mask_1024: torch.Tensor
    lung_logits_256: torch.Tensor


def _has_segment_anything() -> bool:
    return importlib.util.find_spec("segment_anything") is not None


def _checkpoint_available(checkpoint_path: str | None) -> bool:
    return checkpoint_path is not None and Path(checkpoint_path).is_file()


def resolve_component4_backend(backend: str, checkpoint_path: str | None) -> str:
    if backend in {"mock", "medsam"}:
        return backend
    if backend != "auto":
        raise ValueError(f"Unsupported Component 4 backend {backend!r}.")
    if _has_segment_anything() and _checkpoint_available(checkpoint_path):
        return "medsam"
    return "mock"


class MockMedSAMEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 256, 64, 64), device=x.device, dtype=x.dtype)


class MockMedSAMDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, image_embeddings: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        del bbox
        mask = self.conv(image_embeddings)
        return F.interpolate(
            mask,
            size=(LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE),
            mode="bilinear",
            align_corners=False,
        )


class Component4MedSAM(nn.Module):
    """Component 4 lung module with a real MedSAM path and a mock fallback."""

    def __init__(
        self,
        backend: str = "auto",
        *,
        checkpoint_path: str | None = None,
        model_type: str = "vit_b",
        mask_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.mask_threshold = float(mask_threshold)
        self.backend = resolve_component4_backend(backend, checkpoint_path)
        self.active_backend = self.backend

        if self.backend == "medsam":
            self._init_medsam_backend()
        else:
            self._init_mock_backend()

        for param in self.encoder.parameters():
            param.requires_grad = False

        prompt_encoder = getattr(self, "prompt_encoder", None)
        if isinstance(prompt_encoder, nn.Module):
            for param in prompt_encoder.parameters():
                param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = True

    def _init_mock_backend(self) -> None:
        self.encoder = MockMedSAMEncoder()
        self.decoder = MockMedSAMDecoder()
        self.input_size = MEDSAM_IMAGE_SIZE

    def _init_medsam_backend(self) -> None:
        if not _has_segment_anything():
            raise ImportError(
                "segment_anything is not installed. Install it or set Component 4 backend to 'mock'."
            )
        if not _checkpoint_available(self.checkpoint_path):
            raise FileNotFoundError(
                "Component 4 requires a MedSAM checkpoint file. "
                f"Got checkpoint_path={self.checkpoint_path!r}."
            )

        from segment_anything import sam_model_registry  # type: ignore

        try:
            sam = sam_model_registry[self.model_type](checkpoint=None)
        except KeyError as exc:
            raise ValueError(
                f"Unsupported SAM model_type {self.model_type!r}. Expected one of: {sorted(sam_model_registry)}."
            ) from exc
        load_checkpoint_into_module(sam, self.checkpoint_path)

        self.encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.decoder = sam.mask_decoder
        self.input_size = int(getattr(sam.image_encoder, "img_size", MEDSAM_IMAGE_SIZE))

        self.register_buffer(
            "pixel_mean",
            sam.pixel_mean.detach().clone().float().view(1, 3, 1, 1) / 255.0,
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            sam.pixel_std.detach().clone().float().view(1, 3, 1, 1) / 255.0,
            persistent=False,
        )

    def _encode_image(self, x_3ch: torch.Tensor) -> torch.Tensor:
        if self.backend == "medsam":
            normalized = (x_3ch - self.pixel_mean) / self.pixel_std
            return self.encoder(normalized)
        return self.encoder(x_3ch)

    def _whole_image_box(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        max_coord = float(self.input_size - 1)
        return torch.tensor(
            [[0.0, 0.0, max_coord, max_coord]],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1)

    def _decode_real_masks(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = image_embeddings.shape[0]
        boxes = self._whole_image_box(
            batch_size=batch_size,
            device=image_embeddings.device,
            dtype=image_embeddings.dtype,
        )
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, _ = self.decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks

    def predict_masks(self, x_3ch: torch.Tensor) -> LungMaskOutput:
        if x_3ch.ndim != 4 or x_3ch.shape[1] != 3 or tuple(x_3ch.shape[2:]) != (MEDSAM_IMAGE_SIZE, MEDSAM_IMAGE_SIZE):
            raise ValueError(
                f"Expected input [B, 3, {MEDSAM_IMAGE_SIZE}, {MEDSAM_IMAGE_SIZE}], got {tuple(x_3ch.shape)}"
            )

        image_embeddings = self._encode_image(x_3ch)

        if self.backend == "medsam":
            mask_logits = self._decode_real_masks(image_embeddings)
        else:
            batch_size = x_3ch.shape[0]
            bbox = self._whole_image_box(batch_size=batch_size, device=x_3ch.device, dtype=x_3ch.dtype)
            mask_logits = self.decoder(image_embeddings, bbox)

        if tuple(mask_logits.shape[-2:]) != (LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE):
            mask_logits = F.interpolate(
                mask_logits,
                size=(LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        mask_prob = torch.sigmoid(mask_logits)
        mask_256 = (mask_prob > self.mask_threshold).float()
        mask_1024 = F.interpolate(
            mask_256,
            size=(MEDSAM_IMAGE_SIZE, MEDSAM_IMAGE_SIZE),
            mode="nearest",
        )

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
