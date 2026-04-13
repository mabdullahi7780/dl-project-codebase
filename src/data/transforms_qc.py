from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def to_grayscale_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert a grayscale image to a CPU float32 tensor with shape [H, W]."""

    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        raise TypeError(
            f"Unsupported image type: expected numpy.ndarray or torch.Tensor, got {type(image)!r}."
        )

    if tensor.ndim == 2:
        pass
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    elif tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    else:
        raise ValueError(
            "Component 0 expects a grayscale image with shape [H, W], [1, H, W], or [H, W, 1]."
        )

    return tensor.to(dtype=torch.float32).contiguous()


def ensure_chw(image: torch.Tensor) -> torch.Tensor:
    """Ensure a grayscale tensor is represented as [1, H, W]."""

    if image.ndim == 2:
        return image.unsqueeze(0)
    if image.ndim == 3 and image.shape[0] == 1:
        return image
    raise ValueError("Expected a grayscale tensor with shape [H, W] or [1, H, W].")


def resize_chw(image: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Resize a grayscale tensor to the requested size using bilinear interpolation."""

    chw = ensure_chw(image)
    if tuple(chw.shape[-2:]) == tuple(size):
        return chw

    resized = F.interpolate(
        chw.unsqueeze(0),
        size=tuple(int(dim) for dim in size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def center_crop_square(image: torch.Tensor) -> torch.Tensor:
    """Center crop a grayscale image to its largest possible square."""

    chw = ensure_chw(image)
    _, height, width = chw.shape
    crop = min(height, width)
    top = (height - crop) // 2
    left = (width - crop) // 2
    return chw[:, top : top + crop, left : left + crop]


def apply_clahe_2d(
    image: torch.Tensor,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> torch.Tensor:
    """Apply CLAHE to a normalized grayscale image in [0, 1]."""

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for CLAHE. Install `opencv-python-headless` before enabling CLAHE."
        ) from exc

    chw = ensure_chw(image)
    image_uint8 = (
        chw.squeeze(0).clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8).cpu().numpy()
    )
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )
    equalized = clahe.apply(image_uint8).astype(np.float32) / 255.0
    return torch.from_numpy(equalized).unsqueeze(0)
