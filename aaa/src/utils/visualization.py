from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _to_numpy_2d(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image/mask, got shape {array.shape}.")
    return array


def save_mask_png(mask: torch.Tensor | np.ndarray, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = (_to_numpy_2d(mask) > 0.5).astype(np.uint8) * 255
    Image.fromarray(array, mode="L").save(destination)
    return destination


def save_overlay_png(
    image_01: torch.Tensor | np.ndarray,
    lesion_mask: torch.Tensor | np.ndarray,
    lung_mask: torch.Tensor | np.ndarray,
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    image = _to_numpy_2d(image_01).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    lesion = (_to_numpy_2d(lesion_mask) > 0.5)
    lung = (_to_numpy_2d(lung_mask) > 0.5)

    rgb = np.stack([image, image, image], axis=-1)
    rgb[lung, 1] = np.maximum(rgb[lung, 1], 0.6)
    rgb[lesion, 0] = 1.0
    rgb[lesion, 1] *= 0.35
    rgb[lesion, 2] *= 0.35

    Image.fromarray((rgb * 255.0).round().astype(np.uint8), mode="RGB").save(destination)
    return destination
