from __future__ import annotations

import os

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Pick the best available torch device.

    Order: explicit `prefer` > env `TB_DEVICE` > CUDA > MPS > CPU.
    MPS is chosen only for coding/light-inference on Apple Silicon; real
    training is expected on CUDA (Kaggle/Colab/RTX 2070).
    """

    choice = (prefer or os.environ.get("TB_DEVICE") or "").lower().strip()
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(device)}"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
