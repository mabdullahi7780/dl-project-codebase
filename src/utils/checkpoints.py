from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn


def compute_file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Reads the file in 8 MB chunks so large checkpoint files do not need
    to be fully buffered in memory.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        64-character lowercase hex string, e.g.
        ``"a3f5c…"`` — the SHA-256 digest of the file bytes.

    Raises:
        FileNotFoundError: If the path does not point to an existing file.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Cannot compute SHA-256: file not found at {target}")

    hasher = hashlib.sha256()
    chunk_size = 8 * 1024 * 1024  # 8 MB
    with target.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    candidate: dict[str, torch.Tensor] | None = None
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, dict) and any(isinstance(value, torch.Tensor) for value in nested.values()):
                candidate = {str(name): value for name, value in nested.items() if isinstance(value, torch.Tensor)}
                break

        if candidate is None and any(isinstance(value, torch.Tensor) for value in payload.values()):
            candidate = {str(name): value for name, value in payload.items() if isinstance(value, torch.Tensor)}

    if candidate is None:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a recognizable torch state_dict payload."
        )

    return _strip_module_prefix(candidate)


def load_checkpoint_into_module(
    module: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> nn.Module:
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    module.load_state_dict(state_dict, strict=strict)
    return module
