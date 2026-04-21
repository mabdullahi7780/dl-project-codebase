from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from src.utils.checkpoints import (
    compute_file_sha256,
    load_checkpoint_into_module,
    load_checkpoint_state_dict,
)


def test_load_checkpoint_state_dict_supports_nested_payload_and_module_prefix(tmp_path: Path) -> None:
    path = tmp_path / "demo.pt"
    payload = {
        "state_dict": {
            "module.weight": torch.ones(2, 2),
            "module.bias": torch.zeros(2),
        }
    }
    torch.save(payload, path)

    state_dict = load_checkpoint_state_dict(path)

    assert "weight" in state_dict
    assert "bias" in state_dict
    assert all(not key.startswith("module.") for key in state_dict)


def test_load_checkpoint_into_module_restores_weights(tmp_path: Path) -> None:
    module = torch.nn.Linear(2, 2)
    expected = torch.nn.Linear(2, 2)
    expected.weight.data.fill_(3.0)
    expected.bias.data.fill_(1.5)

    path = tmp_path / "linear.pt"
    torch.save(expected.state_dict(), path)

    load_checkpoint_into_module(module, path)

    assert torch.allclose(module.weight, expected.weight)
    assert torch.allclose(module.bias, expected.bias)


# ---------------------------------------------------------------------------
# compute_file_sha256
# ---------------------------------------------------------------------------

def test_compute_file_sha256_returns_64_hex_chars(tmp_path: Path) -> None:
    test_file = tmp_path / "dummy.bin"
    test_file.write_bytes(b"hello world")
    digest = compute_file_sha256(test_file)
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_compute_file_sha256_matches_reference(tmp_path: Path) -> None:
    content = b"TB pipeline checkpoint provenance test"
    test_file = tmp_path / "ref.bin"
    test_file.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert compute_file_sha256(test_file) == expected


def test_compute_file_sha256_different_files_differ(tmp_path: Path) -> None:
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"file one content")
    f2.write_bytes(b"file two content")
    assert compute_file_sha256(f1) != compute_file_sha256(f2)


def test_compute_file_sha256_same_content_same_hash(tmp_path: Path) -> None:
    content = b"\x00" * 1024
    f1 = tmp_path / "c.bin"
    f2 = tmp_path / "d.bin"
    f1.write_bytes(content)
    f2.write_bytes(content)
    assert compute_file_sha256(f1) == compute_file_sha256(f2)


def test_compute_file_sha256_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_file_sha256(tmp_path / "nonexistent.pt")
