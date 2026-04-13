from __future__ import annotations

from pathlib import Path

import torch

from src.utils.checkpoints import load_checkpoint_into_module, load_checkpoint_state_dict


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
