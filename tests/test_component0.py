from __future__ import annotations

import numpy as np
import torch

from src.components import component0_qc as qc


def make_sample(
    *,
    dataset_id: str = "shenzhen",
    image: np.ndarray | None = None,
    view: str | None = "PA",
) -> dict:
    if image is None:
        image = np.linspace(0, 255, 700 * 700, dtype=np.uint8).reshape(700, 700)

    return {
        "image": image,
        "image_id": "demo-001",
        "dataset_id": dataset_id,
        "domain_id": 0,
        "view": view,
        "pixel_spacing_cm": 0.017,
        "path": "demo.png",
        "labels": {"tb": 0},
    }


def test_normalise_by_dataset_scales_montgomery_12bit() -> None:
    image = np.full((700, 700), 4095, dtype=np.uint16)
    normalized = qc.normalise_by_dataset(image, "montgomery")
    assert normalized.dtype == torch.float32
    assert torch.isclose(normalized.max(), torch.tensor(1.0))
    assert torch.isclose(normalized.min(), torch.tensor(1.0))


def test_normalise_by_dataset_scales_8bit_sources() -> None:
    image = np.full((700, 700), 255, dtype=np.uint8)
    normalized = qc.normalise_by_dataset(image, "nih")
    assert normalized.dtype == torch.float32
    assert torch.isclose(normalized.max(), torch.tensor(1.0))
    assert torch.isclose(normalized.min(), torch.tensor(1.0))


def test_harmonise_sample_emits_expected_shapes() -> None:
    harmonised = qc.harmonise_sample(make_sample(), apply_clahe=False)

    assert tuple(harmonised.x_1024.shape) == (1, 1024, 1024)
    assert tuple(harmonised.x_224.shape) == (1, 224, 224)
    assert tuple(harmonised.x_3ch.shape) == (3, 1024, 1024)
    assert harmonised.x_1024.dtype == torch.float32
    assert harmonised.x_224.dtype == torch.float32
    assert harmonised.x_3ch.dtype == torch.float32


def test_clahe_only_touches_x1024(monkeypatch) -> None:
    sentinel = 0.123

    def fake_clahe(x_1024: torch.Tensor, **_: object) -> torch.Tensor:
        return torch.full_like(x_1024, fill_value=sentinel)

    monkeypatch.setattr(qc, "apply_clahe_x1024", fake_clahe)

    sample = make_sample(dataset_id="shenzhen")
    harmonised = qc.harmonise_sample(sample, apply_clahe=True)

    expected_x224 = qc.make_x224_txv(qc.normalise_by_dataset(sample["image"], sample["dataset_id"]))
    expected_x3ch = qc.make_x1024(qc.normalise_by_dataset(sample["image"], sample["dataset_id"])).repeat(3, 1, 1)

    assert torch.allclose(harmonised.x_1024, torch.full((1, 1024, 1024), sentinel))
    assert torch.allclose(harmonised.x_224, expected_x224)
    assert torch.allclose(harmonised.x_3ch, expected_x3ch)


def test_metadata_pass_through_and_standardisation() -> None:
    sample = make_sample(dataset_id="nih", view="PA")
    harmonised = qc.harmonise_sample(sample, apply_clahe=False)

    assert harmonised.meta["image_id"] == "demo-001"
    assert harmonised.meta["dataset_id"] == "nih_cxr14"
    assert harmonised.meta["scanner_domain"] == "nih_cxr14"
    assert harmonised.meta["pixel_spacing_cm"] == 0.017
    assert harmonised.meta["clahe_applied"] is False
    assert harmonised.meta["qc_passed"] is True
