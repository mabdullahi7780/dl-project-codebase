from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.components.component1_dann import (
    Component1DANNModel,
    compute_dann_lambda,
    domain_classification_loss,
)
from src.components.component1_encoder import (
    Component1EncoderConfig,
    build_component1_encoder,
    resolve_component1_backend,
    save_trainable_state_dict,
)
from src.training.train_component1_dann import (
    Component1DomainDataset,
    DomainSampleRef,
    apply_domain_adaptation_augment,
    build_component1_manifest,
)


def _write_png(path: Path, *, value: int = 180) -> None:
    image = np.full((512, 512), value, dtype=np.uint8)
    Image.fromarray(image, mode="L").save(path)


def test_component1_encoder_returns_expected_tensor_contract() -> None:
    encoder = build_component1_encoder(Component1EncoderConfig(backend="auto"))

    x = torch.rand(2, 3, 1024, 1024)
    img_emb = encoder(x)

    assert tuple(img_emb.shape) == (2, 256, 64, 64)
    assert any("qkv" in name for name in encoder.lora_targets)
    assert encoder.active_backend == "mock"


def test_component1_auto_backend_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.components.component1_encoder._has_segment_anything", lambda: True)
    monkeypatch.setattr("src.components.component1_encoder._checkpoint_available", lambda path: path == "demo.pth")

    assert resolve_component1_backend(Component1EncoderConfig(backend="auto", checkpoint_path="demo.pth")) == "segment_anything"
    assert resolve_component1_backend(Component1EncoderConfig(backend="auto", checkpoint_path=None)) == "mock"


def test_component1_dann_forward_and_loss() -> None:
    encoder = build_component1_encoder(Component1EncoderConfig(backend="mock"))
    model = Component1DANNModel(encoder=encoder)

    img_emb, dom_logits = model(torch.rand(2, 3, 1024, 1024), lambda_=0.5)
    loss = domain_classification_loss(dom_logits, torch.tensor([0, 3]))

    assert tuple(img_emb.shape) == (2, 256, 64, 64)
    assert tuple(dom_logits.shape) == (2, 4)
    assert loss.ndim == 0
    assert compute_dann_lambda(5, ramp_epochs=10) == 0.5


def test_build_component1_manifest_supports_disk_and_nih_tar_layout(tmp_path: Path) -> None:
    raw_root = tmp_path / "TB_DATA" / "raw"
    montgomery_root = raw_root / "montgomery" / "images"
    shenzhen_root = raw_root / "shenzhen" / "images" / "images"
    tbx_root = raw_root / "tbx11k"
    nih_root = raw_root / "nih_cxr14"

    montgomery_root.mkdir(parents=True)
    shenzhen_root.mkdir(parents=True)
    (tbx_root / "imgs" / "tb").mkdir(parents=True)
    (tbx_root / "lists").mkdir(parents=True)
    (nih_root / "images").mkdir(parents=True)

    _write_png(montgomery_root / "MCUCXR_0001_0.png", value=140)
    _write_png(shenzhen_root / "CHNCXR_0001_0.png", value=160)
    _write_png(tbx_root / "imgs" / "tb" / "tb0001.png", value=200)

    (tbx_root / "lists" / "all_trainval.txt").write_text("tb/tb0001.png\n", encoding="utf-8")
    (nih_root / "train_val_list.txt").write_text("00000001_000.png\n", encoding="utf-8")

    buffer = io.BytesIO()
    Image.fromarray(np.full((1024, 1024), 90, dtype=np.uint8), mode="L").save(buffer, format="PNG")
    payload = buffer.getvalue()

    archive_path = nih_root / "images" / "images_001.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        info = tarfile.TarInfo(name="images/00000001_000.png")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    paths_yaml = {
        "datasets": {
            "montgomery": str(raw_root / "montgomery"),
            "shenzhen": str(raw_root / "shenzhen"),
            "tbx11k": str(tbx_root),
            "nih_cxr14": str(nih_root),
        }
    }
    component1_yaml = {
        "component1_dann": {
            "data": {
                "manifest_cache": str(tmp_path / "nih_index.json"),
                "tbx_list": "all_trainval.txt",
                "nih_split": None,
                "nih_metadata_csv": "Data_Entry_2017_v2020.csv",
                "domain_sampling_weights": {
                    "montgomery": 100.0,
                    "shenzhen": 20.0,
                    "tbx11k": 2.0,
                    "nih_cxr14": 1.0,
                },
            }
        }
    }

    paths_config = tmp_path / "paths.yaml"
    component1_config = tmp_path / "component1_dann.yaml"
    (nih_root / "Data_Entry_2017_v2020.csv").write_text("Image Index,Finding Labels\n00000001_000.png,No Finding\n", encoding="utf-8")
    paths_config.write_text(yaml.safe_dump(paths_yaml), encoding="utf-8")
    component1_config.write_text(yaml.safe_dump(component1_yaml), encoding="utf-8")

    manifest = build_component1_manifest(paths_config=paths_config, component1_config=component1_config)

    assert len(manifest) == 4
    assert {sample.dataset_id for sample in manifest} == {"montgomery", "shenzhen", "tbx11k", "nih_cxr14"}
    assert any(sample.archive_path is not None for sample in manifest if sample.dataset_id == "nih_cxr14")


def test_apply_domain_adaptation_augment_preserves_shape_and_range() -> None:
    x = torch.full((3, 1024, 1024), 0.5, dtype=torch.float32)
    torch.manual_seed(123)
    augmented = apply_domain_adaptation_augment(x)

    assert tuple(augmented.shape) == (3, 1024, 1024)
    assert float(augmented.min()) >= 0.0
    assert float(augmented.max()) <= 1.0
    assert not torch.allclose(augmented, x)


def test_dataset_augmentation_only_applies_to_selected_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [
        DomainSampleRef(dataset_id="montgomery", domain_id=0, source="a", image_path="ignored"),
        DomainSampleRef(dataset_id="nih_cxr14", domain_id=3, source="b", image_path="ignored"),
    ]
    dataset = Component1DomainDataset(samples, apply_augmentation=True)

    def fake_load_image(sample: DomainSampleRef) -> np.ndarray:
        max_value = 255 if sample.dataset_id == "nih_cxr14" else 4095
        return np.linspace(0, max_value, 700 * 700, dtype=np.float32).reshape(700, 700).astype(
            np.uint8 if max_value == 255 else np.uint16
        )

    monkeypatch.setattr(
        dataset,
        "_load_image",
        fake_load_image,
    )
    monkeypatch.setattr(
        "src.training.train_component1_dann.apply_domain_adaptation_augment",
        lambda x_3ch, **_: torch.zeros_like(x_3ch),
    )

    mont = dataset[0]
    nih = dataset[1]

    assert torch.allclose(mont["x_3ch"], torch.zeros_like(mont["x_3ch"]))
    assert not torch.allclose(nih["x_3ch"], torch.zeros_like(nih["x_3ch"]))


def test_save_trainable_state_dict_saves_only_lora_params(tmp_path: Path) -> None:
    encoder = build_component1_encoder(Component1EncoderConfig(backend="mock"))
    save_path = tmp_path / "adapters.pt"

    save_trainable_state_dict(encoder, save_path)
    payload = torch.load(save_path)

    assert payload
    assert all("lora_" in key for key in payload)
