"""Fast, synthetic tests for the Component 4 fine-tuning pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.components.component4_lung import Component4MedSAM, bce_dice_loss
from src.data.component4_lung_dataset import (
    Component4LungDataset,
    collate_component4_batch,
    load_and_merge_binary_mask,
    parse_manifest,
)


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _make_image(tmp_path: Path, name: str, size: int = 1024) -> Path:
    rng = np.random.default_rng(0)
    arr = rng.integers(32, 224, size=(size, size), dtype=np.uint8)
    path = tmp_path / "images" / f"{name}.png"
    _write_png(path, arr)
    return path


def _make_mask(tmp_path: Path, name: str, size: int = 1024, subdir: str = "masks") -> Path:
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    path = tmp_path / subdir / f"{name}.png"
    _write_png(path, arr)
    return path


def _write_manifest(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "mask_path", "dataset", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def test_parse_manifest_roundtrip(tmp_path: Path) -> None:
    image = _make_image(tmp_path, "a")
    mask = _make_mask(tmp_path, "a")
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "sample_id": "a",
                "image_path": str(image),
                "mask_path": str(mask),
                "dataset": "shenzhen",
                "split": "train",
            }
        ],
    )
    records = parse_manifest(manifest, split="train")
    assert len(records) == 1
    assert records[0].dataset == "shenzhen"
    assert records[0].mask_paths == (mask,)


def test_parse_manifest_rejects_missing_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bad.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("sample_id,image_path\n")
        handle.write("a,foo.png\n")
    with pytest.raises(ValueError, match="missing required columns"):
        parse_manifest(manifest_path)


def test_parse_manifest_rejects_missing_files(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "sample_id": "a",
                "image_path": str(tmp_path / "nope.png"),
                "mask_path": str(tmp_path / "nope_mask.png"),
                "dataset": "shenzhen",
                "split": "train",
            }
        ],
    )
    with pytest.raises(FileNotFoundError):
        parse_manifest(manifest)


def test_load_and_merge_binary_mask_union(tmp_path: Path) -> None:
    left = np.zeros((64, 64), dtype=np.uint8)
    right = np.zeros((64, 64), dtype=np.uint8)
    left[:, :32] = 255
    right[:, 32:] = 128
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    _write_png(left_path, left)
    _write_png(right_path, right)
    merged = load_and_merge_binary_mask((left_path, right_path))
    assert merged.shape == (64, 64)
    assert merged.dtype == np.uint8
    assert merged.sum() == 64 * 64  # full union


def test_load_and_merge_rejects_empty_mask(tmp_path: Path) -> None:
    empty = tmp_path / "empty.png"
    _write_png(empty, np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="empty"):
        load_and_merge_binary_mask((empty,))


def test_manifest_pipe_merges_left_right(tmp_path: Path) -> None:
    image = _make_image(tmp_path, "m1")
    left = _make_mask(tmp_path, "m1", subdir="leftMask")
    right = _make_mask(tmp_path, "m1", subdir="rightMask")
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "sample_id": "m1",
                "image_path": str(image),
                "mask_path": f"{left}|{right}",
                "dataset": "montgomery",
                "split": "train",
            }
        ],
    )
    records = parse_manifest(manifest, split="train")
    dataset = Component4LungDataset(records)
    item = dataset[0]
    assert tuple(item["x_3ch"].shape) == (3, 1024, 1024)
    assert tuple(item["mask"].shape) == (1, 256, 256)
    assert float(item["mask"].max()) == 1.0
    assert float(item["mask"].min()) == 0.0


def test_training_step_smoke(tmp_path: Path) -> None:
    image = _make_image(tmp_path, "s1")
    mask = _make_mask(tmp_path, "s1")
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "sample_id": "s1",
                "image_path": str(image),
                "mask_path": str(mask),
                "dataset": "shenzhen",
                "split": "train",
            }
        ],
    )
    records = parse_manifest(manifest, split="train")
    dataset = Component4LungDataset(records)
    batch = collate_component4_batch([dataset[0]])

    model = Component4MedSAM(backend="mock")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    logits = model.forward_logits(batch["x_3ch"])
    assert tuple(logits.shape) == (1, 1, 256, 256)
    loss = bce_dice_loss(logits, batch["mask"])
    loss.backward()
    # Only decoder should have gradients.
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"missing grad on trainable param {name}"
    optimizer.step()


def test_checkpoint_save_and_load_roundtrip(tmp_path: Path) -> None:
    model_a = Component4MedSAM(backend="mock")
    ckpt = tmp_path / "ckpt.pt"
    torch.save(
        {
            "decoder_state_dict": model_a.decoder_state_dict(),
            "epoch": 3,
            "best_dice": 0.75,
        },
        ckpt,
    )
    model_b = Component4MedSAM(backend="mock")
    # Perturb b's decoder so reload must actually overwrite.
    with torch.no_grad():
        for p in model_b.decoder.parameters():
            p.add_(1.0)
    payload = torch.load(ckpt, map_location="cpu")
    model_b.load_decoder_state_dict(payload["decoder_state_dict"])

    for (ka, va), (kb, vb) in zip(
        model_a.decoder.state_dict().items(),
        model_b.decoder.state_dict().items(),
    ):
        assert ka == kb
        assert torch.allclose(va, vb)
    assert payload["best_dice"] == 0.75
