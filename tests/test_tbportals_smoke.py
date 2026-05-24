"""Smoke test for the TB Portals stack on a synthetic dataset.

Runs end-to-end without real data or pretrained-weight downloads:
synthetic manifest -> split (+ leakage assertion) -> dataset -> MoE+DANN model
forward -> eval metrics. Keeps the whole Day-1 pipeline honest before the real
TB Portals data lands.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.components.component_alp_moe import ALPMoEConfig, ALPMoEModel, load_balancing_loss
from src.data.tbportals import (
    assert_no_patient_leakage,
    load_manifest,
    make_country_split,
    make_synthetic_dataset,
)
from src.data.tbportals_dataset import TBPortalsDataset, build_country_index
from src.evaluation.eval_tbportals import Predictions, evaluate_split, predict


def test_synthetic_manifest_split_and_no_leakage(tmp_path):
    manifest_path = make_synthetic_dataset(tmp_path, n=120, seed=0)
    df = load_manifest(manifest_path)
    assert len(df) == 120

    train_df, val_df, test_df = make_country_split(df, "Romania", val_fraction=0.25, seed=0)
    assert set(test_df["country"]) == {"Romania"}
    assert "Romania" not in set(train_df["country"])
    assert_no_patient_leakage(train_df, val_df, test_df)  # must not raise


def test_dataset_item_shapes(tmp_path):
    manifest_path = make_synthetic_dataset(tmp_path, n=60, seed=1)
    df = load_manifest(manifest_path)
    train_df, _, _ = make_country_split(df, "Moldova", seed=1)
    c2i = build_country_index(train_df)
    ds = TBPortalsDataset(train_df, train=True, country_to_idx=c2i)
    item = ds[0]
    assert item["image"].shape == (3, 224, 224)
    assert 0.0 <= float(item["alp"]) <= 1.0
    assert float(item["cavity"]) in (0.0, 1.0)
    assert int(item["country_idx"]) >= 0


def test_model_forward_and_eval(tmp_path):
    manifest_path = make_synthetic_dataset(tmp_path, n=80, seed=2)
    df = load_manifest(manifest_path)
    train_df, val_df, test_df = make_country_split(df, "Kazakhstan", seed=2)
    c2i = build_country_index(train_df)

    cfg = ALPMoEConfig(num_experts=3, use_dann=True, num_countries=len(c2i), pretrained=False)
    model = ALPMoEModel(cfg)
    loader = DataLoader(
        TBPortalsDataset(test_df, train=False, country_to_idx=c2i),
        batch_size=8,
    )
    batch = next(iter(loader))
    out = model(batch["image"], dann_lambda=1.0)

    b = batch["image"].shape[0]
    assert out.alp.shape == (b,)
    assert torch.all((out.alp >= 0) & (out.alp <= 1))
    assert out.cavity_logit.shape == (b,)
    assert out.gate_weights.shape == (b, 3)
    assert torch.allclose(out.gate_weights.sum(dim=1), torch.ones(b), atol=1e-4)
    assert out.country_logit.shape == (b, len(c2i))
    assert float(load_balancing_loss(out.gate_weights)) >= 1.0 - 1e-4

    preds = predict(model, loader, torch.device("cpu"))
    metrics = evaluate_split(preds, cavity_threshold=0.5)
    for key in ("mae", "pearson", "mae_pct", "rmse", "r2"):
        assert key in metrics["timika"]
    assert "auc" in metrics["cavity"]
