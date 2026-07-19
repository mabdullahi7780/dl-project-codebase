"""Tests for WS-C cavity grounding (scripts/ws_c_cavity_grounding.py).

Pure-function tests run on synthetic data (no Kaggle artefacts, no GPU). The
sextant-recovery test runs against the real local CSVs when present and is
skipped otherwise, so CI stays green on machines without the data.

Runnable with pytest, or standalone: ``python3.11 tests/test_ws_c_cavity_grounding.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.ws_c_cavity_grounding import (  # noqa: E402
    _auc,
    calibration,
    load_sextant_labels,
    patch_to_zone,
    synth_attn,
    zone_attention,
    zone_localization,
    zone_size_share,
)
from src.data.tbportals import SEXTANT_CODES  # noqa: E402


def test_patch_to_zone_partition() -> None:
    p2z = patch_to_zone(7)
    assert p2z.shape == (49,)
    assert set(p2z.tolist()) == set(range(6))         # all 6 zones used
    # upper band (rows 0-2) -> zones 0/1; lower band (rows 5-6) -> zones 4/5
    assert set(p2z[:21].tolist()) <= {0, 1}
    assert set(p2z[35:].tolist()) <= {4, 5}


def test_lr_flip_swaps_columns() -> None:
    base = patch_to_zone(7, flip_lr=False)
    flip = patch_to_zone(7, flip_lr=True)
    # flip swaps the L/R column within each band: zone z -> z XOR 1
    assert np.array_equal(flip, base ^ 1)


def test_zone_attention_sums_to_one() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((10, 49)); a /= a.sum(1, keepdims=True)
    za = zone_attention(a, patch_to_zone(7))
    assert za.shape == (10, 6)
    assert np.allclose(za.sum(1), 1.0)


def test_auc_matches_known_values() -> None:
    assert _auc(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 1, 1])) == 1.0
    assert _auc(np.array([0.4, 0.3, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert abs(_auc(np.array([1, 1, 1, 1.0]), np.array([0, 0, 1, 1])) - 0.5) < 1e-9
    assert np.isnan(_auc(np.array([0.1, 0.2]), np.array([1, 1])))  # degenerate


def _toy_labels(n: int = 200, seed: int = 0):
    """Minimal labels frame: each cavity+ image has 1-2 positive zones."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = np.zeros(6, int)
        if rng.random() < 0.6:
            for z in rng.choice(6, size=rng.integers(1, 3), replace=False):
                y[z] = 1
        rows.append({"image_id": f"img{i}", "country": "Test",
                     "cavity": int(y.sum() > 0), **dict(zip(SEXTANT_CODES, y))})
    return pd.DataFrame(rows)


def test_localization_detects_signal_and_is_null_without_it() -> None:
    labels = _toy_labels()
    p2z, share = patch_to_zone(7), zone_size_share(patch_to_zone(7))
    ysext = labels[SEXTANT_CODES].to_numpy(int)

    strong = synth_attn(labels, signal=4.0, seed=1)["attn"]
    za = zone_attention(strong, p2z)
    assert zone_localization(za, ysext, share)["zone_auc_mean"] > 0.9

    nullsig = synth_attn(labels, signal=0.0, seed=1)["attn"]
    za0 = zone_localization(zone_attention(nullsig, p2z), ysext, share)
    assert abs(za0["zone_auc_mean"] - 0.5) < 0.08          # chance


def test_controls_shuffle_and_flip() -> None:
    labels = _toy_labels()
    p2z = patch_to_zone(7)
    share, share_flip = zone_size_share(p2z), zone_size_share(patch_to_zone(7, flip_lr=True))
    ysext = labels[SEXTANT_CODES].to_numpy(int)
    attn = synth_attn(labels, signal=4.0, seed=2)["attn"]

    auc = zone_localization(zone_attention(attn, p2z), ysext, share)["zone_auc_mean"]
    # L/R-flip must drop AUC (signal lives in the correctly-mapped columns)
    auc_flip = zone_localization(
        zone_attention(attn, patch_to_zone(7, flip_lr=True)), ysext, share_flip)["zone_auc_mean"]
    assert auc_flip < auc - 0.2
    # shuffle patches -> near chance
    rng = np.random.default_rng(0)
    shuf = np.take_along_axis(attn, np.argsort(rng.random(attn.shape), 1), 1)
    auc_shuf = zone_localization(zone_attention(shuf, p2z), ysext, share)["zone_auc_mean"]
    assert abs(auc_shuf - 0.5) < 0.1


def test_calibration_perfect_and_ranges() -> None:
    label = np.array([0, 0, 1, 1, 1, 0, 1, 0.0])
    perfect = calibration(label.astype(float), label)
    assert perfect["brier"] == 0.0 and perfect["ece"] == 0.0
    worst = calibration(1.0 - label, label)
    assert worst["brier"] == 1.0 and abs(worst["ece"] - 1.0) < 1e-9


def test_real_sextant_recovery_if_data_present() -> None:
    """Against real CSVs: recovery must agree 100% with the manifest cavity flag."""
    from scripts.ws_c_cavity_grounding import DEF_ANN, DEF_CXRS, DEF_MANIFEST
    if not (DEF_MANIFEST.exists() and DEF_CXRS.exists() and DEF_ANN.exists()):
        print("  [skip] real TB Portals CSVs not present")
        return
    df = load_sextant_labels()
    assert len(df) > 1000
    any_pos = (df[SEXTANT_CODES].sum(axis=1) > 0).astype(int)
    assert float((any_pos == df["cavity"]).mean()) == 1.0
    # apex prior: upper sextants positive more often than lower
    rates = df[SEXTANT_CODES].mean()
    assert rates["UL"] + rates["UR"] > rates["LL"] + rates["LR"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"ERROR {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
