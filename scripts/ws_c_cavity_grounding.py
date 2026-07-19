"""WS-C: Cavity Grounding — measured cavity calibration + zone-level localization.

Turns Fig. 4 from an illustrative panel into a *measured* claim. Two halves:

  D1 (calibration, unconditional)  — ECE / Brier / reliability of the cavity
      probability, global-CLS head vs spatial head. Needs only (cav_prob, label).

  D2 (localization, needs attention) — map the RAD-DINO 7x7 attention grid onto
      the 6 radiologist sextants and score it against the per-sextant cavity
      labels: zone-attention AUC, pointing-game accuracy (vs chance), energy
      concentration. Plus C4 negative controls (shuffle -> 0.5, L/R-flip drops).

Design split (see ../ws-c.md and ../CLAUDE.md):
  * Label recovery + manifest join run **locally** (CSVs are in the repo).
  * The attention vectors come from Kaggle (trained SpatialCavityHead + grid
    cache). This script consumes them via --attn-npz; with --self-test it
    fabricates attention correlated with the real labels to validate the
    pipeline + controls end-to-end with no GPU.

Honesty (non-negotiable): every localization number prints its chance baseline
and a bootstrap 95% CI; claims are zone-level / segmenter-free / approximate.

CLI examples
------------
    # local self-test (no Kaggle artefacts needed): validates recovery + metrics
    python3.11 scripts/ws_c_cavity_grounding.py --self-test

    # real run once Kaggle attention is downloaded
    python3.11 scripts/ws_c_cavity_grounding.py \
        --attn-npz ws_c/attn_spatial.npz \
        --global-npz ws_c/cavprob_global.npz \
        --out-dir ws_c
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``src`` importable when run as a plain script from the repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.tbportals import (  # noqa: E402
    SEXTANT_CODES,
    aggregate_sextant_cavities,
)

HELD_OUT = ["Romania", "Moldova", "Kazakhstan"]
DEF_MANIFEST = _REPO / "notebooks" / "tbportals_manifest_paper.csv"
DEF_CXRS = _REPO / "TB_Portals_CXRs_August_2023.csv"
DEF_ANN = _REPO / "TB_Portals_CXR_Manual_Annotations_August_2023.csv"


# ---------------------------------------------------------------------------
# Label recovery: sextant matrix joined to manifest image_id
# ---------------------------------------------------------------------------

def _basename_uid(url: str) -> str:
    """manifest image_id == basename(series_instance_content_url) sans .dcm."""
    return re.split(r"[\\/]", str(url))[-1].replace(".dcm", "")


def load_sextant_labels(
    manifest_csv: Path = DEF_MANIFEST,
    cxrs_csv: Path = DEF_CXRS,
    ann_csv: Path = DEF_ANN,
    *,
    countries: list[str] | None = None,
) -> pd.DataFrame:
    """Return one row per manifest image with the 6-dim sextant cavity matrix.

    Columns: image_id, country, cavity (image-level flag from manifest), and the
    6 zone columns in ``SEXTANT_CODES``. Only images that actually carry sextant
    annotations are kept (others cannot be scored for localization).
    """
    countries = countries or HELD_OUT
    man = pd.read_csv(manifest_csv, dtype={"image_id": str})
    cxr = pd.read_csv(cxrs_csv, low_memory=False, dtype=str)
    ann = pd.read_csv(ann_csv, low_memory=False)

    cxr = cxr.dropna(subset=["series_instance_content_url"]).copy()
    cxr["image_id"] = cxr["series_instance_content_url"].map(_basename_uid)
    bridge = cxr.set_index("image_id")["imagingstudy_id"].to_dict()

    zone = aggregate_sextant_cavities(ann)  # keyed by imagingstudy_id
    zone = zone.set_index("imagingstudy_id")

    man = man[man["country"].isin(countries)].copy()
    man["study"] = man["image_id"].map(bridge)
    man = man.join(zone, on="study")
    keep = man["study"].isin(zone.index)
    dropped = int((~keep).sum())
    if dropped:
        print(f"[ws-c] dropped {dropped} images with no sextant annotation "
              f"({len(man)} -> {int(keep.sum())} kept)")
    man = man[keep].copy()
    man["cavity"] = man["cavity"].astype(int)

    # Consistency guard: any-zone-positive must equal the manifest cavity flag.
    any_pos = (man[SEXTANT_CODES].sum(axis=1) > 0).astype(int)
    agree = float((any_pos == man["cavity"]).mean())
    print(f"[ws-c] sextant<->manifest cavity agreement: {agree:.4f} (n={len(man)})")
    return man[["image_id", "country", "cavity", *SEXTANT_CODES]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Patch -> zone mapping (7x7 grid -> 6 sextants), with radiological L/R flip
# ---------------------------------------------------------------------------

def patch_to_zone(grid: int = 7, *, flip_lr: bool = False) -> np.ndarray:
    """Lookup [grid*grid] -> zone index 0..5 (SEXTANT_CODES order).

    Rows are split into upper/middle/lower thirds; columns into two halves.
    CXRs face the patient, so the **image-left** column is the **patient's
    right** lung. SEXTANT codes are patient-anatomy, so image-left -> *Right*
    sextants. ``flip_lr=True`` deliberately inverts this (C4 negative control).
    """
    p = np.arange(grid * grid)
    row, col = p // grid, p % grid
    band = (row * 3) // grid                 # 0 upper, 1 middle, 2 lower
    img_right = (col * 2) // grid            # 0 = image-left, 1 = image-right
    # patient-left lung is on image-right; col_idx 0 = Left code, 1 = Right code.
    patient_left = (img_right == 1)
    col_idx = np.where(patient_left, 0, 1)   # 0 -> *L sextant, 1 -> *R sextant
    if flip_lr:
        col_idx = 1 - col_idx
    return (band * 2 + col_idx).astype(int)


def zone_attention(attn: np.ndarray, p2z: np.ndarray, n_zones: int = 6) -> np.ndarray:
    """Sum patch attention [N, P] into zone attention [N, n_zones] (rows sum to 1)."""
    out = np.zeros((attn.shape[0], n_zones), dtype=np.float64)
    for z in range(n_zones):
        out[:, z] = attn[:, p2z == z].sum(axis=1)
    return out


def zone_size_share(p2z: np.ndarray, n_zones: int = 6) -> np.ndarray:
    """Fraction of patches assigned to each zone (zones are unequal in a 7x7 grid)."""
    P = len(p2z)
    return np.array([(p2z == z).sum() / P for z in range(n_zones)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via Mann-Whitney U (ties -> 0.5). NaN if degenerate."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def zone_localization(zattn: np.ndarray, ysext: np.ndarray,
                      size_share: np.ndarray | None = None) -> dict:
    """Per-image zone-AUC, pointing game, energy concentration (cavity+ images).

    Zones are unequal in size on a 7x7 grid (and larger zones — the apex — are
    also more often positive), so raw zone mass biases AUC/pointing upward and
    the shuffle control would not be a clean null. We therefore rank zones by
    **attention density** = zone mass / zone patch-share; a uniform attention
    then scores every zone equally (AUC -> 0.5). ``energy`` uses raw mass and is
    already normalised against the uniform expectation.

    Returns per-image arrays plus the chance baselines, so the caller can both
    report means and bootstrap them.
    """
    pos_img = ysext.sum(axis=1) > 0
    z, y = zattn[pos_img], ysext[pos_img]
    n_zones = ysext.shape[1]
    if size_share is None:
        size_share = np.full(n_zones, 1.0 / n_zones)
    dens = z / np.maximum(size_share[None, :], 1e-12)   # size-corrected scores

    aucs, hits, energy = [], [], []
    for di, zi, yi in zip(dens, z, y):
        npos = int(yi.sum())
        if 0 < npos < n_zones:                       # AUC needs both classes
            aucs.append(_auc(di, yi))
        hits.append(float(yi[int(np.argmax(di))]))   # pointing game (density)
        energy.append(float(zi[yi == 1].sum() / (npos / n_zones)))  # raw mass vs uniform

    pg_chance = float((y.sum(axis=1) / n_zones).mean())  # E[#pos/6]
    return {
        "n_cavpos_images": int(pos_img.sum()),
        "zone_auc": np.array(aucs, dtype=np.float64),
        "zone_auc_mean": float(np.nanmean(aucs)) if aucs else float("nan"),
        "pointing": np.array(hits, dtype=np.float64),
        "pointing_acc": float(np.mean(hits)) if hits else float("nan"),
        "pointing_chance": pg_chance,
        "energy": np.array(energy, dtype=np.float64),
        "energy_mean": float(np.mean(energy)) if energy else float("nan"),
    }


def calibration(prob: np.ndarray, label: np.ndarray, *, n_bins: int = 15) -> dict:
    """Brier, ECE (equal-width bins), and reliability curve."""
    prob = np.asarray(prob, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    brier = float(np.mean((prob - label) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, centers, freqs, confs, counts = 0.0, [], [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (prob >= lo) & (prob < hi if b < n_bins - 1 else prob <= hi)
        n = int(m.sum())
        centers.append((lo + hi) / 2.0)
        if n:
            acc, conf = float(label[m].mean()), float(prob[m].mean())
            ece += (n / len(prob)) * abs(acc - conf)
            freqs.append(acc); confs.append(conf); counts.append(n)
        else:
            freqs.append(float("nan")); confs.append(float("nan")); counts.append(0)
    return {"brier": brier, "ece": float(ece), "bin_centers": centers,
            "bin_freq": freqs, "bin_conf": confs, "bin_count": counts}


def bootstrap_mean_ci(values: np.ndarray, *, n_boot: int = 2000, seed: int = 1337) -> tuple:
    """Percentile 95% CI of the mean of per-image values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ---------------------------------------------------------------------------
# Attention input: real npz from Kaggle, or synthetic for --self-test
# ---------------------------------------------------------------------------

def load_attn_npz(path: Path, *, require_attn: bool = True) -> dict:
    """Load a Kaggle export.

    Spatial head (``--attn-npz``): image_id [N], attn [N, P], optionally cav_prob [N].
    Global head (``--global-npz``): image_id [N], cav_prob [N] — no attn needed.
    """
    d = np.load(path, allow_pickle=True)
    out = {"image_id": np.asarray(d["image_id"]).astype(str)}
    if "cav_prob" in d:
        out["cav_prob"] = np.asarray(d["cav_prob"], dtype=np.float64)
    if "attn" in d.files:
        attn = np.asarray(d["attn"], dtype=np.float64)
        s = attn.sum(axis=1, keepdims=True)          # renormalise defensively
        out["attn"] = attn / np.maximum(s, 1e-8)
    elif require_attn:
        raise KeyError(f"{path} has no 'attn' array (keys: {list(d.files)})")
    return out


def synth_attn(labels: pd.DataFrame, grid: int = 7, *, seed: int = 0,
               signal: float = 3.0) -> dict:
    """Fabricate attention that concentrates (noisily) on cavity-positive zones.

    Used only by --self-test to exercise recovery + metrics + controls without
    GPU. Higher ``signal`` -> stronger localization (zone-AUC well above 0.5);
    setting it to 0 should collapse AUC to chance, which the test can assert.
    """
    rng = np.random.default_rng(seed)
    p2z = patch_to_zone(grid)                       # correct mapping
    P = grid * grid
    ysext = labels[SEXTANT_CODES].to_numpy(int)
    logits = rng.normal(0, 1.0, size=(len(labels), P))
    for i in range(len(labels)):
        for z in range(6):
            if ysext[i, z]:
                logits[i, p2z == z] += signal
    attn = np.exp(logits - logits.max(axis=1, keepdims=True))
    attn /= attn.sum(axis=1, keepdims=True)
    # cavity prob: noisy-but-calibrated-ish around the label
    base = 0.15 + 0.7 * labels["cavity"].to_numpy(float)
    prob = np.clip(base + rng.normal(0, 0.12, len(labels)), 0.01, 0.99)
    return {"image_id": labels["image_id"].to_numpy(str), "attn": attn, "cav_prob": prob}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(labels: pd.DataFrame, attn: dict, *, grid: int = 7,
        global_prob: dict | None = None) -> dict:
    """Compute D1 + D2 + C4 per held-out country. Returns a JSON-able dict."""
    # align attention rows to label rows by image_id
    aidx = {iid: k for k, iid in enumerate(attn["image_id"])}
    labels = labels[labels["image_id"].isin(aidx)].reset_index(drop=True)
    rows = np.array([aidx[i] for i in labels["image_id"]])
    A = attn["attn"][rows]
    cav_prob = attn.get("cav_prob")
    cav_prob = cav_prob[rows] if cav_prob is not None else None

    p2z = patch_to_zone(grid)
    p2z_flip = patch_to_zone(grid, flip_lr=True)
    share = zone_size_share(p2z)
    share_flip = zone_size_share(p2z_flip)
    rng = np.random.default_rng(0)
    A_shuf = np.take_along_axis(
        A, np.argsort(rng.random(A.shape), axis=1), axis=1)  # shuffle patches per row

    report: dict = {"grid": grid, "n_total": int(len(labels)), "countries": {}}
    for ctry in sorted(labels["country"].unique()):
        m = labels["country"] == ctry
        ysext = labels.loc[m, SEXTANT_CODES].to_numpy(int)
        za = zone_attention(A[m.to_numpy()], p2z)
        loc = zone_localization(za, ysext, share)

        # C4 controls
        za_flip = zone_attention(A[m.to_numpy()], p2z_flip)
        za_shuf = zone_attention(A_shuf[m.to_numpy()], p2z)
        auc_flip = zone_localization(za_flip, ysext, share_flip)["zone_auc_mean"]
        auc_shuf = zone_localization(za_shuf, ysext, share)["zone_auc_mean"]

        lo, hi = bootstrap_mean_ci(loc["zone_auc"])
        plo, phi = bootstrap_mean_ci(loc["pointing"])
        entry = {
            "n_images": int(m.sum()),
            "n_cavpos_images": loc["n_cavpos_images"],
            "zone_auc": round(loc["zone_auc_mean"], 4),
            "zone_auc_ci95": [round(lo, 4), round(hi, 4)],
            "pointing_acc": round(loc["pointing_acc"], 4),
            "pointing_acc_ci95": [round(plo, 4), round(phi, 4)],
            "pointing_chance": round(loc["pointing_chance"], 4),
            "energy_concentration": round(loc["energy_mean"], 4),
            "control_zone_auc_lr_flip": round(auc_flip, 4),
            "control_zone_auc_shuffle": round(auc_shuf, 4),
            "per_zone_pos_rate": {c: round(float(ysext[:, k].mean()), 4)
                                  for k, c in enumerate(SEXTANT_CODES)},
        }
        # D1 calibration (spatial head, and global head if supplied)
        if cav_prob is not None:
            entry["calibration_spatial"] = {
                k: v for k, v in calibration(cav_prob[m.to_numpy()],
                                             labels.loc[m, "cavity"].to_numpy()).items()}
        if global_prob is not None:
            gidx = {iid: k for k, iid in enumerate(global_prob["image_id"])}
            gm = labels[m & labels["image_id"].isin(gidx)]
            if len(gm):
                gp = global_prob["cav_prob"][[gidx[i] for i in gm["image_id"]]]
                entry["calibration_global"] = {
                    k: v for k, v in calibration(gp, gm["cavity"].to_numpy()).items()}
        report["countries"][ctry] = entry
    return report


def _write_outputs(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cavity_grounding_metrics.json").write_text(json.dumps(report, indent=2))
    flat = []
    for ctry, e in report["countries"].items():
        flat.append({"country": ctry,
                     **{k: v for k, v in e.items()
                        if not isinstance(v, dict) and not isinstance(v, list)},
                     "zone_auc_ci_lo": e["zone_auc_ci95"][0],
                     "zone_auc_ci_hi": e["zone_auc_ci95"][1]})
    pd.DataFrame(flat).to_csv(out_dir / "cavity_grounding_metrics.csv", index=False)
    print(f"[ws-c] wrote {out_dir/'cavity_grounding_metrics.json'} (+ .csv)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WS-C cavity grounding")
    ap.add_argument("--manifest", type=Path, default=DEF_MANIFEST)
    ap.add_argument("--cxrs-csv", type=Path, default=DEF_CXRS)
    ap.add_argument("--ann-csv", type=Path, default=DEF_ANN)
    ap.add_argument("--attn-npz", type=Path, default=None,
                    help="Kaggle attention export (image_id, attn[,cav_prob])")
    ap.add_argument("--global-npz", type=Path, default=None,
                    help="global-CLS head cav_prob for the D1 comparison")
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--countries", nargs="+", default=HELD_OUT)
    ap.add_argument("--out-dir", type=Path, default=_REPO / "ws_c")
    ap.add_argument("--self-test", action="store_true",
                    help="fabricate attention from real labels (no Kaggle needed)")
    ap.add_argument("--self-test-signal", type=float, default=3.0)
    args = ap.parse_args(argv)

    labels = load_sextant_labels(args.manifest, args.cxrs_csv, args.ann_csv,
                                 countries=args.countries)
    if args.self_test:
        attn = synth_attn(labels, grid=args.grid, signal=args.self_test_signal)
        glob = None
    elif args.attn_npz:
        attn = load_attn_npz(args.attn_npz)
        glob = load_attn_npz(args.global_npz, require_attn=False) if args.global_npz else None
    else:
        ap.error("provide --attn-npz (real run) or --self-test")

    report = run(labels, attn, grid=args.grid, global_prob=glob)
    _write_outputs(report, args.out_dir)

    print("\n=== WS-C summary (zone-level, segmenter-free, approximate) ===")
    for ctry, e in report["countries"].items():
        print(f"{ctry:11s} n={e['n_images']:>4d} cav+={e['n_cavpos_images']:>4d}  "
              f"zoneAUC={e['zone_auc']:.3f} CI{e['zone_auc_ci95']}  "
              f"point={e['pointing_acc']:.3f}(chance {e['pointing_chance']:.3f})  "
              f"flip={e['control_zone_auc_lr_flip']:.3f} shuf={e['control_zone_auc_shuffle']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
