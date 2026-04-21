"""scripts/calibrate_component7_thresholds.py

Grid-sweep calibration for Component 7 refinement thresholds.

For every combination of:
    weak_boundary_threshold  in [0.3, 0.4, 0.5, 0.6]
    caution_fp_threshold     in [0.55, 0.65, 0.75]
    suppress_fp_threshold    in [0.75, 0.85, 0.95]

the script loads cached predictions (npz or pkl) and ground-truth masks,
calls refine_mask(), and computes pixel-level precision, recall, and F1.

Results are written to:
    outputs/component7_calibration/results.csv
    outputs/component7_calibration/f1_heatmap.png

Usage
-----
python scripts/calibrate_component7_thresholds.py \\
    --predictions-dir cache/component7/predictions \\
    --gt-dir          cache/component7/gt_masks \\
    [--output-dir     outputs/component7_calibration]
    [--boundary-score 0.45]
    [--fp-prob        0.40]

Notes
-----
- Runs fully on CPU — no GPU or model inference required.
- All path arguments are relative to CWD or absolute; no paths are hardcoded.
- Predictions files must be .npz (key "mask") or .pkl containing a NumPy array
  shaped (H, W) or (1, H, W) with values in [0, 1].
- GT mask files must be .npz (key "mask") or .pkl, same shape, binary.
- Files are matched by stem name (prediction stem == gt stem).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_mask(path: Path) -> np.ndarray:
    """Load a mask array from .npz or .pkl.  Returns float32 (H, W)."""
    if path.suffix == ".npz":
        data = np.load(path)
        arr = data["mask"]
    elif path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as fh:
            arr = pickle.load(fh)
    else:
        raise ValueError(f"Unsupported mask file format: {path.suffix!r}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {arr.shape} for {path}")
    return arr


def _iter_pairs(
    pred_dir: Path, gt_dir: Path
) -> Iterator[tuple[Path, Path]]:
    """Yield (pred_path, gt_path) for every file stem that appears in both dirs."""
    pred_stems = {p.stem: p for p in pred_dir.iterdir() if p.suffix in {".npz", ".pkl", ".pickle"}}
    gt_stems   = {p.stem: p for p in gt_dir.iterdir()  if p.suffix in {".npz", ".pkl", ".pickle"}}
    common = sorted(pred_stems.keys() & gt_stems.keys())
    if not common:
        raise FileNotFoundError(
            f"No matching stem names between predictions ({pred_dir}) and GT ({gt_dir})."
        )
    for stem in common:
        yield pred_stems[stem], gt_stems[stem]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _pixel_stats(
    pred_bin: np.ndarray, gt_bin: np.ndarray
) -> tuple[float, float, float]:
    """Return (precision, recall, F1) for two binary masks."""
    tp = float((pred_bin & gt_bin).sum())
    fp = float((pred_bin & ~gt_bin).sum())
    fn = float((~pred_bin & gt_bin).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2.0 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Main grid sweep
# ---------------------------------------------------------------------------

WEAK_BOUNDARY_VALUES   = [0.3, 0.4, 0.5, 0.6]
CAUTION_FP_VALUES      = [0.55, 0.65, 0.75]
SUPPRESS_FP_VALUES     = [0.75, 0.85, 0.95]


def run_calibration(
    pred_dir: Path,
    gt_dir: Path,
    *,
    output_dir: Path,
    boundary_score: float,
    fp_prob: float,
) -> None:
    """Run the full grid sweep and write results CSV + heatmap."""
    from src.components.component7_refine import BaselineRefineConfig, refine_mask

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(_iter_pairs(pred_dir, gt_dir))
    print(f"Found {len(pairs)} matched prediction/GT pairs.")

    # Pre-load all masks once
    pred_tensors: list[torch.Tensor] = []
    gt_arrays:    list[np.ndarray]   = []
    for pred_path, gt_path in pairs:
        pred_arr = _load_mask(pred_path)
        gt_arr   = (_load_mask(gt_path) > 0.5)
        # refine_mask expects a lung mask — use an all-ones mask (conservative: no spill penalty)
        pred_tensors.append(torch.from_numpy(pred_arr))
        gt_arrays.append(gt_arr.astype(bool))

    lung_mask = torch.ones(1, 256, 256)  # dummy full-lung mask for calibration

    grid = list(itertools.product(WEAK_BOUNDARY_VALUES, CAUTION_FP_VALUES, SUPPRESS_FP_VALUES))
    results: list[dict] = []

    print(
        f"Sweeping {len(grid)} threshold combinations "
        f"over {len(pairs)} samples …"
    )

    for wb, cfp, sfp in grid:
        if cfp >= sfp:
            # Physically invalid: caution must be below suppress
            continue

        cfg = BaselineRefineConfig(
            min_area_px=48,
            opening_iters=1,
            closing_iters=1,
            weak_boundary_threshold=wb,
            caution_fp_threshold=cfp,
            suppress_fp_threshold=sfp,
        )

        all_prec, all_rec, all_f1 = [], [], []
        for pred_t, gt_bin in zip(pred_tensors, gt_arrays, strict=True):
            # Resize pred to 256×256 if needed
            if pred_t.shape != (256, 256):
                pred_t = torch.nn.functional.interpolate(
                    pred_t.unsqueeze(0).unsqueeze(0).float(),
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            # Resize gt to 256×256 if needed
            if gt_bin.shape != (256, 256):
                gt_t = torch.from_numpy(gt_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                gt_t = torch.nn.functional.interpolate(gt_t, size=(256, 256), mode="nearest").squeeze()
                gt_bin_256 = (gt_t.numpy() > 0.5)
            else:
                gt_bin_256 = gt_bin

            # Build dummy x_1024 (not used by refine_mask, just passed through)
            x_dummy = torch.zeros(1, 1024, 1024)

            refined = refine_mask(
                pred_t.unsqueeze(0),  # [1, H, W]
                lung_mask[0],         # [H, W] → refine_mask handles ndim check
                x_dummy,
                boundary_score,
                fp_prob,
                cfg,
            )
            refined_bin = (refined.squeeze(0).numpy() > 0.5)
            # Resize gt to match refined if needed
            p, r, f = _pixel_stats(refined_bin, gt_bin_256)
            all_prec.append(p)
            all_rec.append(r)
            all_f1.append(f)

        mean_p  = float(np.mean(all_prec))
        mean_r  = float(np.mean(all_rec))
        mean_f1 = float(np.mean(all_f1))
        results.append(
            {
                "weak_boundary_threshold": wb,
                "caution_fp_threshold":    cfp,
                "suppress_fp_threshold":   sfp,
                "precision":               round(mean_p, 4),
                "recall":                  round(mean_r, 4),
                "f1":                      round(mean_f1, 4),
            }
        )

    # --- write CSV ---
    csv_path = output_dir / "results.csv"
    fieldnames = ["weak_boundary_threshold", "caution_fp_threshold",
                  "suppress_fp_threshold", "precision", "recall", "f1"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to: {csv_path}")

    # --- write heatmap (F1 vs wb × cfp, averaged over sfp) ---
    _write_heatmap(results, output_dir)


def _write_heatmap(results: list[dict], output_dir: Path) -> None:
    """Write a 2-D heatmap PNG showing mean F1 for each (wb, cfp) pair."""
    import matplotlib.pyplot as plt

    wb_vals  = sorted({r["weak_boundary_threshold"] for r in results})
    cfp_vals = sorted({r["caution_fp_threshold"] for r in results})

    # Average F1 over suppress_fp_threshold for each (wb, cfp) cell
    grid = np.zeros((len(cfp_vals), len(wb_vals)))
    counts = np.zeros_like(grid, dtype=int)
    for r in results:
        i = cfp_vals.index(r["caution_fp_threshold"])
        j = wb_vals.index(r["weak_boundary_threshold"])
        grid[i, j] += r["f1"]
        counts[i, j] += 1
    safe_counts = np.where(counts > 0, counts, 1)
    grid /= safe_counts

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(wb_vals)))
    ax.set_xticklabels([str(v) for v in wb_vals])
    ax.set_yticks(range(len(cfp_vals)))
    ax.set_yticklabels([str(v) for v in cfp_vals])
    ax.set_xlabel("weak_boundary_threshold")
    ax.set_ylabel("caution_fp_threshold")
    ax.set_title("Mean F1 (averaged over suppress_fp_threshold)")
    for i in range(len(cfp_vals)):
        for j in range(len(wb_vals)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    heatmap_path = output_dir / "f1_heatmap.png"
    fig.savefig(heatmap_path, dpi=100)
    plt.close(fig)
    print(f"Heatmap written to: {heatmap_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-sweep calibration for Component 7 refinement thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory containing cached prediction masks (.npz or .pkl, key 'mask').",
    )
    parser.add_argument(
        "--gt-dir",
        required=True,
        help="Directory containing ground-truth masks (.npz or .pkl, key 'mask').",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/component7_calibration",
        help="Directory to write results.csv and f1_heatmap.png.",
    )
    parser.add_argument(
        "--boundary-score",
        type=float,
        default=0.45,
        help="Fixed boundary_score to use for all refine_mask calls (simulates median case).",
    )
    parser.add_argument(
        "--fp-prob",
        type=float,
        default=0.40,
        help="Fixed fp_prob to use for all refine_mask calls (simulates median case).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_calibration(
        pred_dir=Path(args.predictions_dir),
        gt_dir=Path(args.gt_dir),
        output_dir=Path(args.output_dir),
        boundary_score=args.boundary_score,
        fp_prob=args.fp_prob,
    )
