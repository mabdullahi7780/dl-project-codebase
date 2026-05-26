"""Strict replication of Kantipudi et al. (JIIM 2024), approach A3.

A3 directly regresses the Timika score with ONE DenseNet121. The paper passes
the output through a sigmoid and multiplies by 140 to span the Timika range
[0, 140], trained with MSE. Input = lung-cropped 224x224, ImageNet norm; the
country-segregated split is identical to the ALP regressor (paper Table 3:
all non-held-out images, 80:20 patient-disjoint).

Test = every image of the held-out country. Unlike A2 there is no separate
cavity model — Timika is predicted end-to-end — so only Timika regression
metrics are reported (vs Kantipudi Table 7, approach A3).

Example::

    python -m src.training.train_a3_direct \
        --manifest local_work/data/processed/tbportals_manifest_paper.csv \
        --crops-dir local_work/data/processed/tbportals_crops \
        --out-dir   checkpoints/paper_a3 \
        --held-outs Romania Moldova Kazakhstan --seeds 0 1 2 \
        --batch-size 60 --accum-steps 5
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from src.components.baseline_paper import TimikaRegressor
from src.data.tbportals import load_manifest, make_country_split
from src.evaluation.eval_tbportals import (
    KANTIPUDI_A3,
    _safe_pearson,
    bootstrap_ci,
    regression_metrics,
)
from src.training.train_baseline_paper import (
    _loader,
    _make_dataset,
    pick_device,
    seed_everything,
)

TIMIKA_MAX = 140.0


def _timika_true_frac(batch) -> torch.Tensor:
    """Timika = ALP%(=alp*100) + 40*cavity, normalised to [0,1] (÷140)."""
    return (batch["alp"] * 100.0 + 40.0 * batch["cavity"]) / TIMIKA_MAX


def train_timika(train_df, val_df, args, device, crops_dir):
    train_loader = _loader(_make_dataset(train_df, True, crops_dir), args.batch_size, True, args.num_workers)
    val_loader = _loader(_make_dataset(val_df, False, crops_dir), args.batch_size, False, args.num_workers)

    model = TimikaRegressor(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=args.lr)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    accum = max(1, args.accum_steps)
    n_batches = len(train_loader)
    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        run = 0.0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(train_loader):
            x = b["image"].to(device, non_blocking=True)
            y = _timika_true_frac(b).to(device)
            with autocast("cuda", enabled=use_amp):
                loss = F.mse_loss(model(x), y)
            scaler.scale(loss / accum).backward()
            run += float(loss.detach()) * x.size(0)
            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        model.eval()
        vsum, n = 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                x = b["image"].to(device)
                y = _timika_true_frac(b).to(device)
                with autocast("cuda", enabled=use_amp):
                    vsum += float(F.mse_loss(model(x), y, reduction="sum"))
                n += x.size(0)
        vmse = vsum / max(1, n)
        print(f"  [A3] epoch {epoch:02d} train_mse={run/len(train_loader.dataset):.6f} val_mse={vmse:.6f}")
        if vmse < best_val:
            best_val = vmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def predict_timika(model, test_df, args, device, crops_dir):
    loader = _loader(_make_dataset(test_df, False, crops_dir), args.batch_size, False, args.num_workers)
    model.eval()
    tt, tp = [], []
    for b in loader:
        x = b["image"].to(device)
        tp.append(model(x).cpu().numpy() * TIMIKA_MAX)
        tt.append(_timika_true_frac(b).numpy() * TIMIKA_MAX)
    return np.concatenate(tt), np.concatenate(tp)


def run_country(df, held_out, seed, args, device, crops_dir, out_dir):
    seed_everything(seed)
    print(f"\n===== A3  {held_out}  seed={seed} =====")
    train_df, val_df, test_df = make_country_split(
        df, held_out, val_fraction=args.val_fraction, seed=seed
    )
    print(f"[A3] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    model, best = train_timika(train_df, val_df, args, device, crops_dir)

    torch.save(model.state_dict(), out_dir / f"timika_a3_{held_out}_seed{seed}.pt")
    t_true, t_pred = predict_timika(model, test_df, args, device, crops_dir)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    m = regression_metrics(t_true, t_pred)
    mae_fn = lambda a, b: float(np.mean(np.abs(b - a)))
    mae_ci = bootstrap_ci(t_true, t_pred, mae_fn)
    pear_ci = bootstrap_ci(t_true, t_pred, _safe_pearson)
    ref = KANTIPUDI_A3.get(held_out, {})

    row = {
        "approach": "A3", "held_out": held_out, "seed": seed, "n_test": len(test_df),
        "timika_mae": m["mae"], "timika_mae_pct": m["mae_pct"], "timika_pearson": m["pearson"],
        "timika_rmse": m["rmse"], "timika_spearman": m["spearman"], "best_val_mse": best,
        "timika_mae_ci_lo": mae_ci[0], "timika_mae_ci_hi": mae_ci[1],
        "timika_pearson_ci_lo": pear_ci[0], "timika_pearson_ci_hi": pear_ci[1],
    }
    print(f"[RESULT] A3 {held_out} seed={seed}  "
          f"Timika_MAE={m['mae']:.2f} ({m['mae_pct']:.2f}%) | their {ref.get('timika_mae', float('nan')):.2f}   "
          f"Pearson={m['pearson']:.2f} | their {ref.get('timika_pearson', float('nan')):.2f}   "
          f"CI95_MAE=[{mae_ci[0]:.2f},{mae_ci[1]:.2f}]")
    return row


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Strict Kantipudi A3 replication (direct Timika).")
    p.add_argument("--manifest", required=True)
    p.add_argument("--crops-dir", default=None,
                   help="Lung-crop directory (paper crops the lung region for A3 too).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--held-outs", nargs="+", default=["Romania", "Moldova", "Kazakhstan"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=300,
                   help="Physical micro-batch. On a 16GB T4 use 60 with --accum-steps 5.")
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-lung-crop", action="store_true",
                   help="Train on whole images instead of lung crops (ablation; paper crops).")
    p.add_argument("--amp", action="store_true", default=True)
    args = p.parse_args(argv)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = pick_device()
    print(f"[paper-a3] device={device}")
    df = load_manifest(args.manifest)

    crops_dir = None if args.no_lung_crop else args.crops_dir
    print(f"[paper-a3] input = {('lung-crop ' + str(crops_dir)) if crops_dir else 'WHOLE image (ablation)'}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for held_out in args.held_outs:
        for seed in args.seeds:
            rows.append(run_country(df, held_out, seed, args, device, crops_dir, out_dir))
            pd.DataFrame(rows).to_csv(out_dir / "results_a3.csv", index=False)

    dfr = pd.DataFrame(rows)
    print("\n[paper-a3] mean +/- std across seeds (vs Kantipudi A3):")
    for ho in args.held_outs:
        s = dfr[dfr["held_out"] == ho]
        if len(s) == 0:
            continue
        ref = KANTIPUDI_A3.get(ho, {})
        print(f"  {ho:12s} Timika_MAE={s['timika_mae'].mean():5.2f}+/-{s['timika_mae'].std():.2f} "
              f"(their {ref.get('timika_mae', float('nan')):.2f})  "
              f"Pearson={s['timika_pearson'].mean():.2f}+/-{s['timika_pearson'].std():.2f} "
              f"(their {ref.get('timika_pearson', float('nan')):.2f})")
    print(f"\n[paper-a3] results -> {out_dir / 'results_a3.csv'}")


if __name__ == "__main__":
    main()
