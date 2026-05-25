"""Strict replication of Kantipudi et al. (JIIM 2024), approach A2.

Two SEPARATE DenseNet121 models are trained per held-out country:

  ALP regressor      MSE loss, sigmoid output in [0,1], 80:20 patient-split of
                     all non-held-out images (paper Table 3).
  Cavity classifier  cross-entropy, BALANCED training set = all non-held-out
                     cavity images + an equal number of randomly-sampled
                     no-cavity images, 80:20 patient-split (paper Table 2).

Test set = every image of the held-out country. The Timika score is assembled
as ``ALP*100 + 40*cavity_pred`` and evaluated against the paper's Table 6/7.

Paper hyperparameters (matched exactly):
  backbone   DenseNet121, ImageNet init
  optimiser  NAdam, lr 1e-3
  epochs     30
  batch      300
  input      lung-cropped 224x224, ImageNet normalisation
  augment    rot +/-15deg (p=0.5), hflip (p=0.5), zoom 10% (p=0.5)
  selection  best validation loss

Example::

    python -m src.training.train_baseline_paper \
        --manifest local_work/data/processed/tbportals_manifest_paper.csv \
        --crops-dir local_work/data/processed/tbportals_crops \
        --out-dir   checkpoints/paper_baseline \
        --held-outs Romania Moldova Kazakhstan --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.components.baseline_paper import ALPRegressor, CavityClassifier
from src.data.tbportals import (
    assert_no_patient_leakage,
    load_manifest,
    make_country_split,
)
from src.data.tbportals_dataset import TBPortalsDataset
from src.evaluation.eval_tbportals import (
    Predictions,
    evaluate_split,
    print_comparison_table,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Cavity balanced split (paper Table 2 protocol) ────────────────────────────

def make_balanced_cavity_split(
    df: pd.DataFrame,
    held_out: str,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test = all held-out images. Train/val = all non-held cavity+ plus an
    equal number of random no-cavity images, split 80:20 by patient."""
    test_df = df[df["country"] == held_out].reset_index(drop=True)
    pool = df[df["country"] != held_out]
    cav = pool[pool["cavity"] == 1]
    nocav = pool[pool["cavity"] == 0]

    n = min(len(cav), len(nocav))
    keep_cav = cav if len(cav) == n else cav.sample(n=n, random_state=seed)
    keep_nocav = nocav if len(nocav) == n else nocav.sample(n=n, random_state=seed)
    balanced = pd.concat([keep_cav, keep_nocav]).reset_index(drop=True)

    patients = np.asarray(balanced["patient_id"].unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * val_fraction)))
    val_patients = set(patients[:n_val])

    val_df = balanced[balanced["patient_id"].isin(val_patients)].reset_index(drop=True)
    train_df = balanced[~balanced["patient_id"].isin(val_patients)].reset_index(drop=True)
    assert_no_patient_leakage(train_df, val_df, test_df)
    return train_df, val_df, test_df


# ── Loaders ───────────────────────────────────────────────────────────────────

def _loader(ds, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )


def _make_dataset(df, train, crops_dir):
    return TBPortalsDataset(
        df, train=train, crops_dir=crops_dir, use_lung_crop=bool(crops_dir),
    )


# ── ALP regressor training ────────────────────────────────────────────────────

def train_alp(train_df, val_df, args, device, crops_dir):
    train_loader = _loader(_make_dataset(train_df, True, crops_dir), args.batch_size, True, args.num_workers)
    val_loader = _loader(_make_dataset(val_df, False, crops_dir), args.batch_size, False, args.num_workers)

    model = ALPRegressor(pretrained=not args.no_pretrained).to(device)
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
            y = b["alp"].to(device)
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
                y = b["alp"].to(device)
                with autocast("cuda", enabled=use_amp):
                    vsum += float(F.mse_loss(model(x), y, reduction="sum"))
                n += x.size(0)
        vmse = vsum / max(1, n)
        print(f"  [ALP] epoch {epoch:02d} train_mse={run/len(train_loader.dataset):.5f} val_mse={vmse:.5f}")
        if vmse < best_val:
            best_val = vmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val


# ── Cavity classifier training ────────────────────────────────────────────────

def train_cavity(train_df, val_df, args, device, crops_dir):
    train_loader = _loader(_make_dataset(train_df, True, crops_dir), args.batch_size, True, args.num_workers)
    val_loader = _loader(_make_dataset(val_df, False, crops_dir), args.batch_size, False, args.num_workers)

    model = CavityClassifier(pretrained=not args.no_pretrained).to(device)
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
            y = b["cavity"].long().to(device)
            with autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
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
                y = b["cavity"].long().to(device)
                with autocast("cuda", enabled=use_amp):
                    vsum += float(F.cross_entropy(model(x), y, reduction="sum"))
                n += x.size(0)
        vce = vsum / max(1, n)
        print(f"  [CAV] epoch {epoch:02d} train_ce={run/len(train_loader.dataset):.5f} val_ce={vce:.5f}")
        if vce < best_val:
            best_val = vce
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val


# ── Combined prediction (separate models -> Timika) ───────────────────────────

@torch.no_grad()
def predict_combined(alp_model, cav_model, test_df, args, device, crops_dir) -> Predictions:
    loader = _loader(_make_dataset(test_df, False, crops_dir), args.batch_size, False, args.num_workers)
    alp_model.eval()
    cav_model.eval()
    at, ap, ct, cp = [], [], [], []
    for b in loader:
        x = b["image"].to(device)
        ap.append(alp_model(x).cpu().numpy() * 100.0)
        cp.append(torch.softmax(cav_model(x), dim=1)[:, 1].cpu().numpy())
        at.append(b["alp"].numpy() * 100.0)
        ct.append(b["cavity"].numpy())
    return Predictions(
        alp_true_100=np.concatenate(at),
        alp_pred_100=np.concatenate(ap),
        cavity_true=np.concatenate(ct),
        cavity_prob=np.concatenate(cp),
    )


# ── One (held-out country, seed) run ──────────────────────────────────────────

def run_country(df, held_out, seed, args, device, crops_dir, out_dir):
    seed_everything(seed)
    print(f"\n===== {held_out}  seed={seed} =====")

    alp_train, alp_val, test_df = make_country_split(
        df, held_out, val_fraction=args.val_fraction, seed=seed
    )
    print(f"[ALP] train={len(alp_train)} val={len(alp_val)} test={len(test_df)}")
    alp_model, alp_best = train_alp(alp_train, alp_val, args, device, crops_dir)

    cav_train, cav_val, _ = make_balanced_cavity_split(
        df, held_out, val_fraction=args.val_fraction, seed=seed
    )
    n_cav = int(cav_train["cavity"].sum()) + int(cav_val["cavity"].sum())
    print(f"[CAV] train={len(cav_train)} val={len(cav_val)} (balanced, cavity+={n_cav})")
    cav_model, cav_best = train_cavity(cav_train, cav_val, args, device, crops_dir)

    tag = f"{held_out}_seed{seed}"
    torch.save(alp_model.state_dict(), out_dir / f"alp_{tag}.pt")
    torch.save(cav_model.state_dict(), out_dir / f"cavity_{tag}.pt")

    preds = predict_combined(alp_model, cav_model, test_df, args, device, crops_dir)
    res = evaluate_split(preds, cavity_threshold=args.cavity_threshold)

    row = {
        "held_out": held_out, "seed": seed, "n_test": len(test_df),
        "alp_mae": res["alp"]["mae"],
        "cavity_auc": res["cavity"]["auc"],
        "cavity_f1": res["cavity"]["f1"],
        "cavity_precision": res["cavity"]["precision"],
        "cavity_recall": res["cavity"]["recall"],
        "timika_mae": res["timika"]["mae"],
        "timika_mae_pct": res["timika"]["mae_pct"],
        "timika_pearson": res["timika"]["pearson"],
        "alp_best_val_mse": alp_best,
        "cav_best_val_ce": cav_best,
    }
    print(f"[RESULT] {held_out} seed={seed}  "
          f"ALP_MAE={row['alp_mae']:.2f}  cavity_AUC={row['cavity_auc']:.3f}  "
          f"Timika_MAE={row['timika_mae']:.2f} ({row['timika_mae_pct']:.2f}%)  "
          f"Pearson={row['timika_pearson']:.2f}")
    return row, res


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Strict Kantipudi A2 replication.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--crops-dir", default=None,
                   help="Lung-crop directory. The paper trains on lung-cropped images; "
                        "omit only for a quick whole-image smoke test.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--held-outs", nargs="+", default=["Romania", "Moldova", "Kazakhstan"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=300,
                   help="Physical micro-batch that fits in VRAM. On a 16GB T4 use 60.")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation steps. effective_batch = batch_size * accum_steps. "
                        "Use --batch-size 60 --accum-steps 5 to reach the paper's effective batch "
                        "of 300 on a 16GB GPU.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--cavity-threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-lung-crop", action="store_true",
                   help="Debug: ignore crops-dir and train on whole images (NOT paper-faithful).")
    p.add_argument("--amp", action="store_true", default=True)
    args = p.parse_args(argv)

    device = pick_device()
    print(f"[paper-baseline] device={device}")
    df = load_manifest(args.manifest)

    crops_dir = None if args.no_lung_crop else args.crops_dir
    if crops_dir is None:
        print("[paper-baseline][WARN] training on WHOLE images (no lung crop). "
              "This is NOT paper-faithful — pass --crops-dir for replication.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    results_last: dict[str, dict] = {}
    for held_out in args.held_outs:
        for seed in args.seeds:
            row, res = run_country(df, held_out, seed, args, device, crops_dir, out_dir)
            rows.append(row)
            results_last[held_out] = res  # last seed feeds the comparison table
            pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    print_comparison_table(results_last)

    df_rows = pd.DataFrame(rows)
    print("\n[paper-baseline] mean +/- std across seeds:")
    for ho in args.held_outs:
        s = df_rows[df_rows["held_out"] == ho]
        if len(s) == 0:
            continue
        print(f"  {ho:12s} ALP_MAE={s['alp_mae'].mean():5.2f}+/-{s['alp_mae'].std():.2f}  "
              f"cavity_AUC={s['cavity_auc'].mean():.3f}+/-{s['cavity_auc'].std():.3f}  "
              f"Timika_MAE={s['timika_mae'].mean():5.2f}  "
              f"Pearson={s['timika_pearson'].mean():.2f}")
    print(f"\n[paper-baseline] results -> {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
