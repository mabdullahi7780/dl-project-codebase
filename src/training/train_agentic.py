"""Agentic Timika pipeline — Rung 1: light heads on cached frozen features.

Rung 1 isolates the two highest-evidence floor-raisers before any agentic
machinery: a CXR-foundation backbone (RAD-DINO, cached) + a balanced-regression
objective (Balanced MSE) + honest model selection (val Pearson, all epochs).

Mode a2: ALP regression head + cavity classifier head (balanced split), Timika =
ALP%*1 + 40*cavity, evaluated under the same country-segregated LOCO protocol and
compared to BOTH our locked baseline (honest target) and Kantipudi (aspirational).

Later rungs swap the ALP head for the severity-MoE + retrieval + critic; the
feature contract and this LOCO/eval harness stay identical.

Example::

    python -m src.training.train_agentic --mode a2 --loss bmc \
        --features raddino_features.npz --manifest tbportals_manifest_paper.csv \
        --out-dir checkpoints/agentic_a2_bmc --seeds 0 1 2 --select-metric pearson
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.cache_features import load_features
from src.components.feature_heads import ClassifierHead, RegressionHead
from src.data.tbportals import load_manifest, make_country_split
from src.evaluation.eval_tbportals import (
    KANTIPUDI_A2,
    LOCKED_BASELINE,
    Predictions,
    _safe_pearson,
    evaluate_split,
)
from src.training.losses import make_reg_loss
from src.training.train_baseline_paper import (
    make_balanced_cavity_split,
    pick_device,
    seed_everything,
)


def _xy(df: pd.DataFrame, feats: dict[str, np.ndarray], device, target: str):
    X, y, ids = [], [], []
    miss = 0
    for r in df.itertuples(index=False):
        iid = str(r.image_id)
        v = feats.get(iid)
        if v is None:
            miss += 1
            continue
        X.append(v)
        ids.append(iid)
        y.append(float(r.alp_0_100) / 100.0 if target == "alp" else float(r.cavity))
    if miss:
        print(f"[agentic] WARNING: {miss}/{len(df)} rows missing from feature cache (skipped)")
    Xt = torch.tensor(np.stack(X), dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    return Xt, yt, ids


def _val_score(vt: np.ndarray, vp: np.ndarray, metric: str) -> float:
    """Higher is better for all (negate error metrics)."""
    if metric == "pearson":
        p = _safe_pearson(vt, vp)
        return -1e9 if np.isnan(p) else p
    if metric == "mae":
        return -float(np.mean(np.abs(vp - vt)))
    return -float(np.mean((vp - vt) ** 2))  # mse


def _train_reg(Xtr, ytr, Xval, yval, dim, args, device):
    head = RegressionHead(dim, args.hidden).to(device)
    loss_mod, loss_fn = make_reg_loss(args.loss)
    params = list(head.parameters())
    if loss_mod is not None:
        loss_mod.to(device)
        params += list(loss_mod.parameters())
    opt = torch.optim.NAdam(params, lr=args.lr, weight_decay=1e-4)

    N = Xtr.size(0)
    best_state, best_score = None, -float("inf")
    yval_np = yval.cpu().numpy()
    for _ in range(args.epochs):
        head.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = loss_fn(head(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vp = head(Xval).cpu().numpy()
        score = _val_score(yval_np, vp, args.select_metric)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head


def _train_cav(Xtr, ytr, Xval, yval, dim, args, device):
    head = ClassifierHead(dim, args.hidden).to(device)
    opt = torch.optim.NAdam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    N = Xtr.size(0)
    ytr_l = ytr.long()
    best_state, best_ce = None, float("inf")
    for _ in range(args.epochs):
        head.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = torch.nn.functional.cross_entropy(head(Xtr[idx]), ytr_l[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vce = float(torch.nn.functional.cross_entropy(head(Xval), yval.long()))
        if vce < best_ce:
            best_ce = vce
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best_ce


def run_country(df, held_out, seed, feats, dim, args, device, out_dir):
    seed_everything(seed)
    print(f"\n===== agentic[{args.tag}]  {held_out}  seed={seed} =====")

    a_tr, a_val, test_df = make_country_split(df, held_out, val_fraction=args.val_fraction, seed=seed)
    Xtr, ytr, _ = _xy(a_tr, feats, device, "alp")
    Xval, yval, _ = _xy(a_val, feats, device, "alp")
    reg = _train_reg(Xtr, ytr, Xval, yval, dim, args, device)

    c_tr, c_val, _ = make_balanced_cavity_split(df, held_out, val_fraction=args.val_fraction, seed=seed)
    Xct, yct, _ = _xy(c_tr, feats, device, "cavity")
    Xcv, ycv, _ = _xy(c_val, feats, device, "cavity")
    cav, cav_ce = _train_cav(Xct, yct, Xcv, ycv, dim, args, device)

    Xte, _, te_ids = _xy(test_df, feats, device, "alp")
    test_df = test_df[test_df["image_id"].astype(str).isin(set(te_ids))].reset_index(drop=True)
    reg.eval(); cav.eval()
    with torch.no_grad():
        alp_pred = reg(Xte).cpu().numpy() * 100.0
        cav_prob = torch.softmax(cav(Xte), dim=1)[:, 1].cpu().numpy()
    alp_true = test_df["alp_0_100"].to_numpy(dtype=np.float32)
    cav_true = test_df["cavity"].to_numpy(dtype=np.float32)

    preds = Predictions(alp_true_100=alp_true, alp_pred_100=alp_pred,
                        cavity_true=cav_true, cavity_prob=cav_prob)
    res = evaluate_split(preds, cavity_threshold=args.cavity_threshold)
    t = res["timika"]
    ci = t.get("mae_ci95", (float("nan"), float("nan")))

    base = LOCKED_BASELINE["a2"].get(held_out, {})
    pap = KANTIPUDI_A2.get(held_out, {})
    base_mae = base.get("timika_mae", float("nan"))
    if ci[1] < base_mae:
        verdict = "BEATS locked baseline (CI excludes it)"
    elif ci[0] > base_mae:
        verdict = "WORSE than baseline (CI excludes it)"
    else:
        verdict = "within noise of baseline"

    row = {
        "tag": args.tag, "backbone": args.backbone_name, "loss": args.loss,
        "held_out": held_out, "seed": seed, "n_test": len(test_df),
        "alp_mae": res["alp"]["mae"], "cavity_auc": res["cavity"]["auc"], "cavity_f1": res["cavity"]["f1"],
        "timika_mae": t["mae"], "timika_mae_pct": t["mae_pct"], "timika_pearson": t["pearson"],
        "timika_mae_ci_lo": ci[0], "timika_mae_ci_hi": ci[1],
        "base_timika_mae": base_mae, "paper_timika_mae": pap.get("timika_mae", float("nan")),
        "cav_best_val_ce": cav_ce,
    }
    print(f"[RESULT] {held_out} s{seed}  Timika_MAE={t['mae']:.2f} "
          f"CI95=[{ci[0]:.2f},{ci[1]:.2f}]  Pearson={t['pearson']:.3f} | "
          f"base {base_mae:.2f}/{base.get('timika_pearson', float('nan')):.2f} | "
          f"paper {pap.get('timika_mae', float('nan')):.2f}  -> {verdict}")

    # per-image predictions for later paired significance tests
    pd.DataFrame({"image_id": test_df["image_id"].to_numpy(), "country": held_out, "seed": seed,
                  "alp_true": alp_true, "alp_pred": alp_pred,
                  "cavity_true": cav_true, "cavity_prob": cav_prob,
                  "timika_true": alp_true + 40.0 * cav_true,
                  "timika_pred": alp_pred + 40.0 * (cav_prob > args.cavity_threshold)}
                 ).to_csv(out_dir / f"preds_{args.tag}_{held_out}_seed{seed}.csv", index=False)
    return row


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Agentic Timika Rung 1 (cached features + balanced regression).")
    p.add_argument("--features", required=True, help=".npz cache from scripts/cache_features.py")
    p.add_argument("--manifest", required=True)
    p.add_argument("--mode", default="a2", choices=["a2"])  # a3/fusion in later rungs
    p.add_argument("--loss", default="bmc", choices=["mse", "bmc"])
    p.add_argument("--select-metric", default="pearson", choices=["pearson", "mae", "mse"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--held-outs", nargs="+", default=["Romania", "Moldova", "Kazakhstan"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--cavity-threshold", type=float, default=0.5)
    p.add_argument("--tag", default=None, help="Output naming tag (default: backbone_loss).")
    args = p.parse_args(argv)

    device = pick_device()
    feats, dim = load_features(args.features)
    z = np.load(args.features, allow_pickle=True)
    args.backbone_name = str(z["backbone"]) if "backbone" in z else "unknown"
    args.tag = args.tag or f"{args.backbone_name}_{args.loss}"
    print(f"[agentic] device={device} backbone={args.backbone_name} dim={dim} "
          f"loss={args.loss} select={args.select_metric} tag={args.tag}")
    df = load_manifest(args.manifest)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for held_out in args.held_outs:
        for seed in args.seeds:
            rows.append(run_country(df, held_out, seed, feats, dim, args, device, out_dir))
            pd.DataFrame(rows).to_csv(out_dir / "results_agentic.csv", index=False)

    dfr = pd.DataFrame(rows)
    print(f"\n[agentic] mean +/- std across seeds (tag={args.tag}):")
    for ho in args.held_outs:
        s = dfr[dfr["held_out"] == ho]
        if not len(s):
            continue
        base = LOCKED_BASELINE["a2"].get(ho, {})
        print(f"  {ho:12s} Timika_MAE={s['timika_mae'].mean():5.2f}+/-{s['timika_mae'].std():.2f} "
              f"Pearson={s['timika_pearson'].mean():.3f} | base {base.get('timika_mae', float('nan')):.2f}"
              f"/{base.get('timika_pearson', float('nan')):.2f} | paper {KANTIPUDI_A2.get(ho, {}).get('timika_mae', float('nan')):.2f}")
    print(f"\n[agentic] results -> {out_dir / 'results_agentic.csv'}")


if __name__ == "__main__":
    main()
