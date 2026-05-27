"""Train the Domain-Adversarial Mixture-of-Experts (DA-MoE) for Timika scoring.

This is the novelty layer on top of the Kantipudi A1/A2/A3 baselines (see
:mod:`src.components.da_moe`). The SAME trainer covers all three approaches via
``--mode``:

    a2      mixture of ALP experts + frozen cavity agent  (vs Kantipudi A2)
    a3      mixture of direct-Timika experts              (vs Kantipudi A3)
    a1      detection-ALP view + frozen cavity agent       (vs Kantipudi A1)
    fusion  all views fused by one gate                    (flagship)

Protocol matches the baselines exactly (country-segregated leave-one-country-out,
patient-disjoint val, cavity on the paper's balanced split). Training is two-phase:

    Phase 1 (--pretrain-epochs)  pretrain experts on their own targets + DANN trunk
    Phase 2 (rest)               train gate + critic on final Timika + DANN + load-balance

Ablations: --no-dann, --no-critic, --lambda-dann, --gate-temp, --n-alp-experts,
--n-timika-experts. Compare against the locked baselines in baseline_runs/.

Example (A2-mode flagship on a T4)::

    python -m src.training.train_da_moe --mode a2 \
        --manifest tbportals_manifest_paper.csv --crops-dir crops \
        --out-dir checkpoints/moe_a2 --seeds 0 1 2 \
        --epochs 30 --pretrain-epochs 10 --batch-size 60 --accum-steps 5 \
        --cavity-no-lung-crop
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.components.da_moe import MoEConfig, RegressionMoE, load_balance_loss
from src.data.tbportals import load_manifest, make_country_split
from src.data.tbportals_dataset import TBPortalsDataset, build_country_index
from src.evaluation.eval_tbportals import (
    KANTIPUDI_A1,
    KANTIPUDI_A2,
    KANTIPUDI_A3,
    TIMIKA_MAX,
    _safe_pearson,
    bootstrap_ci,
    cavity_metrics,
    regression_metrics,
)
from src.training.train_baseline_paper import (
    make_balanced_cavity_split,
    pick_device,
    seed_everything,
    train_cavity,
)

_REF = {"a1": KANTIPUDI_A1, "a2": KANTIPUDI_A2, "a3": KANTIPUDI_A3, "fusion": KANTIPUDI_A2}


# ── Loaders (carry country_to_idx for DANN) ───────────────────────────────────

def _moe_loader(df, train, crops_dir, country_to_idx, batch, workers):
    ds = TBPortalsDataset(
        df, train=train, country_to_idx=country_to_idx,
        crops_dir=crops_dir, use_lung_crop=bool(crops_dir),
    )
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=workers,
                      pin_memory=True, drop_last=False)


def _gather(map_: dict[str, float], ids, device) -> torch.Tensor:
    return torch.tensor([map_.get(str(i), 0.0) for i in ids], dtype=torch.float32, device=device)


# ── Cavity agent (separate, frozen — identical protocol to A2) ────────────────

@torch.no_grad()
def _cavity_prob_map(model, df, args, device, crops_dir) -> dict[str, float]:
    """P(cavity) for every image in df, keyed by image_id (no augmentation)."""
    loader = _moe_loader(df, False, crops_dir, {}, args.batch_size, args.num_workers)
    model.eval()
    out: dict[str, float] = {}
    for b in loader:
        p = torch.softmax(model(b["image"].to(device)), dim=1)[:, 1].cpu().numpy()
        for img_id, prob in zip(b["image_id"], p):
            out[str(img_id)] = float(prob)
    return out


# ── DANN gradient-reversal schedule ───────────────────────────────────────────

def _grl_lambda(epoch: int, total: int, lam_max: float) -> float:
    p = epoch / max(1, total - 1)
    return lam_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ── Losses ────────────────────────────────────────────────────────────────────

def _expert_mse(out, alp_true_frac, timika_true_frac):
    loss = alp_true_frac.new_zeros(())
    n = 0
    if out.expert_alp is not None:
        loss = loss + F.mse_loss(out.expert_alp, alp_true_frac.unsqueeze(1).expand_as(out.expert_alp))
        n += 1
    if out.expert_timika is not None:
        loss = loss + F.mse_loss(out.expert_timika, timika_true_frac.unsqueeze(1).expand_as(out.expert_timika))
        n += 1
    return loss / max(1, n)


# ── Train one (held-out country, seed) ────────────────────────────────────────

def run_country(df, held_out, seed, args, device, crops_dir, cav_crops, det_alp_map, out_dir):
    seed_everything(seed)
    print(f"\n===== DA-MoE[{args.mode}]  {held_out}  seed={seed} =====")

    reg_train, reg_val, test_df = make_country_split(
        df, held_out, val_fraction=args.val_fraction, seed=seed
    )
    country_to_idx = build_country_index(reg_train)
    n_countries = len(country_to_idx)
    print(f"[MoE] reg_train={len(reg_train)} val={len(reg_val)} test={len(test_df)} "
          f"| {n_countries} train-countries for DANN")

    # 1) Cavity agent (frozen) — only for modes that use cavity.
    cav_prob_map: dict[str, float] = {}
    needs_cavity = args.mode in ("a1", "a2", "fusion")
    if needs_cavity:
        cav_train, cav_val, _ = make_balanced_cavity_split(
            df, held_out, val_fraction=args.val_fraction, seed=seed
        )
        print(f"[MoE][CAV] train={len(cav_train)} val={len(cav_val)} (balanced)")
        cav_model, cav_best = train_cavity(cav_train, cav_val, args, device, cav_crops)
        all_imgs = pd.concat([reg_train, reg_val, test_df]).drop_duplicates("image_id")
        cav_prob_map = _cavity_prob_map(cav_model, all_imgs, args, device, cav_crops)
        del cav_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        cav_best = float("nan")

    # 2) Build the MoE.
    cfg = MoEConfig(
        mode=args.mode, n_alp_experts=args.n_alp_experts, n_timika_experts=args.n_timika_experts,
        use_a1_view=bool(det_alp_map) or args.mode == "a1",
        n_train_countries=n_countries, gate_temp=args.gate_temp,
        use_dann=not args.no_dann, use_critic=not args.no_critic, pretrained=not args.no_pretrained,
    )
    model = RegressionMoE(cfg).to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=args.lr)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    train_loader = _moe_loader(reg_train, True, crops_dir, country_to_idx, args.batch_size, args.num_workers)
    val_loader = _moe_loader(reg_val, False, crops_dir, country_to_idx, args.batch_size, args.num_workers)
    accum = max(1, args.accum_steps)
    n_batches = len(train_loader)

    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        phase2 = epoch >= args.pretrain_epochs
        lam = _grl_lambda(epoch, args.epochs, args.lambda_dann) if cfg.use_dann else 0.0
        model.train()
        opt.zero_grad(set_to_none=True)
        run = 0.0
        for i, b in enumerate(train_loader):
            img = b["image"].to(device, non_blocking=True)
            alp_t = b["alp"].to(device)                       # ALP fraction [0,1]
            cav_t = b["cavity"].to(device)
            tim_t = (alp_t * 100.0 + 40.0 * cav_t) / TIMIKA_MAX
            cidx = b["country_idx"].to(device)
            cp = _gather(cav_prob_map, b["image_id"], device) if needs_cavity else None
            da = _gather(det_alp_map, b["image_id"], device) if cfg.use_a1_view else None
            with autocast("cuda", enabled=use_amp):
                out = model(img, cav_prob=cp, det_alp=da, grl_lambda=lam)
                le = _expert_mse(out, alp_t, tim_t)
                if phase2:
                    loss = (F.mse_loss(out.timika_frac, tim_t)
                            + 0.1 * le + args.lb_weight * load_balance_loss(out.gate_w))
                else:
                    loss = le
                if cfg.use_dann and out.country_logits is not None:
                    mask = cidx >= 0
                    if mask.any():
                        loss = loss + F.cross_entropy(out.country_logits[mask], cidx[mask])
            # In degenerate configs the loss can have no trainable parameters,
            # e.g. mode=a1 (a single fixed detection view, no learnable experts)
            # with --no-dann in phase 1, or with --no-dann --no-critic in either
            # phase. Skip the step instead of crashing on backward().
            trainable = loss.requires_grad
            if trainable:
                scaler.scale(loss / accum).backward()
            run += float(loss.detach()) * img.size(0)
            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                if trainable:
                    scaler.step(opt)
                    scaler.update()
                opt.zero_grad(set_to_none=True)
        if not trainable and epoch in (0, args.pretrain_epochs):
            ph0 = "phase 2" if phase2 else "phase 1"
            print(f"  [note] {ph0}: loss has no trainable parameters "
                  f"(expected for mode=a1 with --no-dann/--no-critic); optimisation skipped.")

        # validation: final-Timika MSE (works in both phases)
        model.eval()
        vsum, n = 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                img = b["image"].to(device)
                alp_t = b["alp"].to(device); cav_t = b["cavity"].to(device)
                tim_t = (alp_t * 100.0 + 40.0 * cav_t) / TIMIKA_MAX
                cp = _gather(cav_prob_map, b["image_id"], device) if needs_cavity else None
                da = _gather(det_alp_map, b["image_id"], device) if cfg.use_a1_view else None
                with autocast("cuda", enabled=use_amp):
                    out = model(img, cav_prob=cp, det_alp=da)
                    vsum += float(F.mse_loss(out.timika_frac, tim_t, reduction="sum"))
                n += img.size(0)
        vmse = vsum / max(1, n)
        ph = "P2" if phase2 else "P1"
        print(f"  [{ph}] epoch {epoch:02d} train_loss={run/len(train_loader.dataset):.5f} "
              f"val_timika_mse={vmse:.5f} grl_lambda={lam:.3f}")
        if phase2 and vmse < best_val:
            best_val = vmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / f"moe_{args.mode}_{held_out}_seed{seed}.pt")

    # 3) Test prediction
    row = _evaluate(model, test_df, args, device, crops_dir, cav_prob_map, det_alp_map,
                    cfg, held_out, seed, cav_best)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


@torch.no_grad()
def _evaluate(model, test_df, args, device, crops_dir, cav_prob_map, det_alp_map,
              cfg, held_out, seed, cav_best) -> dict:
    loader = _moe_loader(test_df, False, crops_dir, {}, args.batch_size, args.num_workers)
    model.eval()
    needs_cavity = cfg.mode in ("a1", "a2", "fusion")
    t_pred, alp_pred, alp_true, cav_true, cav_prob = [], [], [], [], []
    for b in loader:
        img = b["image"].to(device)
        cp = _gather(cav_prob_map, b["image_id"], device) if needs_cavity else None
        da = _gather(det_alp_map, b["image_id"], device) if cfg.use_a1_view else None
        out = model(img, cav_prob=cp, det_alp=da)
        t_pred.append(out.timika_frac.cpu().numpy() * TIMIKA_MAX)
        alp_true.append(b["alp"].numpy() * 100.0)
        cav_true.append(b["cavity"].numpy())
        if out.alp_frac is not None:
            alp_pred.append(out.alp_frac.cpu().numpy() * 100.0)
        if needs_cavity:
            cav_prob.append(cp.cpu().numpy())

    t_pred = np.concatenate(t_pred)
    alp_true = np.concatenate(alp_true)
    cav_true = np.concatenate(cav_true)
    t_true = alp_true + 40.0 * cav_true

    tim = regression_metrics(t_true, t_pred)
    mae_fn = lambda a, b: float(np.mean(np.abs(b - a)))
    pear_fn = lambda a, b: _safe_pearson(a, b)
    ci_mae = bootstrap_ci(t_true, t_pred, mae_fn)
    ci_pear = bootstrap_ci(t_true, t_pred, pear_fn)

    row = {
        "approach": f"MoE-{cfg.mode}", "held_out": held_out, "seed": seed, "n_test": len(test_df),
        "timika_mae": tim["mae"], "timika_mae_pct": tim["mae_pct"], "timika_rmse": tim["rmse"],
        "timika_pearson": tim["pearson"], "timika_spearman": tim["spearman"],
        "timika_mae_ci_lo": ci_mae[0], "timika_mae_ci_hi": ci_mae[1],
        "timika_pearson_ci_lo": ci_pear[0], "timika_pearson_ci_hi": ci_pear[1],
        "cav_best_val_ce": cav_best,
    }
    if alp_pred:
        a = regression_metrics(alp_true, np.concatenate(alp_pred))
        row["alp_mae"] = a["mae"]
    if cav_prob:
        c = cavity_metrics(cav_true, np.concatenate(cav_prob), threshold=args.cavity_threshold)
        row["cavity_auc"] = c["auc"]; row["cavity_f1"] = c["f1"]

    ref = _REF.get(cfg.mode, {}).get(held_out, {})
    print(f"[RESULT] MoE-{cfg.mode} {held_out} seed={seed}  "
          f"Timika_MAE={row['timika_mae']:.2f} ({row['timika_mae_pct']:.2f}%) | paper "
          f"{ref.get('timika_mae', float('nan')):.2f}   "
          f"Pearson={row['timika_pearson']:.2f} | paper {ref.get('timika_pearson', float('nan')):.2f}")
    return row


# ── det_alp CSV (A1 view) ─────────────────────────────────────────────────────

def _load_det_alp(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    d = pd.read_csv(path, dtype={"image_id": str})
    col = "det_alp" if "det_alp" in d.columns else "alp_pred_100"
    m = {str(i): float(v) for i, v in zip(d["image_id"], d[col])}
    print(f"[MoE] loaded {len(m)} det_alp values from {path}")
    return m


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train the DA-MoE Timika scorer (a1/a2/a3/fusion).")
    p.add_argument("--mode", choices=["a1", "a2", "a3", "fusion"], default="a2")
    p.add_argument("--manifest", required=True)
    p.add_argument("--crops-dir", default=None, help="Lung-crop dir for the regression trunk input.")
    p.add_argument("--det-alp-csv", default=None,
                   help="CSV (image_id,det_alp) of A1 detection-ALP for the A1 view. "
                        "Required for --mode a1; optional extra view for --mode fusion.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--held-outs", nargs="+", default=["Romania", "Moldova", "Kazakhstan"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pretrain-epochs", type=int, default=10, help="Phase-1 expert-pretraining epochs.")
    p.add_argument("--batch-size", type=int, default=60)
    p.add_argument("--accum-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--cavity-threshold", type=float, default=0.5)
    p.add_argument("--n-alp-experts", type=int, default=4)
    p.add_argument("--n-timika-experts", type=int, default=4)
    p.add_argument("--gate-temp", type=float, default=1.0)
    p.add_argument("--lambda-dann", type=float, default=1.0, help="Max GRL strength (0 disables effect).")
    p.add_argument("--lb-weight", type=float, default=0.01, help="Load-balance regulariser weight.")
    p.add_argument("--no-dann", action="store_true")
    p.add_argument("--no-critic", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-lung-crop", action="store_true")
    p.add_argument("--cavity-no-lung-crop", action="store_true",
                   help="Train the cavity agent on whole images (match the locked A2 config).")
    p.add_argument("--amp", action="store_true", default=True)
    args = p.parse_args(argv)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = pick_device()
    print(f"[da-moe] device={device} mode={args.mode} dann={not args.no_dann} critic={not args.no_critic}")
    df = load_manifest(args.manifest)

    base_crops = None if args.no_lung_crop else args.crops_dir
    cav_crops = None if args.cavity_no_lung_crop else base_crops
    det_alp_map = _load_det_alp(args.det_alp_csv)
    if args.mode == "a1" and not det_alp_map:
        raise SystemExit("[da-moe] --mode a1 requires --det-alp-csv (detection ALP per image_id).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for held_out in args.held_outs:
        for seed in args.seeds:
            rows.append(run_country(df, held_out, seed, args, device,
                                    base_crops, cav_crops, det_alp_map, out_dir))
            pd.DataFrame(rows).to_csv(out_dir / "results_moe.csv", index=False)

    dfr = pd.DataFrame(rows)
    ref = _REF.get(args.mode, {})
    print(f"\n[da-moe] mean +/- std across seeds (vs Kantipudi {args.mode.upper()}):")
    for ho in args.held_outs:
        s = dfr[dfr["held_out"] == ho]
        if len(s) == 0:
            continue
        r = ref.get(ho, {})
        print(f"  {ho:12s} Timika_MAE={s['timika_mae'].mean():5.2f}+/-{s['timika_mae'].std():.2f} "
              f"(paper {r.get('timika_mae', float('nan')):.2f})  "
              f"Pearson={s['timika_pearson'].mean():.2f} (paper {r.get('timika_pearson', float('nan')):.2f})")
    print(f"\n[da-moe] results -> {out_dir / 'results_moe.csv'}")


if __name__ == "__main__":
    main()
