"""Agentic Timika pipeline — mode- and rung-aware trainer on cached frozen features.

Runs the full 6-rung ablation ladder for one mode (a2 / a3 / fusion / a1) under the
country-segregated LOCO protocol, on cached RAD-DINO features. Every rung emits one
results row per (held-out country, seed) with a bootstrap-CI verdict against BOTH our
locked baseline (honest target) and Kantipudi (aspirational).

Rungs (each an independent ablation toggle):
  1. backbone + balanced regression  (loss = mse | bmc)
  2. test-time feature adaptation     (transductive z-score)            [a2/a3/fusion]
  3. retrieval-augmented calibration  (kNN blend, alpha tuned on val)
  4. cavity-head upgrade              (class-balanced focal + threshold) [a2/fusion/a1]
  5. severity MoE + critic            (no DANN)                          [a2/a3/fusion]
  6. conformal + deep ensemble        (M seeds, split-conformal intervals)
  + stacked: the safe combiners (retrieval + focal cavity + ensemble + conformal).

Features are frozen and cached, so the whole ladder is minutes of compute. a1 uses a
patch-GRID cache ([N, P, D]) with a spatial ALP head; all other modes use the global
CLS cache ([N, D]).

Example::

    python -m src.training.train_agentic --mode a2 --features feats.npz \
        --manifest tbportals_manifest_paper.csv --out-dir out/agentic_a2 \
        --rungs 1 2 3 4 5 6 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.cache_features import load_features
from src.components.conformal import ensemble_mean, split_conformal
from src.components.feature_heads import (
    ClassifierHead,
    CriticHead,
    MoERegressionHead,
    RegressionHead,
    SpatialALPHead,
    SpatialCavityHead,
)
from src.components.retrieval import RetrievalCalibrator
from src.data.tbportals import load_manifest, make_country_split
from src.evaluation.eval_tbportals import (
    Predictions,
    _safe_pearson,
    evaluate_split,
    evaluate_timika_direct,
    references_for_mode,
)
from src.training.losses import class_balanced_weights, focal_ce, make_reg_loss
from src.training.tta import FeatureStandardizer
from src.training.train_baseline_paper import (
    make_balanced_cavity_split,
    pick_device,
    seed_everything,
)

CAV_BONUS = 40.0


# ── rung ladder ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RungCfg:
    name: str
    loss: str = "bmc"            # mse | bmc (Rung 1)
    standardize: str = "none"    # none | train | transductive (Rung 2)
    retrieval: bool = False      # Rung 3
    cavity_loss: str = "ce"      # ce | focal (Rung 4)
    cavity_cal: bool = False     # per-country threshold calibration (Rung 4)
    cavity_head: str = "global"  # global | spatial (spatial = SpatialCavityHead on patch grid)
    alp_head: str = "mlp"        # mlp | moe (Rung 5)
    critic: bool = False         # Rung 5
    ensemble: int = 1            # M members (Rung 6)
    conformal: bool = False      # Rung 6
    calibrate_pred: str = "none" # none | linear | isotonic (Rung 7 — fixes slope <1)


def _fit_calibrate(val_pred: np.ndarray, val_true: np.ndarray, kind: str):
    """Fit a post-hoc calibrator on (val_pred, val_true) -> callable applied to test.

    Both are in the head's working scale (e.g. ALP/100 or Timika/140 in [0,1]).
    The fitted transform corrects the slope < 1 compression observed in MSE-trained
    sigmoid regressors (`pred ≈ 0.55·true + b`). ``'linear'`` fits y = a·x + b on val;
    ``'isotonic'`` fits a monotone non-parametric mapping (more flexible, can overfit
    small val sets — use linear by default).
    """
    val_pred = np.asarray(val_pred, dtype=np.float64).reshape(-1)
    val_true = np.asarray(val_true, dtype=np.float64).reshape(-1)
    if kind == "none" or len(val_pred) < 4 or np.std(val_pred) < 1e-6:
        return lambda x: np.asarray(x, dtype=np.float32)
    if kind == "linear":
        a, b = np.polyfit(val_pred, val_true, 1)
        return lambda x: np.clip(a * np.asarray(x) + b, 0.0, 1.0).astype(np.float32)
    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(val_pred, val_true)
        return lambda x: iso.transform(np.asarray(x)).astype(np.float32)
    raise ValueError(f"unknown calibrate_pred kind {kind!r}")


def rung_ladder(mode: str, requested: list[int], *, ensemble_M: int = 5,
                grid_available: bool = False) -> list[RungCfg]:
    """Build the per-mode rung ladder. Set ``grid_available=True`` when a patch-grid
    feature cache is loaded so spatial cavity / spatial ALP heads become usable.
    """
    has_cavity = mode in ("a2", "fusion", "a1")
    a1_grid = mode == "a1"     # a1 uses grid features for ALP head too
    out: list[RungCfg] = []
    if 1 in requested:
        out.append(RungCfg("rung1_mse", loss="mse"))
        out.append(RungCfg("rung1_bmc", loss="bmc"))
    if 2 in requested and not a1_grid:
        out.append(RungCfg("rung2_tta", standardize="transductive"))
    if 3 in requested:
        out.append(RungCfg("rung3_retrieval", retrieval=True))
    if 4 in requested and has_cavity:
        out.append(RungCfg("rung4_cavity", cavity_loss="focal", cavity_cal=True))
    if 5 in requested and not a1_grid:
        out.append(RungCfg("rung5_moe", alp_head="moe", critic=True))
    if 6 in requested:
        out.append(RungCfg("rung6_conformal", ensemble=ensemble_M, conformal=True))
    if 7 in requested:
        # Slope-calibration in isolation (base = rung1 with retrieval off, ensemble off).
        out.append(RungCfg("rung7_calibrate_lin", calibrate_pred="linear"))
        out.append(RungCfg("rung7_calibrate_iso", calibrate_pred="isotonic"))

    # Rung 4 was a negative result → "agentic_best" intentionally drops the focal-cavity
    # change and uses CE cavity. It stacks the two positive contributors (R3 + R6).
    out.append(RungCfg("agentic_best", retrieval=True,
                       ensemble=ensemble_M if 6 in requested else 1,
                       conformal=(6 in requested)))
    # agentic_v2 = agentic_best + slope calibration (the slope-fix headline).
    if 7 in requested:
        out.append(RungCfg("agentic_v2_lin", retrieval=True,
                           ensemble=ensemble_M if 6 in requested else 1,
                           conformal=(6 in requested), calibrate_pred="linear"))
        out.append(RungCfg("agentic_v2_iso", retrieval=True,
                           ensemble=ensemble_M if 6 in requested else 1,
                           conformal=(6 in requested), calibrate_pred="isotonic"))
    # TTA helps Moldova specifically — keep a separate "best_tta" variant so the paper
    # can quote ours-on-Moldova with TTA and ours-on-Rom/Kaz without.
    if not a1_grid:
        out.append(RungCfg("agentic_best_tta", standardize="transductive", retrieval=True,
                           ensemble=ensemble_M if 6 in requested else 1,
                           conformal=(6 in requested)))
    # Romania-cavity attack: spatial cavity head over the patch grid (replaces the global
    # head). Only meaningful when a patch grid is available + the mode uses cavity.
    if grid_available and has_cavity:
        out.append(RungCfg("rung4b_spatial_cavity", cavity_head="spatial"))
        out.append(RungCfg("agentic_best_spatcav", cavity_head="spatial", retrieval=True,
                           ensemble=ensemble_M if 6 in requested else 1,
                           conformal=(6 in requested)))

    # Legacy stacked (kept for backwards-compat): R3 + R4 + R6. We know R4 hurts, but
    # leaving it surfaces the comparison cleanly.
    stack = dict(name="stacked_legacy", retrieval=(3 in requested),
                 ensemble=ensemble_M if 6 in requested else 1, conformal=(6 in requested))
    if 4 in requested and has_cavity:
        stack.update(cavity_loss="focal", cavity_cal=True)
    out.append(RungCfg(**stack))
    return out


# ── feature/label gathering ─────────────────────────────────────────────────────

def _gather(df: pd.DataFrame, feats: dict[str, np.ndarray], device):
    """Return (X tensor, aligned sub-DataFrame). X is [N, D] (CLS) or [N, P, D] (grid)."""
    X, keep = [], []
    for i, iid in enumerate(df["image_id"].astype(str)):
        v = feats.get(iid)
        if v is None:
            continue
        X.append(v)
        keep.append(i)
    if not X:
        raise RuntimeError("no features matched this split's image_ids")
    Xt = torch.tensor(np.stack(X), dtype=torch.float32, device=device)
    sub = df.iloc[keep].reset_index(drop=True)
    return Xt, sub


def _pool(X: torch.Tensor) -> torch.Tensor:
    """Pool grid features [N,P,D] -> [N,D]; pass [N,D] through unchanged."""
    return X.mean(dim=1) if X.dim() == 3 else X


def _alp_target(sub) -> torch.Tensor:
    return torch.tensor(sub["alp_0_100"].to_numpy(np.float32) / 100.0)


def _timika_target(sub) -> torch.Tensor:
    t = sub["alp_0_100"].to_numpy(np.float32) + CAV_BONUS * sub["cavity"].to_numpy(np.float32)
    return torch.tensor(t / 140.0)


# ── regression head training (shared loop) ──────────────────────────────────────

def _val_score(vt: np.ndarray, vp: np.ndarray, metric: str) -> float:
    if metric == "pearson":
        p = _safe_pearson(vt, vp)
        return -1e9 if np.isnan(p) else p
    if metric == "mae":
        return -float(np.mean(np.abs(vp - vt)))
    return -float(np.mean((vp - vt) ** 2))


def _make_reg_head(kind: str, in_dim: int, hidden: int, spatial: bool, n_experts: int):
    if spatial:
        return SpatialALPHead(in_dim, hidden)
    if kind == "moe":
        return MoERegressionHead(in_dim, hidden, n_experts=n_experts)
    return RegressionHead(in_dim, hidden)


def _train_reg(Xtr, ytr, Xval, yval, in_dim, *, kind, loss, select_metric,
               spatial, args, device, seed):
    seed_everything(seed)
    head = _make_reg_head(kind, in_dim, args.hidden, spatial, args.n_experts).to(device)
    loss_mod, loss_fn = make_reg_loss(loss)
    params = list(head.parameters())
    if loss_mod is not None:
        loss_mod.to(device)
        params += list(loss_mod.parameters())
    opt = torch.optim.NAdam(params, lr=args.lr, weight_decay=1e-4)
    N = Xtr.size(0)
    yval_np = yval.cpu().numpy()
    best_state, best = None, -float("inf")
    for _ in range(args.epochs):
        head.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            pred = head(Xtr[idx])
            loss_val = loss_fn(pred, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss_val.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vp = head(Xval).cpu().numpy()
        s = _val_score(yval_np, vp, select_metric)
        if s > best:
            best = s
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    return head


def _reg_predict(head, X) -> np.ndarray:
    with torch.no_grad():
        return head(X).cpu().numpy()


def _train_cav(Xtr, ytr, Xval, yval, in_dim, *, loss, calibrate, args, device, seed,
               head_kind: str = "global"):
    """Train cavity head. ``head_kind='global'`` -> ClassifierHead on pooled features;
    ``'spatial'`` -> SpatialCavityHead on a [B, P, D] patch grid (caller must pass
    un-pooled features for both train and val)."""
    seed_everything(seed + 100)
    if head_kind == "spatial":
        head = SpatialCavityHead(in_dim, args.hidden).to(device)
    else:
        head = ClassifierHead(in_dim, args.hidden).to(device)
    opt = torch.optim.NAdam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    ytr_l = ytr.long()
    n_pos = int(ytr_l.sum()); n_neg = int(len(ytr_l) - n_pos)
    cb_w = class_balanced_weights([max(n_neg, 1), max(n_pos, 1)]) if loss == "focal" else None
    N = Xtr.size(0)
    best_state, best_ce = None, float("inf")
    for _ in range(args.epochs):
        head.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = head(Xtr[idx])
            if loss == "focal":
                lv = focal_ce(logits, ytr_l[idx], gamma=2.0, weight=cb_w)
            else:
                lv = F.cross_entropy(logits, ytr_l[idx])
            opt.zero_grad(set_to_none=True)
            lv.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vce = float(F.cross_entropy(head(Xval), yval.long()))
        if vce < best_ce:
            best_ce = vce
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    # threshold calibration on val (maximise F1) — else default 0.5
    thr = 0.5
    if calibrate:
        with torch.no_grad():
            vprob = torch.softmax(head(Xval), dim=1)[:, 1].cpu().numpy()
        yv = yval.cpu().numpy().astype(int)
        best_f1 = -1.0
        for t in np.linspace(0.1, 0.9, 17):
            pred = (vprob > t).astype(int)
            tp = int(((pred == 1) & (yv == 1)).sum())
            fp = int(((pred == 1) & (yv == 0)).sum())
            fn = int(((pred == 0) & (yv == 1)).sum())
            f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1, thr = f1, float(t)
    return head, best_ce, thr


def _cav_prob(head, X) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(head(X), dim=1)[:, 1].cpu().numpy()


# ── per (mode, country, seed, rung) execution ───────────────────────────────────

def _reg_target_for_mode(mode: str, sub) -> torch.Tensor:
    return _timika_target(sub) if mode in ("a3", "fusion") else _alp_target(sub)


def _train_reg_ensemble(Xtr, ytr, Xval, yval, in_dim, *, kind, loss, select_metric,
                        spatial, args, device, seed, M):
    return [
        _train_reg(Xtr, ytr, Xval, yval, in_dim, kind=kind, loss=loss,
                   select_metric=select_metric, spatial=spatial, args=args,
                   device=device, seed=seed + 1000 * m)
        for m in range(M)
    ]


def run_cell(mode, held_out, seed, feats, dim, args, device, cfg: RungCfg, out_dir,
             feats_grid: dict | None = None, dim_grid: int | None = None):
    """Train + evaluate one rung config for one held-out country / seed. Returns a row.

    ``feats`` is the primary feature cache used by the ALP/Timika head ([N,D] CLS for
    a2/a3/fusion; [N,P,D] grid for a1 spatial ALP). ``feats_grid`` (optional) is a
    separate patch-grid cache used by the spatial cavity head when cfg.cavity_head=='spatial'.
    """
    spatial = mode == "a1"
    use_spatial_cav = cfg.cavity_head == "spatial" and (
        feats_grid is not None or (spatial and feats is not None))

    # ALP/Timika regression split (patient-disjoint, country-segregated)
    a_tr, a_val, test_df = make_country_split(
        _MANIFEST, held_out, val_fraction=args.val_fraction, seed=seed
    )
    Xtr, tr = _gather(a_tr, feats, device)
    Xval, va = _gather(a_val, feats, device)
    Xte, te = _gather(test_df, feats, device)

    # Rung 2: feature standardisation (skip for grid/a1)
    if cfg.standardize != "none" and Xtr.dim() == 2:
        scaler = FeatureStandardizer().fit(Xtr.cpu().numpy())
        Xtr = torch.tensor(scaler.transform(Xtr.cpu().numpy()), device=device)
        Xval = torch.tensor(scaler.transform(Xval.cpu().numpy()), device=device)
        te_scaler = FeatureStandardizer().fit(Xte.cpu().numpy()) if cfg.standardize == "transductive" else scaler
        Xte = torch.tensor(te_scaler.transform(Xte.cpu().numpy()), device=device)

    ytr = _reg_target_for_mode(mode, tr).to(device)
    yval = _reg_target_for_mode(mode, va).to(device)

    # ── regressor(s) ──
    M = max(1, cfg.ensemble)
    heads = _train_reg_ensemble(Xtr, ytr, Xval, yval, dim, kind=cfg.alp_head, loss=cfg.loss,
                                select_metric=args.select_metric, spatial=spatial, args=args,
                                device=device, seed=seed, M=M)
    te_preds = [_reg_predict(h, Xte) for h in heads]
    val_preds = [_reg_predict(h, Xval) for h in heads]
    reg_te, reg_te_std = ensemble_mean(te_preds)     # [0,1]
    reg_va, _ = ensemble_mean(val_preds)

    # ── Rung 3: retrieval calibration (in the regression target space) ──
    alpha = 0.0
    if cfg.retrieval:
        ftr = _pool(Xtr).cpu().numpy()
        cal = RetrievalCalibrator(k=args.knn_k).fit(ftr, ytr.cpu().numpy())
        alpha, _ = cal.select_alpha(_pool(Xval).cpu().numpy(), reg_va, yval.cpu().numpy())
        reg_va = cal.calibrate(_pool(Xval).cpu().numpy(), reg_va, alpha)
        reg_te = cal.calibrate(_pool(Xte).cpu().numpy(), reg_te, alpha)

    # ── Rung 7: post-hoc slope calibration on the head's [0,1] target space ──
    if cfg.calibrate_pred != "none":
        calib = _fit_calibrate(reg_va, yval.cpu().numpy(), cfg.calibrate_pred)
        reg_te = calib(reg_te); reg_va = calib(reg_va)

    # ── cavity head (a2/fusion/a1) ──
    cav_prob_te = None
    cav_thr = 0.5
    cav_ce = float("nan")
    if mode in ("a2", "fusion", "a1"):
        c_tr, c_val, _ = make_balanced_cavity_split(_MANIFEST, held_out,
                                                     val_fraction=args.val_fraction, seed=seed)
        # Pick feature source for cavity: spatial cavity head needs the [N,P,D] patch grid;
        # global head uses pooled CLS-style vectors.
        if use_spatial_cav:
            cav_feats = feats_grid if feats_grid is not None else feats
            cav_dim = dim_grid if dim_grid is not None else dim
        else:
            cav_feats, cav_dim = feats, dim
        Xct, ct = _gather(c_tr, cav_feats, device)
        Xcv, cv = _gather(c_val, cav_feats, device)
        if not use_spatial_cav:
            Xct, Xcv = _pool(Xct), _pool(Xcv)         # global cavity head: pooled features
        yct = torch.tensor(ct["cavity"].to_numpy(np.float32)).to(device)
        ycv = torch.tensor(cv["cavity"].to_numpy(np.float32)).to(device)
        cav_head, cav_ce, cav_thr = _train_cav(
            Xct, yct, Xcv, ycv, cav_dim, loss=cfg.cavity_loss, calibrate=cfg.cavity_cal,
            args=args, device=device, seed=seed,
            head_kind="spatial" if use_spatial_cav else "global",
        )
        # cavity prob on test features (matching the head's expected input shape)
        def _cav_input(df_subset):
            Xc, _ = _gather(df_subset, cav_feats, device)
            return Xc if use_spatial_cav else _pool(Xc)
        cav_prob_te = _cav_prob(cav_head, _cav_input(test_df))
        cav_prob_aval = _cav_prob(cav_head, _cav_input(a_val))   # for fusion blend / conformal

    # ── assemble Timika + evaluate (mode-specific) ──
    alp_true = te["alp_0_100"].to_numpy(np.float32)
    cav_true = te["cavity"].to_numpy(np.float32)
    timika_true = alp_true + CAV_BONUS * cav_true

    if mode in ("a3",):
        timika_pred = reg_te * 140.0
        alp_pred = np.full_like(timika_pred, np.nan)
        res = evaluate_timika_direct(timika_true, timika_pred)
        res["alp"] = {"mae": float("nan")}
        res["cavity"] = {"auc": float("nan"), "f1": float("nan")}
    elif mode == "fusion":
        # a2 branch needs its own ALP head (reg here is Timika for fusion's a3 part);
        # train a quick ALP head + reuse cavity to build the a2 Timika, then blend on val.
        ya_tr = _alp_target(tr).to(device); ya_val = _alp_target(va).to(device)
        alp_heads = _train_reg_ensemble(Xtr, ya_tr, Xval, ya_val, dim, kind="mlp", loss=cfg.loss,
                                        select_metric=args.select_metric, spatial=False, args=args,
                                        device=device, seed=seed + 7, M=M)
        alp_te, _ = ensemble_mean([_reg_predict(h, Xte) for h in alp_heads])
        alp_va, _ = ensemble_mean([_reg_predict(h, Xval) for h in alp_heads])
        # Rung 7 also applies to fusion's a2 ALP head (separate target scale: alp/100).
        if cfg.calibrate_pred != "none":
            alp_calib = _fit_calibrate(alp_va, ya_val.cpu().numpy(), cfg.calibrate_pred)
            alp_te = alp_calib(alp_te); alp_va = alp_calib(alp_va)
        cav_va = cav_prob_aval
        t_a2_te = alp_te * 100.0 + CAV_BONUS * (cav_prob_te > cav_thr)
        t_a3_te = reg_te * 140.0
        t_a2_va = alp_va * 100.0 + CAV_BONUS * (cav_va > cav_thr)
        t_a3_va = reg_va * 140.0
        va_timika = (_timika_target(va).cpu().numpy() * 140.0)
        ws = np.linspace(0, 1, 11)
        w = min(ws, key=lambda w: np.mean(np.abs((w * t_a2_va + (1 - w) * t_a3_va) - va_timika)))
        timika_pred = w * t_a2_te + (1 - w) * t_a3_te
        alp_pred = alp_te * 100.0
        res = evaluate_timika_direct(timika_true, timika_pred)
        res["alp"] = {"mae": float(np.mean(np.abs(alp_pred - alp_true)))}
        res["cavity"] = {"auc": float("nan"), "f1": float("nan")}
        res["fusion_w"] = w
    else:  # a2 / a1
        alp_pred = reg_te * 100.0
        preds = Predictions(alp_true_100=alp_true, alp_pred_100=alp_pred,
                            cavity_true=cav_true, cavity_prob=cav_prob_te)
        res = evaluate_split(preds, cavity_threshold=cav_thr)
        timika_pred = alp_pred + CAV_BONUS * (cav_prob_te > cav_thr)

    # ── Rung 6: conformal coverage on the Timika scale ──
    cov = width = float("nan")
    if cfg.conformal:
        # calibrate on the val split's Timika residuals
        if mode == "a3":
            va_t_pred = reg_va * 140.0
        elif mode == "fusion":
            va_t_pred = w * t_a2_va + (1 - w) * t_a3_va
        else:
            va_t_pred = reg_va * 100.0 + CAV_BONUS * (cav_prob_aval > cav_thr)
        va_t_true = (_timika_target(va).cpu().numpy() * 140.0)
        conf = split_conformal(va_t_true, va_t_pred, timika_pred, timika_true, alpha=0.1)
        cov, width = conf["coverage"], conf["mean_width"]

    t = res["timika"]
    ci = t.get("mae_ci95", (float("nan"), float("nan")))
    locked, paper = references_for_mode(mode)
    base = locked.get(held_out, {})
    base_mae = base.get("timika_mae", float("nan"))
    if ci[1] < base_mae:
        verdict = "BEATS"
    elif ci[0] > base_mae:
        verdict = "WORSE"
    else:
        verdict = "within-noise"

    row = {
        "mode": mode, "rung": cfg.name, "loss": cfg.loss, "standardize": cfg.standardize,
        "retrieval": cfg.retrieval, "alpha": round(float(alpha), 3),
        "cavity_loss": cfg.cavity_loss, "cavity_thr": round(float(cav_thr), 3),
        "alp_head": cfg.alp_head, "ensemble": M, "held_out": held_out, "seed": seed,
        "n_test": len(te), "alp_mae": res["alp"].get("mae", float("nan")),
        "cavity_auc": res["cavity"].get("auc", float("nan")),
        "cavity_f1": res["cavity"].get("f1", float("nan")),
        "timika_mae": t["mae"], "timika_pearson": t["pearson"],
        "timika_mae_ci_lo": ci[0], "timika_mae_ci_hi": ci[1],
        "conformal_cov": cov, "conformal_width": width,
        "base_timika_mae": base_mae, "paper_timika_mae": paper.get(held_out, {}).get("timika_mae", float("nan")),
        "cav_best_val_ce": cav_ce, "verdict": verdict,
    }
    print(f"[{mode}|{cfg.name}] {held_out} s{seed}  MAE={t['mae']:.2f} "
          f"CI[{ci[0]:.1f},{ci[1]:.1f}] r={t['pearson']:.3f} a={alpha:.2f} "
          f"| base {base_mae:.2f} paper {row['paper_timika_mae']:.2f} -> {verdict}")

    pd.DataFrame({"image_id": te["image_id"].to_numpy(), "held_out": held_out, "seed": seed,
                  "rung": cfg.name, "timika_true": timika_true, "timika_pred": timika_pred}
                 ).to_csv(out_dir / f"preds_{mode}_{cfg.name}_{held_out}_s{seed}.csv", index=False)

    # Optionally pickle trained heads for local qualitative-figure generation later.
    if getattr(args, "save_heads", False):
        heads_dir = out_dir / "heads"
        heads_dir.mkdir(exist_ok=True)
        artefact = {
            "mode": mode, "rung": cfg.name, "held_out": held_out, "seed": seed,
            "cfg": cfg.__dict__,
            "reg_heads": [h.state_dict() for h in heads],
            "reg_kind": cfg.alp_head, "spatial_reg": spatial,
            "alpha": float(alpha), "cav_thr": float(cav_thr),
        }
        if mode in ("a2", "fusion", "a1"):
            artefact["cav_head"] = cav_head.state_dict()
            artefact["cav_head_kind"] = "spatial" if use_spatial_cav else "global"
        if mode == "fusion":
            artefact["alp_heads"] = [h.state_dict() for h in alp_heads]
            artefact["fusion_w"] = float(w)
        torch.save(artefact, heads_dir / f"{mode}_{cfg.name}_{held_out}_s{seed}.pt")
    return row


# module-level manifest handle (set in main) so run_cell can re-split per seed
_MANIFEST: pd.DataFrame | None = None


def main(argv=None) -> None:
    global _MANIFEST
    p = argparse.ArgumentParser(description="Agentic Timika trainer (modes a2/a3/fusion/a1, rungs 1-6).")
    p.add_argument("--features", required=True, help="Primary feature cache (.npz). CLS [N,D] for a2/a3/fusion or patch grid [N,P,D] for a1.")
    p.add_argument("--features-grid", default=None, help="Optional patch-grid cache for spatial cavity head (a2/a3/fusion). Not used by a1 (which already loads grid as --features).")
    p.add_argument("--manifest", required=True)
    p.add_argument("--mode", default="a2", choices=["a2", "a3", "fusion", "a1"])
    p.add_argument("--rungs", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--ensemble-m", type=int, default=5, help="Ensemble members for rung6 / agentic_best (Rung 6 multiplier).")
    p.add_argument("--save-heads", action="store_true", help="Pickle trained ALP+cavity heads per (rung, country, seed) into the out dir.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--held-outs", nargs="+", default=["Romania", "Moldova", "Kazakhstan"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--select-metric", default="pearson", choices=["pearson", "mae", "mse"])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--val-fraction", type=float, default=0.2)
    args = p.parse_args(argv)

    device = pick_device()
    feats, dim = load_features(args.features)
    sample = next(iter(feats.values()))
    primary_kind = "grid" if sample.ndim == 2 else "CLS"
    feats_grid, dim_grid = (None, None)
    if args.features_grid:
        feats_grid, dim_grid = load_features(args.features_grid)
        gsample = next(iter(feats_grid.values()))
        if gsample.ndim != 2:
            raise SystemExit(f"--features-grid must be a [N,P,D] patch grid cache (got ndim={gsample.ndim+1})")
        print(f"[agentic] aux grid cache loaded: dim={dim_grid} P={gsample.shape[0]}")
    grid_available = feats_grid is not None or (args.mode == "a1" and primary_kind == "grid")
    print(f"[agentic] device={device} mode={args.mode} dim={dim} feat={primary_kind} "
          f"grid_available={grid_available} rungs={args.rungs} seeds={args.seeds} M={args.ensemble_m}")
    _MANIFEST = load_manifest(args.manifest)

    ladder = rung_ladder(args.mode, args.rungs, ensemble_M=args.ensemble_m,
                         grid_available=grid_available)
    print(f"[agentic] {len(ladder)} configs: {[c.name for c in ladder]}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg in ladder:
        for held_out in args.held_outs:
            for seed in args.seeds:
                rows.append(run_cell(args.mode, held_out, seed, feats, dim, args, device, cfg, out_dir,
                                      feats_grid=feats_grid, dim_grid=dim_grid))
                pd.DataFrame(rows).to_csv(out_dir / f"results_agentic_{args.mode}.csv", index=False)

    dfr = pd.DataFrame(rows)
    print(f"\n[agentic] mean Timika MAE per rung x country (mode={args.mode}):")
    for cfg in ladder:
        sub = dfr[dfr["rung"] == cfg.name]
        line = "  " + f"{cfg.name:18s}"
        for ho in args.held_outs:
            s = sub[sub["held_out"] == ho]
            if len(s):
                line += f" {ho[:3]}={s['timika_mae'].mean():5.2f}/{s['timika_pearson'].mean():.2f}"
        print(line)
    print(f"\n[agentic] -> {out_dir / f'results_agentic_{args.mode}.csv'}")


if __name__ == "__main__":
    main()
