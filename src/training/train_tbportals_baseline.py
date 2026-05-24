"""Train the TB Portals ALP + cavity model under Kantipudi's country-segregated
protocol.

Day-1 default (``--num-experts 1``, no ``--use-dann``) reproduces Kantipudi A2:
one DenseNet121 ALP regressor (MSE) + one cavity classifier (BCE), NAdam 1e-3,
30 epochs, early stopping on validation ALP-MAE. The same script also trains the
Day-2 variants (``--num-experts K``, ``--use-dann``) so the ablation ladder shares
one code path. Checkpoints are saved per epoch for T4-session resumability.

Example (one held-out country, 3 seeds)::

    python -m src.training.train_tbportals_baseline \
        --manifest data/processed/tbportals_manifest.csv \
        --held-outs Romania --seeds 0 1 2 \
        --out-dir checkpoints/tbportals/baseline
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.components.component1_dann import compute_dann_lambda
from src.components.component_alp_moe import ALPMoEConfig, ALPMoEModel, load_balancing_loss
from src.core.device import describe_device, pick_device
from src.core.seed import seed_everything
from src.data.tbportals import load_manifest, make_country_split
from src.data.tbportals_dataset import TBPortalsDataset, build_country_index
from src.evaluation.eval_tbportals import evaluate_split, predict, regression_metrics


def build_loaders(train_df, val_df, test_df, country_to_idx, args):
    common = dict(
        country_to_idx=country_to_idx,
        crops_dir=args.crops_dir,
        use_lung_crop=args.use_lung_crop,
    )
    train_ds = TBPortalsDataset(train_df, train=True, **common)
    val_ds = TBPortalsDataset(val_df, train=False, **common)
    test_ds = TBPortalsDataset(test_df, train=False, **common)
    loader = lambda ds, shuffle: DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)


def cavity_pos_weight(train_df) -> float:
    n_pos = int(train_df["cavity"].sum())
    n_neg = len(train_df) - n_pos
    return float(n_neg / max(1, n_pos))


def run_single(df, held_out: str, seed: int, args, device) -> dict:
    seed_everything(seed)
    train_df, val_df, test_df = make_country_split(
        df, held_out, val_fraction=args.val_fraction, seed=seed
    )
    country_to_idx = build_country_index(train_df)
    train_loader, val_loader, test_loader = build_loaders(
        train_df, val_df, test_df, country_to_idx, args
    )

    cfg = ALPMoEConfig(
        num_experts=args.num_experts,
        gate_temperature=args.gate_temperature,
        use_dann=args.use_dann,
        num_countries=len(country_to_idx),
        pretrained=not args.no_pretrained,
    )
    model = ALPMoEModel(cfg).to(device)

    optimizer = torch.optim.NAdam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    pos_weight = torch.tensor([cavity_pos_weight(train_df)], device=device)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{held_out}_seed{seed}_K{args.num_experts}_dann{int(args.use_dann)}"
    best_path = out_dir / f"{tag}_best.pt"
    last_path = out_dir / f"{tag}_last.pt"

    start_epoch = 0
    best_val_mae = float("inf")
    if args.resume and last_path.is_file():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mae = ckpt.get("best_val_mae", float("inf"))
        print(f"[train] resumed {tag} from epoch {start_epoch} (best_val_mae={best_val_mae:.3f})")

    patience_left = args.patience
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # MoE phase ordering: warm experts under a uniform gate first, then
        # train the gate. Guards against routing a randomly-initialized gate.
        warming = args.num_experts > 1 and epoch < args.gate_warm_epochs
        model.set_gate_trainable(not warming)
        dann_lambda = (
            compute_dann_lambda(epoch, args.dann_ramp, args.dann_max_lambda)
            if args.use_dann
            else 0.0
        )

        running = 0.0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            alp_true = batch["alp"].to(device)
            cav_true = batch["cavity"].to(device)
            country = batch["country_idx"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(images, dann_lambda=dann_lambda, uniform_gate=warming)
                # Normalise ALP loss to [0,1] scale so it is comparable to
                # BCE (~0.3-0.7). Without this, MSE on [0,100] is ~300× larger
                # than BCE, starving the cavity head of gradient.
                loss_alp = F.mse_loss(out.alp / 100.0, alp_true / 100.0)
                loss_cav = F.binary_cross_entropy_with_logits(
                    out.cavity_logit, cav_true, pos_weight=pos_weight
                )
                loss = loss_alp + args.cavity_alpha * loss_cav
                if args.use_dann and out.country_logit is not None:
                    valid = country >= 0
                    if valid.any():
                        loss = loss + args.dann_beta * F.cross_entropy(
                            out.country_logit[valid], country[valid]
                        )
                if out.gate_weights is not None and not warming:
                    loss = loss + args.lb_weight * load_balancing_loss(out.gate_weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * images.shape[0]

        val_preds = predict(model, val_loader, device)
        val_mae = regression_metrics(val_preds.alp_true_100, val_preds.alp_pred_100)["mae"]
        try:
            val_cav_auc = roc_auc_score(val_preds.cavity_true, val_preds.cavity_prob)
        except ValueError:
            val_cav_auc = float("nan")
        train_loss = running / max(1, len(train_loader.dataset))
        print(
            f"[train] {tag} epoch {epoch:02d} loss={train_loss:.4f} "
            f"val_ALP_MAE={val_mae:.3f} val_cavity_AUC={val_cav_auc:.3f} "
            f"lambda={dann_lambda:.2f}"
            f"{' (gate-warm)' if warming else ''}"
        )

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "epoch": epoch, "best_val_mae": best_val_mae, "config": dataclasses.asdict(cfg)},
            last_path,
        )
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            patience_left = args.patience
            torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg),
                        "epoch": epoch, "val_mae": val_mae}, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[train] {tag} early stop at epoch {epoch} (best val ALP MAE={best_val_mae:.3f})")
                break

    # Final test eval with the best checkpoint.
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model"])
    test_preds = predict(model, test_loader, device)
    metrics = evaluate_split(test_preds, cavity_threshold=args.cavity_threshold)

    row = {
        "held_out": held_out, "seed": seed, "num_experts": args.num_experts,
        "use_dann": int(args.use_dann), "n_test": len(test_df),
        "best_val_alp_mae": round(best_val_mae, 3),
        "timika_mae": round(metrics["timika"]["mae"], 3),
        "timika_mae_pct": round(metrics["timika"]["mae_pct"], 3),
        "timika_pearson": round(metrics["timika"]["pearson"], 3),
        "alp_mae": round(metrics["alp"]["mae"], 3),
        "cavity_auc": round(metrics["cavity"]["auc"], 3),
        "cavity_f1": round(metrics["cavity"]["f1"], 3),
        "best_ckpt": str(best_path),
    }
    print(f"[train] {tag} TEST -> {json.dumps(row)}")
    _append_results(out_dir / "results.csv", row)
    return row


def _append_results(path: Path, row: dict) -> None:
    import pandas as pd

    df = pd.DataFrame([row])
    if path.is_file():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train TB Portals ALP+cavity model.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--held-outs", nargs="+", default=["Romania", "Moldova", "Kazakhstan"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--cavity-alpha", type=float, default=0.1,
                   help="Weight on cavity BCE loss. With ALP MSE normalised to [0,1], "
                        "default 0.1 gives roughly equal gradient contribution from both heads.")
    p.add_argument("--cavity-threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--crops-dir", default=None)
    p.add_argument("--use-lung-crop", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--resume", action="store_true")
    # MoE / DANN (Day 2; defaults give the Day-1 single-head baseline)
    p.add_argument("--num-experts", type=int, default=1)
    p.add_argument("--gate-temperature", type=float, default=1.0)
    p.add_argument("--gate-warm-epochs", type=int, default=5)
    p.add_argument("--lb-weight", type=float, default=0.01)
    p.add_argument("--use-dann", action="store_true")
    p.add_argument("--dann-beta", type=float, default=0.5)
    p.add_argument("--dann-max-lambda", type=float, default=1.0)
    p.add_argument("--dann-ramp", type=int, default=10)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    device = pick_device()
    print(f"[train] device={describe_device(device)}")
    df = load_manifest(args.manifest)
    rows = []
    for held_out in args.held_outs:
        for seed in args.seeds:
            rows.append(run_single(df, held_out, seed, args, device))
    print(f"\n[train] DONE {len(rows)} run(s). Results -> {Path(args.out_dir) / 'results.csv'}")


if __name__ == "__main__":
    main()
