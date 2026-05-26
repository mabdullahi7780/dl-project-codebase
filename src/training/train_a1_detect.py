"""Strict replication of Kantipudi et al. (JIIM 2024), approach A1.

A1 is the explainable, detection-based path:

  ALP    = |lesion_boxes  ∩  lung_mask| / |lung_mask| * 100
           lesion_boxes : YOLOv5 trained on TBX11K (passed via --yolo-weights)
           lung_mask    : MedSAM (same module as the lung-crop step)
  cavity = DenseNet121 cavity classifier (identical to A2; reused here)
  Timika = ALP + 40 * cavity

The cavity classifier is trained per held-out country (paper Table 2 balanced
split). The YOLO detector is country-agnostic and trained once on TBX11K, so it
is loaded, not retrained, here. ALP needs the full-resolution image (lesions are
detected on the original CXR — the paper found cropping HURTS detection), so we
run MedSAM + YOLO per test image rather than reusing the 224 crops.

Example::

    python -m src.training.train_a1_detect \
        --manifest local_work/data/processed/tbportals_manifest_paper.csv \
        --crops-dir local_work/data/processed/tbportals_crops \
        --yolo-weights /kaggle/working/tbx11k_yolo/best.pt \
        --medsam-ckpt <medsam.pth> --lung-decoder-ckpt <decoder.pt> \
        --out-dir checkpoints/paper_a1 --seeds 0 1 2 \
        --batch-size 60 --accum-steps 5 --cavity-no-lung-crop
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.components.component4_lung import Component4MedSAM
from src.data.tbportals import load_manifest, make_country_split
from src.evaluation.eval_tbportals import (
    KANTIPUDI_A1,
    Predictions,
    evaluate_split,
    regression_metrics,
)
from src.training.train_baseline_paper import (
    _loader,
    _make_dataset,
    make_balanced_cavity_split,
    pick_device,
    seed_everything,
    train_cavity,
)

MEDSAM_SIZE = 1024


def _load_1024_3ch(path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    img = Image.open(path).convert("L")
    w, h = img.size
    arr = np.asarray(img.resize((MEDSAM_SIZE, MEDSAM_SIZE)), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None].repeat(1, 3, 1, 1)
    return t, (w, h)


@torch.no_grad()
def predict_cavity_probs(model, test_df, args, device, crops_dir) -> np.ndarray:
    loader = _loader(_make_dataset(test_df, False, crops_dir), args.batch_size, False, args.num_workers)
    model.eval()
    probs = []
    for b in loader:
        x = b["image"].to(device)
        probs.append(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


@torch.no_grad()
def detection_alp(image_path: str, medsam, yolo, device, conf: float) -> float:
    """ALP% = |lesion boxes ∩ lung mask| / |lung mask| * 100, all in 1024-space."""
    x, (w, h) = _load_1024_3ch(image_path)
    lung = medsam.predict_masks(x.to(device)).lung_mask_1024[0, 0].cpu().numpy() > 0.5
    lung_area = int(lung.sum())
    if lung_area == 0:
        return 0.0
    res = yolo.predict(image_path, conf=conf, verbose=False)[0]
    lesion = np.zeros_like(lung, dtype=bool)
    sx, sy = MEDSAM_SIZE / w, MEDSAM_SIZE / h
    for box in res.boxes.xyxy.cpu().numpy():
        x0, y0, x1, y1 = box
        X0, Y0 = max(0, int(x0 * sx)), max(0, int(y0 * sy))
        X1, Y1 = min(MEDSAM_SIZE, int(x1 * sx)), min(MEDSAM_SIZE, int(y1 * sy))
        lesion[Y0:Y1, X0:X1] = True
    inter = int((lesion & lung).sum())
    return 100.0 * inter / lung_area


def run_country(df, held_out, seed, args, device, cav_crops, out_dir, medsam, yolo):
    seed_everything(seed)
    print(f"\n===== A1  {held_out}  seed={seed} =====")

    # 1) cavity classifier (identical protocol to A2)
    cav_train, cav_val, test_df = make_balanced_cavity_split(
        df, held_out, val_fraction=args.val_fraction, seed=seed
    )
    print(f"[A1][CAV] train={len(cav_train)} val={len(cav_val)} test={len(test_df)}")
    cav_model, cav_best = train_cavity(cav_train, cav_val, args, device, cav_crops)
    torch.save(cav_model.state_dict(), out_dir / f"cavity_a1_{held_out}_seed{seed}.pt")
    cavity_prob = predict_cavity_probs(cav_model, test_df, args, device, cav_crops)
    del cav_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2) detection-based ALP (YOLO lesions ∩ MedSAM lung), per test image
    print(f"[A1][ALP] running MedSAM+YOLO over {len(test_df)} test images ...")
    alp_pred = np.array([detection_alp(p, medsam, yolo, device, args.yolo_conf)
                         for p in test_df["image_path"]], dtype=np.float32)
    alp_true = test_df["alp_0_100"].to_numpy(dtype=np.float32)
    cavity_true = test_df["cavity"].to_numpy(dtype=np.float32)

    preds = Predictions(
        alp_true_100=alp_true, alp_pred_100=alp_pred,
        cavity_true=cavity_true, cavity_prob=cavity_prob,
    )
    res = evaluate_split(preds, cavity_threshold=args.cavity_threshold)
    ref = KANTIPUDI_A1.get(held_out, {})
    row = {
        "approach": "A1", "held_out": held_out, "seed": seed, "n_test": len(test_df),
        "alp_mae": res["alp"]["mae"], "alp_pred_mean": float(alp_pred.mean()),
        "cavity_auc": res["cavity"]["auc"], "cavity_f1": res["cavity"]["f1"],
        "timika_mae": res["timika"]["mae"], "timika_mae_pct": res["timika"]["mae_pct"],
        "timika_pearson": res["timika"]["pearson"], "cav_best_val_ce": cav_best,
    }
    print(f"[RESULT] A1 {held_out} seed={seed}  "
          f"ALP_MAE={row['alp_mae']:.2f}  cavity_AUC={row['cavity_auc']:.3f}  "
          f"Timika_MAE={row['timika_mae']:.2f} ({row['timika_mae_pct']:.2f}%) | their "
          f"{ref.get('timika_mae', float('nan')):.2f}   "
          f"Pearson={row['timika_pearson']:.2f} | their {ref.get('timika_pearson', float('nan')):.2f}")
    return row


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Strict Kantipudi A1 replication (detection-based ALP).")
    p.add_argument("--manifest", required=True)
    p.add_argument("--crops-dir", default=None, help="Lung-crop dir for the cavity classifier input.")
    p.add_argument("--yolo-weights", required=True, help="Trained YOLOv5 .pt (TBX11K lesion detector).")
    p.add_argument("--medsam-ckpt", required=True)
    p.add_argument("--lung-decoder-ckpt", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--held-outs", nargs="+", default=["Romania", "Moldova", "Kazakhstan"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=300)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--cavity-threshold", type=float, default=0.5)
    p.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO detection confidence threshold.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-lung-crop", action="store_true")
    p.add_argument("--cavity-no-lung-crop", action="store_true",
                   help="Train the cavity classifier on whole images (match the locked A2 config).")
    p.add_argument("--amp", action="store_true", default=True)
    args = p.parse_args(argv)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = pick_device()
    print(f"[paper-a1] device={device}")
    df = load_manifest(args.manifest)

    base_crops = None if args.no_lung_crop else args.crops_dir
    cav_crops = None if args.cavity_no_lung_crop else base_crops
    print(f"[paper-a1] cavity input = {('lung-crop ' + str(cav_crops)) if cav_crops else 'WHOLE image'}")

    # Load the lesion detector + lung segmenter once (reused across countries).
    from ultralytics import YOLO  # imported lazily so the module compiles without it
    yolo = YOLO(args.yolo_weights)
    medsam = Component4MedSAM(backend="medsam", checkpoint_path=args.medsam_ckpt).to(device)
    if medsam.active_backend != "medsam":
        raise SystemExit("[paper-a1] MedSAM backend unavailable; cannot compute detection ALP.")
    if args.lung_decoder_ckpt:
        medsam.load_trained_decoder(args.lung_decoder_ckpt)
    medsam.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for held_out in args.held_outs:
        for seed in args.seeds:
            rows.append(run_country(df, held_out, seed, args, device, cav_crops, out_dir, medsam, yolo))
            pd.DataFrame(rows).to_csv(out_dir / "results_a1.csv", index=False)

    dfr = pd.DataFrame(rows)
    print("\n[paper-a1] mean +/- std across seeds (vs Kantipudi A1):")
    for ho in args.held_outs:
        s = dfr[dfr["held_out"] == ho]
        if len(s) == 0:
            continue
        ref = KANTIPUDI_A1.get(ho, {})
        print(f"  {ho:12s} ALP_MAE={s['alp_mae'].mean():5.2f}  cavity_AUC={s['cavity_auc'].mean():.3f}  "
              f"Timika_MAE={s['timika_mae'].mean():5.2f}+/-{s['timika_mae'].std():.2f} "
              f"(their {ref.get('timika_mae', float('nan')):.2f})  "
              f"Pearson={s['timika_pearson'].mean():.2f} (their {ref.get('timika_pearson', float('nan')):.2f})")
    print(f"\n[paper-a1] results -> {out_dir / 'results_a1.csv'}")


if __name__ == "__main__":
    main()
