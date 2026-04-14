# Component 4 (Lung Mask) Fine-Tuning

Trains the **MedSAM ViT-B mask decoder** on paired chest X-ray images and lung
masks. The image encoder and prompt encoder stay frozen. Loss is BCE + Dice.
Best checkpoint is chosen by **highest validation Dice** — that matches the
downstream metric and is less noisy than raw loss, which BCE can dominate.

## Expected dataset layout

Montgomery (raw distribution):

```
<montgomery_root>/
  CXR_png/<id>.png
  ManualMask/leftMask/<id>.png
  ManualMask/rightMask/<id>.png
```

Shenzhen (actual layout):

```
<shenzhen_root>/
  images/<id>.png
  mask/<id>_mask.png
```

Note: Shenzhen mask filenames have a `_mask` suffix (e.g. `CHNCXR_0001_0_mask.png`)
that the manifest builder strips automatically when matching to image stems.

For Montgomery you have two options:
1. **Pre-merge** left+right into one binary PNG with the script below (recommended).
2. **On-the-fly merge**: put both paths in `mask_path` separated by `|`.

## Manifest format

CSV with header:

```
sample_id,image_path,mask_path,dataset,split
montgomery/MCUCXR_0001_0,/data/montgomery/CXR_png/MCUCXR_0001_0.png,/data/montgomery/MergedMask/MCUCXR_0001_0.png,montgomery,train
shenzhen/CHNCXR_0001_0,/data/shenzhen/images/CHNCXR_0001_0.png,/data/shenzhen/mask/CHNCXR_0001_0_mask.png,shenzhen,val
montgomery/MCUCXR_0002_0,/data/montgomery/CXR_png/MCUCXR_0002_0.png,/data/montgomery/ManualMask/leftMask/MCUCXR_0002_0.png|/data/montgomery/ManualMask/rightMask/MCUCXR_0002_0.png,montgomery,train
```

- `split` ∈ `{train, val, test}`.
- `dataset` ∈ `{montgomery, shenzhen, tbx11k, nih_cxr14}` (only those with masks are usable here).
- `mask_path` may contain one path, or two paths separated by `|` for on-the-fly union merging.
- Paths can be absolute or relative to the repo root.

## Commands

### 1. Pre-merge Montgomery masks (one-time)

```bash
python3.11 -m scripts.merge_montgomery_lung_masks \
  --montgomery-root "/Volumes/Toshiba HDR/TB_DATA/raw/montgomery" \
  --out-dir         "/Volumes/Toshiba HDR/TB_DATA/raw/montgomery/MergedMask"
```

### 2. Build the manifest

```bash
python3.11 -m scripts.build_component4_manifest \
  --montgomery-images "/Volumes/Toshiba HDR/TB_DATA/raw/montgomery/CXR_png" \
  --montgomery-masks  "/Volumes/Toshiba HDR/TB_DATA/raw/montgomery/MergedMask" \
  --shenzhen-images   "/Volumes/Toshiba HDR/TB_DATA/raw/shenzhen/images" \
  --shenzhen-masks    "/Volumes/Toshiba HDR/TB_DATA/raw/shenzhen/mask" \
  --val-frac 0.15 --test-frac 0.10 \
  --out data/splits/component4_manifest.csv
```

### 3. Dry-run (local MPS / CPU)

Validates manifest parsing, dataset shapes, and one forward pass with a mock
or real backend — no training, no checkpoint write.

```bash
python3.11 -m src.training.train_component4_lung \
  --config configs/component4_lung.yaml \
  --dry-run
```

### 4. Train

```bash
python3.11 -m src.training.train_component4_lung \
  --config configs/component4_lung.yaml
```

Outputs under `checkpoints/component4/` (configurable):
- `component4_mask_decoder.pt` — best by val Dice
- `last_component4_mask_decoder.pt` — most recent epoch
- `training_history.jsonl` — per-epoch train/val metrics

### 5. Resume

```bash
python3.11 -m src.training.train_component4_lung \
  --config configs/component4_lung.yaml \
  --resume checkpoints/component4/last_component4_mask_decoder.pt
```

Or set `training.resume_checkpoint` in the YAML.

### 6. Evaluate trained decoder via batch inference

After training, point Component 4 at the fine-tuned decoder by loading its
state dict on top of the MedSAM checkpoint. Easiest path: keep the MedSAM
backbone checkpoint at `checkpoints/medsam/medsam_vit_b.pth` and overlay the
decoder weights (see `Component4MedSAM.load_decoder_state_dict`). Then run:

```bash
python3.11 -m src.app.batch_infer \
  --csv   data/splits/eval_manifest.csv \
  --outdir outputs/component4_eval \
  --config configs/baseline.yaml
```

## Running on Kaggle

Kaggle free tier (T4 16GB, no internet by default):

1. **Upload** these as a Kaggle Dataset (or attach existing ones):
   - This repo source tree.
   - `checkpoints/medsam/medsam_vit_b.pth`.
   - Montgomery + Shenzhen images and masks (or pre-built manifest).
2. In the notebook, install `segment_anything` (internet toggle ON for the
   install cell, or use a pre-built wheel dataset):
   ```
   !pip install -q git+https://github.com/facebookresearch/segment-anything.git
   ```
3. Set env vars pointing at the Kaggle input dirs and run:
   ```bash
   export COMPONENT4_TRAIN_MANIFEST=/kaggle/input/tb-manifests/component4_manifest.csv
   export COMPONENT4_VAL_MANIFEST=$COMPONENT4_TRAIN_MANIFEST
   export COMPONENT4_SAVE_DIR=/kaggle/working/checkpoints/component4
   python3.11 -m src.training.train_component4_lung --config configs/component4_lung.yaml --dry-run
   python3.11 -m src.training.train_component4_lung --config configs/component4_lung.yaml
   ```
4. Keep `batch_size: 2` (T4 OOMs at 4 with 1024² inputs). `amp: true` is
   honoured only on CUDA.

## Files to upload to Kaggle

- Repo source (or a Kaggle Dataset wrapping it).
- `checkpoints/medsam/medsam_vit_b.pth` (MedSAM base weights).
- Montgomery & Shenzhen images + masks (or pre-merged Montgomery masks).
- A pre-built `component4_manifest.csv` with absolute Kaggle-side paths.

## Troubleshooting

- **Whole-image mask (Montgomery)**: confirms the base MedSAM decoder is
  under-fit for Montgomery — exactly what this fine-tune fixes.
- **OOM on Kaggle**: drop `batch_size` to 1 and set `num_workers: 0`.
- **MPS crashes**: stay on CPU dry-run; real training belongs on CUDA.
- **AMP warnings on non-CUDA**: expected — AMP is silently disabled.
