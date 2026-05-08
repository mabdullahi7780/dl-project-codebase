# Domain-Adaptive Mixture-of-Experts for Automated Tuberculosis Severity Scoring from Chest X-Rays

**Course:** Deep Learning (Semester 6)  
**Author:** M. Abdullah Irfan  
**Supervisor:** Dr. Murtaza Taj

---

## Overview

This repository implements an end-to-end pipeline that takes a raw chest X-ray (CXR) image and produces a **Timika severity score** — the clinical metric used in Papua New Guinea field trials to stratify TB patients for treatment decisions. The Timika score is defined as:

```
Timika Score = ALP + 40 × cavity_flag
```

where **ALP** (Affected Lung Percentage) is the fraction of the lung area covered by TB lesions, and `cavity_flag` indicates radiological cavitation (+40 points).

No prior published system automates this end-to-end scoring across multiple heterogeneous datasets. This project is the first to do so.

---

## Project Structure: Baseline → Improvement 1 → Improvement 2

The project is organized as three progressive stages, each building on the previous.

### Baseline — Existing SOTA

The starting reference point is the **TBX11K dataset paper** (Liu et al., 2020), which reports TB detection AUC of **0.958** on TBX11K. This paper, and the broader published literature, focus on **binary TB detection** — they do not produce lesion masks, do not estimate ALP, and do not compute Timika severity scores.

**Gap:** No system in the literature automates Timika scoring from raw CXR images.

---

### Improvement 1 — Multi-Dataset TB Detection + Lung Segmentation Pipeline

We implement a full baseline pipeline:

```
C0 (QC) → C1 (Domain-Adaptive Encoder) → C2 (TB Classifier) → C4 (Lung Segmentation)
       → Grad-CAM Lesion Proposer → C7 (Heuristic Refinement) → C8 (Timika Scoring)
```

**Key contributions in Improvement 1:**

| Component | What it does | Novel aspect |
|---|---|---|
| **C1** MedSAM + LoRA + DANN | Extracts domain-invariant image embeddings | DANN adversarial training on ViT-B with LoRA adapters — generalizes across 4 datasets |
| **C2** TXV DenseNet + TB Head | Binary TB classifier | Trainable routing head + binary TB logit on top of frozen DenseNet121 |
| **C4** MedSAM Lung Decoder | Lung mask prediction | Fine-tuned MedSAM decoder on Montgomery + Shenzhen GT masks |
| **Lesion Proposer** | Coarse lesion mask | Grad-CAM from TB-mimic pathology classes, Otsu-thresholded within lung |
| **C8** Timika Scoring | ALP + severity label | First automated Timika score from CXR |

**Improvement 1 Results:**

| Metric | Our System | SOTA Reference |
|---|---|---|
| Lung Dice — Shenzhen | **0.959** | MedSAM zero-shot ~0.85 |
| Lung Dice — Montgomery | **0.886** | MedSAM zero-shot ~0.85 |
| TB AUROC — TBX11K | **0.983** | Liu et al. 2020: 0.958 |
| TB AUROC — Shenzhen | **0.851** | — |
| TB AUROC — Overall | **0.984** | — |
| Timika AUROC — TBX11K | **0.774** | Not reported in literature |
| Timika AUROC — Shenzhen | 0.555 | Not reported in literature |

---

### Improvement 2 — Mixture-of-Experts Lesion Decoder

We replace the Grad-CAM lesion proposer with a **Mixture-of-Experts (MoE)** decoder:

```
C0 → C1 → C2 → C4 → C3 (Routing Gate) → C5 (4 Expert Decoders) → C6 (Fusion)
                   → C7 (Boundary Critic + Reprompt Refiner) → C8 (Cavity-Aware Timika)
```

The four expert decoders each specialize in a distinct TB pathology pattern:

- **Expert 1 — Consolidation:** Large opaque airspace filling
- **Expert 2 — Cavity:** Ring-enhancing thin-walled lesions (triggers +40 Timika points)
- **Expert 3 — Fibrosis:** Reticular/linear scarring patterns
- **Expert 4 — Nodule:** Small focal densities

A learned **routing gate** (C3) assigns each image a soft mixture of the four experts based on visual features and domain context. A **boundary critic** (ResNet18) validates mask anatomical plausibility, and a **reprompt refiner** re-queries the fibrosis expert at uncertain boundaries.

**Improvement 2 Results (MoE vs Improvement 1 Baseline):**

| Metric | Baseline (I1) | MoE (I2) | Change |
|---|---|---|---|
| Timika AUROC — Shenzhen | 0.555 | **0.741** | **+18.6 pp** |
| ALP mean — NIH (healthy) | 38.2% | **24.6%** | **−13.6 pp** |
| ALP mean — Montgomery | 59.8% | **22.1%** | **−37.7 pp** |
| ALP mean — Shenzhen | 47.0% | **37.8%** | −9.2 pp |
| Cavity detection (Shenzhen TB+) | 0% | Active | Expert 2 enabled |

The MoE's adaptive Otsu threshold (floor=0.5, ceil=0.8) prevents experts from over-firing on healthy tissue, dramatically reducing false-positive lesion area on NIH-CXR14 images.

---

## Datasets

| Dataset | Images | TB+ | Role |
|---|---|---|---|
| Montgomery County | 138 | ~70 | Train + eval; lung GT masks available |
| Shenzhen Hospital | 662 | ~336 | Train + eval; main Timika benchmark |
| TBX11K | 8,976 | 1,200 | Train + eval; bounding box annotations |
| NIH-CXR14 | 112,120 | 0* | Healthy ALP calibration only |

*NIH-CXR14 has no TB/non-TB labels; used to measure false-positive lesion rate.

All datasets are publicly available on Kaggle. See `configs/paths.yaml` for expected directory structure and `.env.example` for environment variable setup.

---

## Repository Structure

```
.
├── configs/                    # All training and inference hyperparameters
│   ├── baseline.yaml           # Baseline pipeline config
│   ├── moe.yaml                # MoE pipeline config (gate_only, load_balance, etc.)
│   ├── component1_dann.yaml    # C1 DANN training
│   ├── component2_txv.yaml     # C2 TB head training
│   ├── component4_lung.yaml    # C4 lung decoder training
│   └── paths.yaml              # Dataset root paths
│
├── src/
│   ├── app/
│   │   └── infer.py            # End-to-end inference entry point (baseline + MoE)
│   ├── components/             # One module per pipeline component (C0–C10)
│   ├── training/               # Standalone training scripts
│   ├── evaluation/             # Evaluation logic (baseline_eval, moe_eval)
│   ├── data/                   # Dataset classes and preprocessing
│   ├── utils/                  # Morphology, checkpoints, visualization
│   └── core/                   # Types, device, seed, constants
│
├── scripts/
│   ├── kaggle_moe_train.py     # Kaggle bootstrap: runs all 4 MoE training phases
│   ├── kaggle_baseline_eval.py # Kaggle bootstrap: runs baseline evaluation
│   ├── cache_moe_embeddings.py # Phase 0: build embedding cache for MoE training
│   └── ...
│
├── notebooks/
│   ├── train_component4_sprint1.ipynb   # Sprint 1: lung segmentation training
│   ├── train_component2_sprint2.ipynb   # Sprint 2: TB head training
│   ├── train_moe_sprint3.ipynb          # Sprint 3: full MoE training (paper Figure 2)
│   ├── eval_baseline.ipynb              # Improvement 1 evaluation results
│   ├── eval_moe.ipynb                   # Improvement 2 evaluation results + figures
│   └── component*.ipynb                 # Component development notebooks
│
├── tests/                      # 20 unit/integration tests (pytest)
├── checkpoints/                # Trained model weights
│   ├── component1/             # LoRA + DANN adapters
│   ├── component2/             # TB routing head
│   ├── component4/             # Lung mask decoder
│   └── component_moe/          # Expert decoders, gate, boundary critic
├── outputs/                    # Sample inference outputs (demo)
├── plan.md                     # Full architectural design brief
├── sprint_plan.md              # Sprint history with confirmed metrics
└── requirements.txt
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mabdullahi7780/dl-project-codebase.git
cd dl-project-codebase

# Install dependencies (Python 3.11, PyTorch recommended)
pip install -r requirements.txt

# Copy environment template and fill in dataset paths
cp .env.example .env
```

---

## Running the Pipeline

### End-to-End Inference (Baseline)

```bash
python -m src.app.infer \
  --image path/to/cxr.png \
  --dataset shenzhen \
  --outdir outputs/my_run \
  --config configs/baseline.yaml
```

### End-to-End Inference (MoE — requires trained checkpoints)

```bash
python -m src.app.infer \
  --image path/to/cxr.png \
  --dataset shenzhen \
  --outdir outputs/my_run \
  --config configs/moe.yaml
```

### Training

#### Sprint 1 — Lung Segmentation (Component 4)

```bash
python -m src.training.train_component4_lung --config configs/component4_lung.yaml
```

#### Sprint 2 — TB Classification Head (Component 2)

```bash
python -m src.training.train_component2_txv --config configs/component2_txv.yaml
```

#### Sprint 3 — MoE Training (four phases, must run in order)

```bash
# Phase 0: Build embedding cache (PREREQUISITE — do not skip)
python scripts/cache_moe_embeddings.py --config configs/moe.yaml --cache-dir /path/to/cache

# Phase 1: Pretrain expert decoders independently
python -m src.training.train_experts --config configs/moe.yaml --cache-dir /path/to/cache --all

# Phase 2: Train routing gate (experts frozen, gate_only=true)
python -m src.training.train_moe_joint --config configs/moe.yaml --cache-dir /path/to/cache

# Phase 3: Train boundary critic (optional)
python -m src.training.train_boundary_critic --config configs/moe.yaml --cache-dir /path/to/cache

# OR: run all phases with the Kaggle bootstrap script
python scripts/kaggle_moe_train.py --mode full --phase all
```

> **Important:** Phase 0 must run before Phase 1. Phases must not be skipped. If any supervision hyperparameter changes (threshold, weights), Phase 0 must be re-run to rebuild the cache.

### Evaluation

```bash
# Run on Kaggle using the provided notebooks:
# notebooks/eval_baseline.ipynb  →  Improvement 1 metrics
# notebooks/eval_moe.ipynb       →  Improvement 2 metrics + figures
```

### Tests

```bash
pytest
```

---

## Notebooks Guide

| Notebook | Purpose | When to run |
|---|---|---|
| `train_component4_sprint1.ipynb` | Train C4 lung decoder on Kaggle T4 | Once, Sprint 1 |
| `train_component2_sprint2.ipynb` | Train C2 TB head on Kaggle T4 | Once, Sprint 2 |
| `train_moe_sprint3.ipynb` | Full MoE training (Phases 0–3) | Once, Sprint 3 |
| `eval_baseline.ipynb` | Evaluate Improvement 1 on 560 held-out images | After Sprint 2 |
| `eval_moe.ipynb` | Evaluate Improvement 2; generates all paper figures | After Sprint 3 |
| `component0_qc.ipynb` | QC + harmonization development | Reference |
| `component1_dann.ipynb` | DANN domain adaptation exploration | Reference |
| `component2_txv.ipynb` | TXV pathology feature exploration | Reference |
| `component4_lung.ipynb` | Lung segmentation development | Reference |
| `component7_verification.ipynb` | Boundary critic development | Reference |
| `component9_json_output.ipynb` | Structured evidence JSON development | Reference |
| `component10_biogpt.ipynb` | BioGPT report generator (stub) | Future work |

---

## MoE Architecture — Key Design Decisions

### Why gate_only=true in Phase 2?

Expert decoders are trained in Phase 1 and have learned fine-grained pathology boundaries. Exposing them to gradient updates in Phase 2 with a freshly initialized gate causes catastrophic forgetting. `gate_only=true` freezes the expert bank and fusion module — only the ~50K gate parameters update. This converges in 12–15 epochs on a single T4.

To run full joint training (gate + experts + fusion together):
```yaml
# configs/moe.yaml
moe_training:
  joint:
    gate_only: false  # re-enables expert + fusion gradients
    epochs: 20        # increase epochs to compensate
```

### Adaptive Lesion Threshold

The fused lesion probability map is binarized using Otsu's method computed **within the lung mask**, then clamped to `[0.5, 0.8]`. This prevents Otsu from picking a threshold below 0.5 on healthy images with weak diffuse responses — the root cause of the 34.8% false-positive ALP seen in early experiments.

### Pseudo-Supervision Strategy

Since no pixel-level TB lesion GT exists for most datasets, Phase 0 generates pseudo-targets:
- **TBX11K:** bounding-box masks dilated and eroded (highest quality)
- **Shenzhen/Montgomery:** Grad-CAM pseudo-CAM from trained C2 TB head
- **Hard-negative supervision:** images where all experts report low confidence receive zero-mask targets with weight 0.3

---

## Results Summary

### Improvement 1 vs SOTA

| Metric | SOTA | Ours |
|---|---|---|
| TB detection AUROC (TBX11K) | 0.958 (Liu et al. 2020) | **0.983** |
| Lung segmentation Dice (Shenzhen) | ~0.850 (MedSAM, Ma et al. 2024) | **0.959** |
| Lung segmentation Dice (Montgomery) | ~0.850 (MedSAM, Ma et al. 2024) | **0.886** |
| Automated Timika AUROC | *Not reported* | **0.774** (TBX11K) |

### Improvement 2 vs Improvement 1

| Metric | Baseline (I1) | MoE (I2) | Δ |
|---|---|---|---|
| Timika AUROC — Shenzhen | 0.555 | **0.741** | +18.6 pp |
| ALP — NIH healthy images | 38.2% | **24.6%** | −13.6 pp |
| ALP — Montgomery | 59.8% | **22.1%** | −37.7 pp |

---

## Known Limitations

- **Montgomery:** Only 28 held-out evaluation images (±0.19 AUROC std). Timika AUROC estimates for Montgomery are high-variance and should not be over-interpreted.
- **NIH ALP target:** Target ≤10% not yet met (24.6%). Requires explicit hard-negative supervision for NIH images in Phase 0 caching.
- **Cavity detection sensitivity:** Expert 2 trained on pseudo-CAM only (no explicit cavity GT). Detection rate improves with labeled cavity data.
- **DANN convergence:** Domain classifier converges to a degenerate single-class prediction rather than random chance (~25%). Feature extractor remains useful despite this.
- **Report generation:** Component 10 is a deterministic template (faithfulness = 1.0 by construction). BioGPT-based generation is left as future work.

---

## References

- Liu et al. (2020). *Rethinking Computer-Aided Tuberculosis Diagnosis.* CVPR. (TBX11K dataset)
- Ma et al. (2024). *Segment Anything in Medical Images.* Nature Communications. (MedSAM)
- Cohen et al. (2022). *TorchXRayVision: A library of chest X-ray datasets and models.* MIDL. 
- Ganin et al. (2016). *Domain-Adversarial Training of Neural Networks.* JMLR. (DANN)
- Wasserman et al. (2019). *The Timika score for TB severity.* PNG field trial data.
