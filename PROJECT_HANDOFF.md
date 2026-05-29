# Project Handoff — Automated Timika Scoring (ICONIP 2026)

> Brain-dump briefing for a partner taking over. Last updated against commit
> `f3c913b` on branch `cleaned-repo`. Read Section 7 ("Crucial Context") first —
> several non-obvious decisions will save you days.

---

## 1. Project Overview

**Goal:** Publish an ICONIP 2026 paper that improves on **Kantipudi et al., "Automated
Pulmonary Tuberculosis Severity Assessment on Chest X-rays" (JIIM 2024, DOI
10.1007/s10278-024-01052-7)** — the current SOTA for computing the **Timika
severity score** from a chest X-ray (CXR).

**The Timika score** is a clinically validated TB severity index:
```
Timika = ALP + 40 * cavity        (range 0–140)
  ALP   = Affected Lung Percentage (0–100), how much lung shows abnormality
  cavity= 1 if cavitation present, else 0   (adds a flat +40)
```

**Our contribution (the novelty):** an **agentic, domain-adversarial Mixture-of-Experts
(DA-MoE)** that fuses Kantipudi's three estimation strategies and specifically
attacks the **cross-country generalization failure** their own paper exposes but
never fixes. Evaluation uses their exact **country-segregated (leave-one-country-out)
protocol** on the **TB Portals** dataset, which has real per-image ALP + cavity +
Timika ground truth.

**Headline framing (deferred until results are in):** "Robust cross-country Timika
scoring via an agentic MoE." If gains are statistically significant → lead with
cross-country generalization; if marginal → pivot to robustness / variance reduction.
The *method we build is identical either way*, so the title is not blocking.

---

## 2. Tech Stack & Architecture

**Languages/frameworks:** Python 3.12, PyTorch, torchvision (DenseNet121), MONAI-style
transforms, `segment-anything` (MedSAM), `ultralytics` (YOLOv5 for A1), scikit-learn /
scipy (metrics), pandas/numpy. Notebooks run on **Kaggle free-tier T4 (16 GB)**.

**No database.** All state is files: CSV manifests, cached PNG crops, `.pt` checkpoints.

**The data contract** — everything downstream reads one manifest schema:
```
manifest.csv -> image_id, image_path, patient_id, country, alp_0_100, cavity
```
Only `src/data/tbportals.py` knows the raw TB Portals schema; it produces this manifest.

### Two distinct codebases live in this repo — DO NOT CONFUSE THEM

**(A) LEGACY — the "old direction" (mostly superseded, partially reused).**
A 9-component pipeline (`src/components/component0..10_*.py`) built for an earlier
draft paper that used Shenzhen/Montgomery/TBX11K/NIH and a *proxy* Timika AUROC
(no real Timika ground truth). That draft PDF is `automated-timika-scoring-paper.pdf`
in the repo root. Reusable pieces from it:
- `component4_lung.py` — MedSAM ViT-B lung segmenter (fine-tuned decoder, Dice ~0.95).
- `component1_dann.py` — DANN gradient-reversal layer (for the new MoE's domain head).
- `component8_metrics.py` / `component8_timika.py` — Timika math.
- `component5_experts.py`, `component6_fusion.py`, `component3_routing.py`,
  `component7_*` (boundary critic / refinement) — MoE+critic scaffolding to adapt.

**(B) CURRENT — the "ICONIP pivot" (Kantipudi replication on TB Portals).** This is
where all recent work is. Key modules:

| File | Role |
|---|---|
| `src/data/tbportals.py` | manifest load, `make_country_split` (LOCO + patient split), `assert_no_patient_leakage`, synthetic data |
| `src/data/tbportals_dataset.py` | `TBPortalsDataset` — 224×224, ImageNet norm, augment (rot±15/hflip/zoom), optional lung crop |
| `src/components/baseline_paper.py` | `ALPRegressor` (A2 ALP), `CavityClassifier` (A2 cavity), `TimikaRegressor` (A3) — all DenseNet121 |
| `src/training/train_baseline_paper.py` | **A2** training (2 separate DenseNets) + per-head crop flags |
| `src/training/train_a3_direct.py` | **A3** direct-Timika regressor |
| `src/training/train_a1_detect.py` | **A1** detection-ALP (YOLO ∩ MedSAM lung) + cavity |
| `src/evaluation/eval_tbportals.py` | `Predictions`, `evaluate_split`, bootstrap CIs, `KANTIPUDI_A1/A2/A3` reference numbers, comparison table |
| `scripts/build_paper_manifest.py` | subsample Aug-2023 export to Kantipudi Table 1 (exactly 5,010 images) |
| `scripts/cache_lung_crops.py` | MedSAM → lung bbox → 224 crop cache |
| `scripts/prepare_tbx11k_yolo.py` | TBX11K VOC-XML → YOLO format (official train/val split) |
| `notebooks/tbportals_A1.ipynb`, `_A2.ipynb`, `_A3.ipynb` | the three runnable Kaggle notebooks |

**`src/components/component_alp_moe.py`** + `train_tbportals_baseline.py` are an earlier
ALP-MoE attempt on TB Portals (pre-strict-replication). Useful reference for the MoE
head, but the new MoE (Section 6) supersedes it.

### Model architecture (Kantipudi A2 — what we replicate faithfully)
- DenseNet121, ImageNet-init, global-avg-pooled to `f ∈ R^1024`.
- **ALP regressor:** `f → Linear(1024,1) → sigmoid` ∈ [0,1] (×100 = %). MSE loss.
- **Cavity classifier:** `f → Linear(1024,2)` logits. Cross-entropy, balanced training.
- **A3 (TimikaRegressor):** `f → Linear(1024,1) → sigmoid` (×140 = Timika). MSE.
- Hyperparameters (matched to paper): NAdam lr=1e-3, 30 epochs, **effective batch 300**,
  ImageNet normalization, augment rot±15°/hflip/zoom10% (each p=0.5), best-val-loss model
  selection.

---

## 3. Current State of the Codebase

**Implemented + smoke-tested (synthetic CPU runs pass end-to-end):**
- A2 pipeline (`train_baseline_paper.py`) — **fully run on Kaggle**, results analyzed (see §5).
- A3 pipeline (`train_a3_direct.py`) — code complete, synthetic smoke test passed; **not yet run on Kaggle with real data.**
- A1 pipeline (`train_a1_detect.py` + `prepare_tbx11k_yolo.py`) — code complete & compiles;
  the **TBX11K→YOLO conversion is verified working on Kaggle** (599 train / 200 val,
  1211 lesion boxes, 0 unmatched). YOLO train + A1 eval **not yet run.**
- Eval harness (`eval_tbportals.py`) — regression + cavity metrics + bootstrap 95% CIs +
  Kantipudi A1/A2/A3 reference tables. Working.
- Manifest builder — produces exactly 5,010 images matching Kantipudi Table 1. Verified.
- MedSAM lung-crop caching — working; crops cached and reusable as a Kaggle dataset.

**Three standalone Kaggle notebooks** (`tbportals_A1/A2/A3.ipynb`), each: clone repo →
build manifest → MedSAM crops → train → zip outputs. A1 additionally: inspect TBX11K →
convert to YOLO → train YOLOv5n → detection-ALP eval.

**Verification status honesty:** "smoke-tested" = the code path runs end-to-end on tiny
synthetic data; it does **not** mean the metrics are validated. Only the **A2 full run**
has produced real, analyzed numbers so far.

---

## 4. Active Work in Progress

**Exactly where we left off:** mid-way through running the three baseline notebooks on Kaggle.

- **A1 (`tbportals_A1.ipynb`):** Cell 4 (TBX11K→YOLO conversion) **just succeeded** after a
  multi-step debug (see §5). Output: `official split: train=599 val=200 ... 1211 lesion
  boxes, 0 unmatched`. **Next cells to run:** cell 5 (YOLO train ~30–60 min) → cell 6
  (A1 detection-ALP + cavity eval).
- **A2 (`tbportals_A2.ipynb`) and A3 (`tbportals_A3.ipynb`):** ready to run, **not yet
  started** with the locked config. Should be run in parallel Kaggle sessions.

**The locked A2 config** uses `--cavity-no-lung-crop` (whole-image cavity) + cropped ALP
— see §5 for why. This whole-image-cavity A2 run has **not been done yet**; the existing
A2 numbers are from the *cropped*-cavity run.

**Files touched most recently:**
- `scripts/prepare_tbx11k_yolo.py` — added `--train-list`/`--val-list` (official split) and
  stem-based filename matching (TBX11K XML `<filename>` has no extension).
- `src/training/train_baseline_paper.py` — added `--cavity-no-lung-crop` / `--alp-no-lung-crop`
  (per-head crop control); `predict_combined` now runs each head on its own loader.
- New: `train_a3_direct.py`, `train_a1_detect.py`, `prepare_tbx11k_yolo.py`, 3 notebooks.

**Task board (in-tool):** #15 lock A2 baseline, #16 build agentic DA-MoE, #17 A3 (done-code),
#18 A1 (stretch), #19 MoE eval/ablation harness.

---

## 5. Known Issues & Blockers

**Diagnosed scientific findings (these are results, not bugs to fix):**
- **Moldova ALP collapses under domain shift — this is the paper's motivation, not a bug.**
  Moldova's test ALP mean ≈ 40 vs the training-pool mean ≈ 26 (gap +13.8). Under
  leave-one-country-out the model regresses toward the training prior: our Moldova ALP MAE
  ≈ 24.7 barely beats a constant mean-predictor (≈ 25.5). Romania/Kazakhstan have small
  gaps and match the paper. **Do NOT try to "fix" Moldova in the baseline — the MoE/DANN is
  what's supposed to improve it.**
- **Cavity AUC ~0.08–0.10 below paper, caused by lung-crop clipping the apex** (where
  cavities sit). Whole-image cavity beats MedSAM-cropped on all 3 countries
  (≈0.751/0.846/0.764 whole vs 0.722/0.796/0.752 cropped; paper 0.80/0.88/0.85). The paper
  itself reports the same crop-hurts-detection effect for its YOLO detector. Hence the
  locked A2 trains cavity on whole images.

**A2 baseline results so far (3-seed mean, cavity CROPPED — pre-lock):**

| Country | ALP MAE (ours/paper) | Cavity AUC | Timika MAE | Pearson |
|---|---|---|---|---|
| Romania | 12.49 / 11.86 ✅ | 0.722 / 0.80 | 19.93 / 18.70 | 0.67 / 0.70 |
| Kazakhstan | 12.33 / 12.16 ✅ | 0.752 / 0.85 | 21.85 / 19.62 | 0.63 / 0.70 |
| Moldova | 24.72 / 16.24 ❌ | 0.796 / 0.88 | 30.34 / 18.85 | 0.685 / 0.84 |

**Environment / workflow gotchas (real time-sinks we already hit):**
1. **Git push must stay small.** Result/checkpoint zips (`checkpoints_paper_baseline.zip`
   = 453 MB, etc.) once bloated a push to 665 MB → HTTP 408 timeout. These are now in
   `.gitignore` (`*.zip`, `tbportals_paper_baseline/`, `automated-timika-scoring-paper.pdf`).
   **Never `git add -A` blindly here.**
2. **Kaggle module caching.** After `git pull` updates a script, re-running a cell that
   already did `from X import ...` uses the **cached** module. You MUST **restart the kernel**
   (and ideally `shutil.rmtree` the clone + re-run cell 0) to load new code. We lost time to
   this — the symptom is seeing an *old* error/log message.
3. **GPU OOM from MedSAM.** MedSAM ViT-B holds ~14 GB across a kernel session. `cache_lung_crops.py`
   now skips loading MedSAM entirely if all crops are cached, and training frees models
   between runs. If you OOM, restart the kernel.
4. **T4 can't fit batch 300.** Use `--batch-size 60 --accum-steps 5` (= effective 300 via
   gradient accumulation). Note: BatchNorm still sees the 60-sample micro-batch — a known,
   accepted minor deviation from the paper.

**No COVID-19 lung-segmentation data.** The v7labs COVID-19 dataset (what Kantipudi used to
train its UNet lung segmenter) ships **masks but not images** (2020 URL tokens are dead;
images need a V7 account via `darwin-py`). **Decision: we use the existing fine-tuned MedSAM
decoder as the lung segmenter for all approaches** (Dice ~0.95, on par with the paper's UNet).
The COVID-UNet path is abandoned.

---

## 6. Next Steps & Pending Tasks

**Phase 1 — Lock the three baselines (the GATE; do not build the MoE until these hold up):**
1. Run **A2** notebook (locked: `--cavity-no-lung-crop`, 3 seeds) → confirm cavity AUC rises
   toward ~0.80 and Romania/Kazakhstan ALP still match.
2. Run **A3** notebook (3 seeds) → compare to Kantipudi A3.
3. Finish **A1** notebook: cell 5 (YOLO train) → cell 6 (eval) → compare to Kantipudi A1.
4. Collect 3-seed means + bootstrap CIs for all three; this is the "Repro" row of the paper.

**Phase 2 — Build the agentic DA-MoE (flagship on A2), task #16.** See Section 7 for the
locked design. Two-phase training (agents first, then gate+critic). Target: beat the locked
A2 baseline on **Timika Pearson**, especially Moldova.

**Phase 3 — Fold the A1 detection view into the MoE gate (stretch).**

**Phase 4 — Eval + ablations (task #19):** ablation matrix
`baseline / +MoE-fusion / +DANN / +critic / full`, λ_DANN sweep, gate temperature,
bootstrap CIs, significance testing.

**Kantipudi reference numbers to beat (Romania / Moldova / Kazakhstan):**
- A2: ALP MAE 11.86/16.24/12.16; cavity AUC 0.80/0.88/0.85; Timika MAE 18.70/18.85/19.62; Pearson 0.70/0.84/0.70.
- A3: Timika MAE 19.67/18.98/22.12; Pearson 0.70/0.85/0.74.
- A1: Timika MAE 23.83/24.44/22.13; Pearson 0.59/0.80/0.68. (A1 is their *worst* approach.)

---

## 7. Crucial Context & Decisions

1. **TWO PAPERS — do not confuse them.**
   - `baseline_paper.pdf` lives in the **parent** `proj/` folder (NOT in the repo) = the
     **Kantipudi JIIM 2024** baseline we replicate.
   - `automated-timika-scoring-paper.pdf` in the repo **root** = **our own earlier draft**
     (Irfan/Habib/Taj), built on the abandoned Shenzhen/Montgomery/TBX11K + proxy-AUROC
     direction. Reuse its infra, not its data/eval.

2. **Replicate all 3 Kantipudi approaches; flagship = A2.** A1 = lung-seg + YOLO lesion-det +
   cavity (worst, most work). A2 = ALP regressor + cavity classifier (best). A3 = direct
   Timika regressor. **YOLO/TBX11K only feed A1**, so they are NOT needed for A2/A3.

3. **The MoE design is locked (user-approved):**
   - **Roster A — clinical sub-task "agents," each supervised by a real label:**
     (1) cavity agent → cavity classification, (2) ALP agent → ALP regression,
     (3) lung-region agent → lung mask (reuse MedSAM), (4) Timika-arbiter agent → direct
     Timika (this *is* A3).
   - **Fusion gate:** a learned MLP routes per-image over Timika "views": A2-compositional
     (`100·ALP + 40·cavity`), A3-direct (arbiter), and (stretch) A1-detection. Kantipudi ran
     these as *separate* systems; **our novelty is unifying them with learned routing.**
   - **DANN:** country-adversarial head via gradient reversal on `f`, **trained on training
     countries only** (domain generalization, fair to the LOCO protocol). Reuse `component1_dann`.
   - **Critic loop ("agentic"):** a critic head flags low-confidence / high-disagreement
     cases and re-weights toward the trusted view (perception→critique→refine). Reuse
     `component7_*`.
   - **Loss:** `MSE(ALP) + α·BCE(cavity) + β·MSE(arbiter) + γ·MSE(fused) + λ·CE_GRL(country) + δ·critic + ε·load_balance`.
   - **Training order matters (hard rule):** train experts/agents BEFORE the gate — a gate
     trained on random experts learns random routing.

4. **Country-segregated protocol (must match Kantipudi exactly):** test = ALL images of the
   held-out country (Romania / Moldova / Kazakhstan); never train on the held-out country.
   ALP/A3 train-val = patient-disjoint 80:20 of the other countries (Table 3). Cavity train-val
   = all non-held-out cavity+ images + an equal random sample of cavity− , patient-disjoint
   80:20 (Table 2). **Patient-level leakage is the #1 silent metric inflator** —
   `assert_no_patient_leakage` guards every split; never bypass it.

5. **Dataset = TB Portals (NIH), Jan-2023 release ≈ 5,010 images, 8 countries.** We don't have
   that exact release, so `build_paper_manifest.py` subsamples our Aug-2023 export to Table 1's
   exact per-(country,cavity) counts with a fixed seed (deterministic, reproduces the
   *distribution*). `image_id` is set to the PNG **stem** (the raw id contains "/" and breaks
   crop paths) — do not revert this.

6. **Lung segmenter = fine-tuned MedSAM** (`checkpoints/component4/component4_mask_decoder.pt`,
   git-tracked, ~46 MB), not the paper's COVID-trained UNet (data unavailable). This is the one
   intentional deviation from strict replication, and it's fine (Dice ~0.95).

7. **TBX11K specifics (for A1):** VOC-XML annotations in `annotations/xml/`; the XML
   `<filename>` has **no extension** (`tb0003`), so the converter matches images by **stem**.
   Lesion classes (`ActiveTuberculosis`, `ObsoletePulmonaryTuberculosis`) merge to one
   `lesion` class. We use TBX11K's **official** `lists/TBX11K_train.txt`/`_val.txt` split
   (599/200, 1211 boxes) rather than Kantipudi's unpublished 511/128/160 — close enough; A1
   is low-stakes.

8. **Git / collaboration rules:**
   - Branch is **`cleaned-repo`**, remote `https://github.com/mabdullahi7780/dl-project-codebase.git`.
   - Kaggle notebooks **clone from the remote**, so any code change must be **pushed** before
     it can run on Kaggle. The assistant cannot push; the human pushes.
   - Large artifacts are `.gitignore`d — keep them out of commits.

9. **Local dev:** venv at `timika/Scripts/python.exe` (Windows). Synthetic-data smoke tests
   (e.g. `make_synthetic_dataset`) let you validate code paths on CPU without Kaggle/GPU.
