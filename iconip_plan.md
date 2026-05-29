# ICONIP 2026 Plan — Domain-Adversarial Mixture-of-Experts for Automated Timika Scoring

**Status target:** course project → publication (ICONIP 2026).
**Baseline / SOTA being beaten:** Kantipudi et al., *Automated Pulmonary Tuberculosis Severity Assessment on Chest X-rays*, JIIM 2024 (DOI 10.1007/s10278-024-01052-7), best model **A2**.
**Compute:** Kaggle free-tier T4 (16 GB). **Timeline:** 3 days.

---

## 0. Thesis (what we claim)

Kantipudi established automated Timika scoring on TB Portals, but accuracy degrades and destabilizes across **unseen countries** (Pearson falls to 0.70 on Romania/Kazakhstan). We improve **cross-country generalization** with a **domain-adversarial Mixture-of-Experts regressor**, evaluated under their **exact country-segregated protocol**.

Two contributions, both aimed at their measured weakness:
1. **MoE regression head** — severity-regime specialization vs. their single dense regressor.
2. **DANN** — country-invariant features for better transfer to unseen countries.

Incremental novelty (known techniques, new task/benchmark). Honest about that — acceptance hinges on **fair comparison + statistical significance**, with the cross-country generalization framing as the lead.

---

## 1. Architecture

- **Backbone:** DenseNet121, ImageNet-pretrained. Input = cropped-lung 224×224, ImageNet-normalized. Pooled features `f ∈ R^1024`. (Match Kantipudi so MoE/DANN is the only difference.)
- **ALP MoE head:** K experts `e_k(f) → a_k ∈ [0,1]` (sigmoid MLP). Gate `g(f)=softmax(W_g f / τ)`. `ALP = 100·Σ_k g_k a_k`.
- **Cavity head:** binary classifier on `f` (BCE, class-weighted). Shared backbone = "unified model vs their two separate nets."
- **DANN head:** country classifier on `f` via Gradient Reversal Layer (reuse `component1_dann.py`). λ ramps `2/(1+e^{−γp})−1`. Trained on **training-countries only** (domain generalization, fair to Kantipudi).
- **Timika:** `ALP + 40·1[cavity_prob > t]`, `t` tuned on val. Reuse `component8_metrics.compute_timika_score`.
- **Loss:** `MSE(ALP/100) + α·BCE(cavity) + β·CE(country via GRL)`.

---

## 2. Data pipeline (manifest contract)

Only `src/data/tbportals.py` touches TB Portals' raw schema. Everything else reads:

```
manifest.csv → image_id, image_path, patient_id, country, alp_0_100, cavity
```

- ALP ← `overall percent of abnormal volume`; cavity ← `are cavities present`; country ← patient/condition table. Behind a `COLUMN_MAP` you fix once you see the file.
- **Splits:** 3 country-segregated configs (hold out Romania / Moldova / Kazakhstan); test = all held-out-country images; training countries patient-disjoint.
- **Hard assertion:** no `patient_id` in both train and test (patient leakage = #1 silent metric inflator).
- **Lung crop:** reuse MedSAM (`component4_lung`) → bbox → 224, cached to disk. Optional fallback: full image (flag `use_lung_crop`).

---

## 3. Day-by-day (gates matter more than the clock)

### Day 1 — Data + reproduce Kantipudi → THE GATE
- [ ] `tbportals.py` (adapter + splits + leakage assertion + synthetic manifest)
- [ ] `tbportals_dataset.py` (Dataset + augment + optional lung crop)
- [ ] `cache_lung_crops.py`
- [ ] `component_alp_moe.py` (backbone + single head [Day1] + MoE/DANN heads [Day2], flag-gated)
- [ ] `train_tbportals_baseline.py` (single ALP regressor + cavity classifier)
- [ ] `eval_tbportals.py` (MAE / MAE% / Pearson / Spearman / RMSE / R² + cavity AUC/F1 + bootstrap CIs + comparison table)
- [ ] `configs/tbportals.yaml`
- [ ] Notebooks 01–04
- **GATE:** reproduce A2 → MAE ~18–19 / Pearson ~0.70–0.84 / cavity AUC ~0.80–0.88, ≥3 seeds. If not, parsing is wrong (ALP scale? country join? leakage?). **Do not proceed to MoE/DANN on a broken baseline.**

### Day 2 — MoE + DANN
- [ ] MoE head, **two-phase** (Phase 1 warm experts under frozen/uniform gate; Phase 2 train gate) + load-balancing loss.
- [ ] DANN head; λ sweep {0.1, 0.5, 1.0}; watch adversarial instability.
- [ ] Checkpoint every epoch (T4 sessions die) — resumable.

### Day 3 — Eval + ablations + paper
- [ ] Full country-segregated eval, all variants, bootstrap 95% CIs, mean±std over seeds.
- [ ] Ablation matrix (§5).
- [ ] Paper: add Kantipudi citation; fix detection SOTA to **SymFormer (0.982, not Liu 0.958)**; reframe novelty as improving cross-country generalization of Kantipudi.
- [ ] Buffer (Day-1 parsing usually slips).

---

## 4. Training config

| | Setting |
|---|---|
| Optimizer / lr | NAdam, 1e-3 |
| Epochs / batch | 30 / 32–64 |
| Augment | ±15° rotation, x-flip, 10% zoom (identical to Kantipudi) |
| Norm | ImageNet mean/std |
| Early stop | val ALP-MAE |
| Seeds | ≥3 (report mean±std + CIs) |

Compute is not the bottleneck (~1 min/epoch on T4). Time goes to parsing + protocol fidelity.

---

## 5. Ablation matrix

| Variant | ALP MAE ↓ | Timika MAE ↓ | Pearson ↑ | Cavity AUC ↑ |
|---|---|---|---|---|
| Kantipudi A2 (reported) | 11.86/16.24/12.16 | 18.70/18.85/19.62 | 0.70/0.84/0.70 | 0.80/0.88/0.85 |
| Repro A2 (ours) | gate | gate | gate | gate |
| + MoE head | | | | |
| + DANN | | | | |
| Full (MoE+DANN) | | | | |

Columns are Romania / Moldova / Kazakhstan. Sweeps: #experts {2,3,4,6}, routing {soft, top-1}, gate τ, λ_DANN. Optional rung: TXV CXR-pretrained backbone (keep `active_backend != "xrv"` guard).

---

## 6. Risk register (brutal)

1. **Reproduction gate fails (highest prob):** ALP scale 0–1 vs 0–100, wrong country join, patient leakage. Mitigate: assertions + print label distributions before training.
2. **DANN-DG doesn't help / hurts:** DG is harder than adaptation. Fallback: report MoE-only win and/or add transductive UDA column (unlabeled held-out images, labels untouched), reported transparently.
3. **MoE collapse to one expert:** load-balancing loss + temperature.
4. **Marginal/insignificant gains:** pivot narrative to robustness / cross-country variance reduction. Don't oversell.
5. **Single-radiologist GT:** noise ceiling; acknowledge as limitation.
6. **Release-version drift** vs Jan-2023 ~5k: report own counts; held-out countries fixed → comparison valid.

**Not achievable in 3 days (skip):** GT-supervised pixel lesion segmentation (no masks), polished YOLO explainable arm, BioGPT report work.

---

## 7. Critical guards (never violate — from prior incidents)

- Keep the `active_backend != "xrv"` RuntimeError guard if using the TXV backbone (mock returns zeros → AUROC 0.5).
- **MoE phase ordering:** experts before gate (gate on random experts = random routing).
- TBX11K detection eval: stratified sampling (limit draws only TB+ otherwise) — only relevant to the detection ablation.

---

## 8. File map

**New:** `src/data/tbportals.py`, `src/data/tbportals_dataset.py`, `src/components/component_alp_moe.py`, `src/training/train_tbportals_baseline.py`, `src/training/train_tbportals_moe_dann.py`, `src/evaluation/eval_tbportals.py`, `scripts/cache_lung_crops.py`, `configs/tbportals.yaml`, `notebooks/tbportals_0{1,2,3,4}_*.ipynb`.

**Reuse:** `component4_lung` (lung crop), `component1_dann` (GRL), `component8_metrics.compute_timika_score` (Timika), `core/{device,seed,constants}`, `ablations/` CSV pattern.

---

## 9. Publication read

ICONIP is a credible Springer-LNCS venue, not top-tier. Clean SOTA reproduction + MoE/DANN + statistically significant cross-country gains + proper ablations = realistic accept. MoE/DANN are well-known, so reviewers judge on fair comparison + significance. Lead with cross-country generalization. If gains are within noise, pivot to robustness/variance-reduction and be upfront.
