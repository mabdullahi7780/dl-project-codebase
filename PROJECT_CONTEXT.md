# Complete Project Context — TB Portals Timika Severity Scoring
## Agentic 6-Rung Pipeline on a Frozen CXR Foundation Backbone
### For Report Writing / LLM Handoff

---

## 0. What This Document Is

This is a complete, corrected, self-contained briefing of the entire project, **rewritten to
reflect the current research direction**. The project pivoted away from its original
Domain-Adversarial Mixture-of-Experts (DA-MoE) plan after that approach failed; the MoE now lives
as **one rung (Rung 5) of a six-rung agentic pipeline built on a frozen chest-X-ray foundation
model**, with the domain adversary removed.

It is written to be handed to another LLM (or a human) to help write a research report /
conference paper (target: ICONIP 2026). Read it top-to-bottom. Every number here is grounded in
the actual codebase, experimental logs, and the reference paper. **Where something is a hope, a
hypothesis, or an unrun experiment, it is labelled as such** — do not let the paper claim more
than the evidence supports.

**Status legend used throughout:**
- ✅ **RUN** — executed, numbers in hand.
- 🛠️ **IMPLEMENTED** — code exists and is smoke-tested, but the experiment has not been run yet.
- 📋 **PLANNED** — designed, not yet implemented.
- ❌ **ABANDONED** — built, run, failed; kept as a negative result / cautionary finding.

---

## 1. The Research Question (REVISED)

**Original (abandoned) question:** Can a Domain-Adversarial Mixture-of-Experts beat Kantipudi et
al. (JIIM 2024) on cross-country TB severity scoring? → *This was answered "no, as built" and the
direction was dropped (see §6).*

**Current question:** Can an **agentic, rung-structured pipeline built on a frozen chest-X-ray
foundation model (RAD-DINO)** — stacking balanced regression, test-time feature adaptation,
retrieval-based label calibration, a cavity-head upgrade, a severity-specialised
Mixture-of-Experts, and conformal uncertainty — beat (a) **our own faithfully-replicated
Kantipudi baseline** (the honest target) and ideally (b) **Kantipudi's reported numbers** (the
aspirational target), specifically on the cross-country generalisation failure that the SOTA
paper itself exposes (Moldova)?

**The most important empirical findings (✅ RUN — 5 seeds × M=10 ensemble × 7 rungs × all four modes, final headline run):**

1. **Foundation backbone (Rung 1) closes most of the Moldova gap on its own** — RAD-DINO frozen
   features + light head: Moldova Timika MAE 30.68 → 22.28, beats Kantipudi on Kazakhstan
   (18.07 < 19.62).
2. **Retrieval calibration (Rung 3) is the most consistent positive contributor** (α≈0.20–0.30
   selected on val). Paired bootstrap (A2): significant on Moldova (Δ −0.25, CI excludes 0) and
   Kazakhstan (Δ −0.50); never a regression on any cell.
3. **Spatial cavity head (Rung 4b) is the decisive single rung on the headline configuration.**
   Cavity AUC on Romania jumps from 0.679 (global CLS) to 0.721 (spatial attention); Timika
   Romania 20.89 → 19.62 (A2 best+spat-cav); Fusion best+spat-cav = **Rom 19.12 / Mol 21.59 /
   Kaz 16.74**. Spatial cavity also lifts the regression slope (the only intervention that does).
4. **The Moldova shift is label + *minor* covariate** — transductive feature standardization
   (Rung 2) gives the best Moldova number in the entire run (Fusion best+TTA = **19.98 ± 0.26**)
   while degrading other countries. **Country-conditional TTA** is itself a finding.
5. **Deep ensemble + conformal (Rung 6) on fusion** delivers calibrated severity intervals across
   the held-out shift: marginal coverage 0.88–0.92 against nominal 0.90, mean width 82 Timika
   points. Per-country decomposition surfaces residual Moldova under-coverage (0.83) — honestly
   reported as the remaining covariate-shift residue TTA cannot fully remove.
6. **A1 spatial RAD-DINO head crushes its locked baseline by 7–11 Timika points** and **beats
   Kantipudi's A1 paper numbers on all three countries** (A1 best+spat-cav: 19.65 / 21.05 / 17.89
   vs paper 23.83 / 24.44 / 22.13) — a novel detection-free regional ALP estimator.
7. **Three publishable negative results**:
   - Rung 4 (class-balanced focal cavity) *harms* cavity AUC on an already-balanced split — the
     intervention is wrong for the data distribution; R4b (spatial attention) is the correct fix.
   - Rung 5 (MoE on frozen features) is a null result — DomainBed thesis transfers to medical
     CXR severity regression.
   - Rung 7 (post-hoc isotonic slope calibration) is a null on real data — synthetic tests showed
     a +0.10–0.18 slope lift; real-data lift is +0.00–0.02. The val set is too small/close to
     train for the calibrator to learn a useful curve. The spatial cavity head moves the slope
     more (R4b Rom slope 0.596 → 0.666) than any post-hoc calibrator.

This reframes the contribution: the foundation backbone is the floor-raiser; the agentic rungs
add **statistically significant additional lift** on top of that floor — most strongly via
Rung 3 (retrieval) and Rung 4b (spatial cavity attention). See §6.2 for the full scoreboard.

---

## 2. The Reference / SOTA Paper (unchanged — still the benchmark)

**Kantipudi et al., "Automated Pulmonary Tuberculosis Severity Assessment on Chest X-Rays Using
Deep Learning," *Journal of Imaging Informatics in Medicine* (JIIM), 2024.**

### 2.1 The Timika Score

A radiologist severity scale in the range **[0, 140]**:

```
Timika = ALP + 40 × cavity_flag
```

- **ALP (Affected Lung Percentage):** percentage of lung field occupied by TB lesions, 0–100.
- **cavity_flag:** binary (1 = cavities present). Presence adds a fixed 40-point penalty.

The 40-point cavity penalty is critical to the error analysis: a *single* cavity
misclassification costs 40 Timika points, so a weak cavity classifier can dominate Timika MAE even
when ALP regression is excellent.

### 2.2 Kantipudi's Three Approaches

| Approach | Description | Their verdict |
|----------|-------------|---------------|
| **A1** | YOLOv5 lesion detection → geometry-based ALP + separate cavity DenseNet121 | their worst |
| **A2** | DenseNet121 ALP regressor + DenseNet121 cavity classifier (two networks) | **their best** |
| **A3** | Single DenseNet121 directly regresses Timika end-to-end | middle |

### 2.3 Kantipudi's Per-Country Numbers (LOCO: Romania / Moldova / Kazakhstan held out)

**A2 (their best):**

| Country | Timika MAE | MAE% | Pearson | ALP MAE | Cavity AUC | Cavity F1 |
|---------|-----------|------|---------|---------|-----------|-----------|
| Romania | 18.70 | 13.36% | 0.70 | 11.86 | 0.80 | 0.81 |
| Moldova | 18.85 | 13.46% | 0.84 | 16.24 | 0.88 | 0.71 |
| Kazakhstan | 19.62 | 14.01% | 0.70 | 12.16 | 0.85 | 0.72 |

**A3:**

| Country | Timika MAE | MAE% | Pearson |
|---------|-----------|------|---------|
| Romania | 19.67 | 14.05% | 0.70 |
| Moldova | 18.98 | 13.56% | 0.85 |
| Kazakhstan | 22.12 | 15.80% | 0.74 |

**A1:**

| Country | Timika MAE | MAE% | Pearson |
|---------|-----------|------|---------|
| Romania | 23.83 | 17.02% | 0.59 |
| Moldova | 24.44 | 17.46% | 0.80 |
| Kazakhstan | 22.13 | 15.81% | 0.68 |

### 2.4 Kantipudi's Architecture (for the related-work / baseline section)

- **Lung segmenter:** ResNet18-UNet encoder-decoder, trained on a COVID-19 CXR set (6,396 frontal
  CXRs, polygonal masks), Dice 0.96 ± 0.02. **The COVID-19 training data URLs are dead.**
- **ALP regressor (A2):** DenseNet121 (ImageNet) → Linear(1024→1) → sigmoid; ALP fraction × 100.
  MSE loss. Lung-cropped 224×224 input.
- **Cavity classifier (A2):** DenseNet121 → Linear(1024→2), cross-entropy on a balanced split.
- **A3:** single DenseNet121 → Linear(1024→1) → sigmoid × 140, direct Timika, MSE.
- **Training:** NAdam, lr 1e-3, 30 epochs, effective batch 300, ImageNet norm, light augmentation,
  best-val checkpoint.
- **Key quote (p.2182):** *"TB lesions are most often present at the lung apex."* Relevant to the
  apex-clipping hypothesis (§7.2) and the spatial A1 head (§5, Rung-A1).

---

## 3. The Dataset: TB Portals

**Source:** TB Portals Published Imaging Data, August 2023 release. Multi-country TB CXR database
with radiologist-annotated Timika scores.

**Schema:** two CSVs joined on `imagingstudy_id`: `TB_Portals_CXRs_August_2023.csv` (image
metadata) + `TB_Portals_CXR_Manual_Annotations_August_2023.csv` (ALP 0–100, cavities present).
Sextant-level annotation rows are aggregated to one row per image (ALP taken from the overall-lung
column; cavity = 1 if any sextant has any cavity count > 0).

**Our manifest:** 5,010 images after a seed-42 subsample to match Kantipudi's reported Table 1
totals. **This is almost certainly *not* their exact image set** (TB Portals grows over time and
they never published a file list). Columns:
`image_id, image_path, patient_id, country, alp_0_100, cavity`.

**Split:** patient-disjoint Leave-One-Country-Out (LOCO). `make_country_split()` guarantees no
patient appears in two splits (`assert_no_patient_leakage`, a hard guard — never disable it).
ALP/A3 train-val = patient-disjoint 80:20 of the non-held-out countries; cavity uses a *balanced*
split (all non-held cavity+ plus an equal number of random cavity−, patient-disjoint).

**Country distribution (the root of the whole project):**

| Held-out | Test ALP mean | Train-pool ALP mean | Gap |
|----------|--------------|--------------------|-----|
| Romania | ≈31.9 | ≈27.4 | +4.4 |
| **Moldova** | **≈39.8** | ≈26.0 | **+13.8** |
| Kazakhstan | ≈22.6 | ≈28.1 | −5.5 |

Moldova is by far the sickest cohort. This **label-distribution gap** is the documented cause of
the Moldova generalisation failure (§7). Kazakhstan is *healthier* than the pool — easier in
principle, which is consistent with it being the country we now beat the paper on.

---

## 4. The Pivot: From DA-MoE to the Agentic Pipeline (read this before §5)

The project originally bet on a Domain-Adversarial MoE. That model **failed** (§6): the domain
adversary (DANN) at full strength collapsed the shared trunk's regression features, and on the
direct-Timika mode the model degenerated to predicting a near-constant value (Pearson ≈ 0).

The diagnosis that drove the pivot is conceptual, not just a tuning bug:

- **Moldova's failure is *label* (target) shift, not *covariate* (style) shift.** The features
  are fine; the *label distribution* of the test country is shifted into a sparsely-sampled,
  higher-severity region.
- **Domain-invariance is the wrong tool under label shift.** DANN/CORAL try to make features
  country-indistinguishable. But severity *is* correlated with country (Moldova is sicker), so
  forcing country-invariance provably suppresses the very severity signal the regressor needs
  (cf. *On Target Shift in Adversarial Domain Adaptation*). The DA-MoE failure is a concrete
  instance of this theorem.
- **DomainBed lesson (Gulrajani & Lopez-Paz 2021):** across many domain-generalisation
  algorithms, none reliably beats a well-tuned ERM by more than ~1 point; the real gains come
  from **backbone, augmentation, tuning, and model selection**, not from invariance penalties.

So the pivot is principled: **stop fighting the domain signal; instead (a) start from features
that already generalise across populations (a CXR foundation model), (b) correct the label-prior
with a balanced objective, and (c) calibrate predictions using evidence retrieved from the
training distribution** — none of which throw away severity information. The MoE is retained, but
only as a severity-specialised mixture on frozen features, with the adversary removed.

---

## 5. The Agentic 6-Rung Architecture (the core contribution)

**Shared design principle — frozen features, cached once.** All rungs run on a **frozen**
backbone whose per-image embeddings are computed *once* and cached to disk
(`scripts/cache_features.py` → `.npz`). Downstream heads then train on cached vectors in seconds.
The features are **mode- and rung-independent**, so the entire experimental ladder (all rungs ×
a2/a3/fusion) reuses a single cache — the whole sweep costs *minutes of GPU*, not hours. (A1 is
the one exception: it caches a spatial patch grid instead of the global vector — see Rung-A1.)

**Primary backbone — RAD-DINO** (`microsoft/rad-dino`): a DINOv2 ViT self-supervised on ~900k
chest X-rays, 86.6M params, 768-d CLS embedding (`pooler_output`), shorter-side 518. **Not
gated** (no HF token). Frozen; fine-tuning is reportedly unnecessary. Fallbacks/controls:
TorchXRayVision DenseNet121 (`txrv`, 1024-d) and ImageNet DenseNet121 (`densenet`, 1024-d, =
Kantipudi's backbone, for an apples-to-apples "backbone-only" ablation).

**Modes (cover all of Kantipudi's approaches):**
- **a2** — ALP regression head + cavity classifier head; `Timika = 100·ALP + 40·cavity`.
- **a3** — single head regresses Timika directly (× 140); no separate cavity model.
- **fusion** — blend the a2 and a3 Timika predictions (blend weight selected on validation).
- **a1** — *spatial* ALP head over a pooled patch-token grid (per-cell lesion involvement →
  mean → ALP), a frozen-feature stand-in for Kantipudi's detection-geometry ALP, plus a cavity
  head. **Highest-risk, lowest-expected-accuracy mode** — A1 was always the weakest; its value is
  a spatial/interpretable story, not beating a2/a3 on MAE.

### The rungs

Each rung is an **ablation toggle** so its contribution is measured independently against the
locked baseline with bootstrap CIs. Cumulative "stacked" configs combine only the rungs that
demonstrably help.

**Rung 1 — Foundation backbone + balanced regression.** ✅ RUN (a2).
Light MLP head on cached RAD-DINO features; honest model selection (best val Pearson across all
epochs). Two loss variants: plain **MSE** (isolates the *backbone* lift) and **Balanced MSE / BMC**
(Ren et al., CVPR 2022 — a batch-Monte-Carlo objective for imbalanced regression that corrects the
training-label prior so the sparse high-severity region is not down-weighted; learnable
`noise_sigma`, no prior needed). *Targets:* the Moldova mean-collapse. *Measured result:* see §6.1
— large win, mostly from the backbone; BMC adds a small consistent edge on the hard countries.

**Rung 2 — Test-time feature adaptation.** ✅ RUN — **partial win, refines the diagnosis.**
Standardise features (z-score), either fit on the train pool (inductive) or re-fit on the held-out
country at test (transductive ≈ test-time BN adaptation); plus CORAL (Sun & Saenko 2016) 2nd-order
covariance alignment. *Targets:* covariate shift. *Measured result:* **TTA dropped Moldova's
ALP MAE 20.87→17.36 (A2) and Timika MAE 21.84→20.07 (fusion, the best Moldova number anywhere)**,
but *degraded* Romania and Kazakhstan. This refines the diagnosis: Moldova has **label + minor
covariate shift**, and TTA is a country-conditional tool — itself a publishable finding.

**Rung 3 — Retrieval-augmented label calibration.** ✅ RUN — **the most consistent positive rung.** *(Novelty centrepiece.)*
For each test image, retrieve its k nearest neighbours among the *train pool* in frozen feature
space (cosine, softmax-over-distance kernel) and read their labels; blend the head's prediction
with the retrieved label estimate: `y = (1−α)·head + α·kNN`. The blend weight **α is selected on
the validation split** (never on the test country); **α = 0 recovers the pure head**, making the
retrieval contribution directly measurable. *Targets:* label-shift mean-collapse *directly* — if a
Moldova image's neighbours are high-severity training images, the calibration pulls the prediction
up off the mean. *Measured result:* α≈0.27–0.31 selected on val; **most consistent positive
contributor — never regresses, lifts every (mode, country) cell**. First config to clear
Romania: A3 R3 = 20.11 (ties locked); fusion R3 = 19.95 (inches under locked baseline,
the only number to do so cleanly via a single rung). A transductive, training-free
calibrator over a CXR-foundation feature space, applied to cross-country Timika regression,
is (to our knowledge) novel.

**Rung 4 — Cavity-head upgrade.** ❌ RUN — **honest negative result (publishable cautionary finding).**
Class-balanced focal loss (Lin et al. 2017 focal + Cui et al. 2019 effective-number reweighting)
+ per-held-out-country decision-threshold calibration on the balanced validation split.
*Targets:* Romania (low cavity AUC). *Measured result:* **focal *hurt* cavity AUC** (A2: 0.794
→ 0.762; Romania specifically 0.675 → 0.648), and the calibrated threshold (0.43) increased
false positives at +40 Timika each. **Why it backfired:** the cavity training split is *already*
balanced by `make_balanced_cavity_split`; layering class-balanced + focal on an already-balanced
distribution is the wrong intervention. **The Romania cavity bottleneck is therefore intrinsic
feature quality on Romania-specific cavity morphology, not class imbalance** — motivates the
spatial-cavity-head fix planned next (cavities are localized; the global CLS pooler washes them
out — patch-level scoring should preserve the signal).

**Rung 5 — Severity-specialised Mixture-of-Experts + uncertainty critic.** ❌ RUN — **null result (publishable).**
K regression experts softly gated by a learned gate, plus a **critic** head (regresses
`1 − |error|`). No shared trunk, no DANN — so nothing collapses. *Measured result:* **slightly
worse or null across all (mode, country) cells**. Why: experts all see the same frozen feature
vector, so the gate has no extra information; K experts on shared inputs ≈ MLP with more
parameters and noisier optimisation. This is the **DomainBed thesis transferring to medical
CXR severity**: when features are already strong, stacking complex routing on top doesn't help.
Combined with the prior DA-MoE failure, two negative MoE results form a clear methods-discussion
point.

**Rung 6 — Conformal prediction + deep ensemble.** ✅ RUN — **headline rung (point MAE + intervals).**
M=5 independently-seeded heads averaged + split-conformal Timika prediction intervals (Vovk;
Lei et al. 2018) calibrated on a held-out split. *Measured result:* **small consistent
point-MAE bump (e.g. fusion Kaz 17.01→16.86, A3 Kaz 18.45→18.21) and conformal coverage 0.87–0.90
under domain shift** against nominal 0.90 — i.e. the finite-sample marginal guarantee survives
cross-country shift in our setting. Mean width 80–95 Timika points (large but honest given
Moldova's variance). Best single config: **fusion · rung6_conformal → Rom 19.96 / Mol 22.42
/ Kaz 16.86** — the run's overall headline.

### Why this is novel (for the paper's claims)

A linear/MLP probe on a frozen foundation model is a **strong baseline, not by itself a novel
method.** The defensible novelty is the *composition*: (1) reframing the documented Moldova
failure as label shift and showing — with a clean ablation — that domain-invariance (DA-MoE)
*hurts* while a foundation backbone + balanced loss *helps*; (2) a **retrieval-based label
calibrator** over CXR-foundation features for cross-country severity (Rung 3); (3) a
**DANN-free severity MoE on frozen features** (Rung 5) that recovers the MoE idea without the
collapse; (4) **conformal severity intervals under domain shift** with country-conditional
coverage (Rung 6). If any rung yields a null, the paper reports it as a null — a negative result
on a real clinical task is still a contribution.

---

## 6. Results — The Honest Scoreboard

### 6.1 Rung 1 (RAD-DINO + light head) — ✅ RUN on Kaggle, 3 seeds, mode a2

3-seed mean Timika MAE / Pearson, vs **our locked A2 baseline** (honest target) and **Kantipudi
A2** (aspirational). Verdict from the bootstrap 95% CI (worst-seed bound) vs the locked baseline.

**Plain MSE (isolates the backbone):**

| Country | Timika MAE | Pearson | ALP MAE | Cavity AUC | Locked MAE | Kantipudi | Verdict vs locked |
|---------|-----------|---------|---------|-----------|-----------|-----------|-------------------|
| Romania | 20.39 ± 0.85 | 0.647 | 12.28 | 0.676 | 20.11 | 18.70 | within noise |
| Moldova | 22.13 ± 0.42 | 0.769 | 21.03 | 0.849 | 30.68 | 18.85 | **BEATS (−8.55)** |
| Kazakhstan | 17.93 ± 0.91 | 0.752 | 10.82 | 0.842 | 21.35 | 19.62 | **BEATS (−3.42)** |

**Balanced MSE / BMC (adds the imbalanced-regression loss):**

| Country | Timika MAE | Pearson | ALP MAE | Cavity AUC | Locked MAE | Kantipudi | Verdict vs locked |
|---------|-----------|---------|---------|-----------|-----------|-----------|-------------------|
| Romania | 20.53 ± 0.20 | 0.643 | 12.76 | 0.676 | 20.11 | 18.70 | within noise |
| Moldova | **21.79 ± 0.64** | **0.782** | 20.87 | 0.849 | 30.68 | 18.85 | **BEATS (−8.89)** |
| Kazakhstan | **17.73 ± 0.53** | **0.755** | 10.77 | 0.842 | 21.35 | 19.62 | **BEATS (−3.62)** |

**Brutally honest reading of Rung 1 — what is real and what is not:**

1. **Moldova is the headline.** 30.68 → 21.79 Timika MAE, an **8.9-point drop**, with the bootstrap
   CI nowhere near the baseline (worst-seed upper bound 24.1 ≪ 30.68); Pearson 0.70 → 0.78. The
   gap that defined the entire project is **mostly closed by the backbone swap alone.** Scientific
   finding: Moldova's failure was *not* irreducible — a CXR foundation model trained on diverse
   populations generalises where ImageNet-DenseNet collapsed to the mean.
2. **Kazakhstan now beats Kantipudi outright** (17.73 < 19.62). **Moldova does NOT** (21.79 vs
   18.85, +2.9) and **Romania does NOT** (20.5 vs 18.7, +1.8). So vs the aspirational target we
   are 1 / 3.
3. **The win is the backbone, not the loss.** BMC over plain MSE buys ~0.3–0.4 MAE and ~+0.01
   Pearson on the two hard countries and is slightly *worse* on Romania. Keep BMC (free, consistent
   on the hard tail), but report it honestly as a **marginal** effect, not a headline contribution.
4. **Romania is flat, and the reason is cavity, not ALP.** Romania's ALP MAE is excellent (≈12.8)
   but its Timika MAE is ≈20.5 — the extra ≈8 points are cavity misclassifications. **RAD-DINO's
   Romania cavity AUC (0.676) is actually *worse* than the locked DenseNet baseline (≈0.726).**
   The foundation backbone is **not** uniformly better — it is clearly better on Moldova/Kazakhstan
   cavity (0.85 / 0.84) but worse on Romania cavity. This is exactly why Rung 4 (cavity upgrade)
   targets Romania.
5. **Rung 1 is a linear/MLP probe on frozen features — a strong baseline, not yet "the method."**
   The publishable novelty must come from Rungs 2–6 adding measurable lift on top of this floor.
6. **Caveats:** single train-val split per seed (3 seeds); CIs are test-set bootstrap; the 5,010
   subsample is not Kantipudi's exact set; comparisons are primarily ours-vs-our-locked-baseline
   (the only apples-to-apples anchor), with Kantipudi as an aspirational reference.

**Verdict:** Rung 1 cleared the pre-registered kill-criterion decisively (beats the locked
baseline on 2/3 countries, within noise on the 3rd). The foundation-backbone thesis is validated;
Rungs 2–6 proceed.

### 6.2 Final headline run — ✅ RUN (5 seeds × M=10 ensemble × 7 rungs × 4 modes)

Full per-mode scoreboard with paired-bootstrap CIs is in
`baseline_runs/agentic_runs/AGENTIC_RESULTS.md`. Headline numbers (5-seed mean ± std Timika MAE,
Δ vs locked baseline; **bold** = paired-bootstrap CI excludes 0):

| Mode \ Country | Romania (lock) | Moldova (lock) | Kazakhstan (lock) |
|---|---|---|---|
| **A1 best+spat-cav** | 19.65 ± 0.33 (**−7.19**) | 21.05 ± 0.89 (**−11.71**) | 17.89 ± 0.44 (**−3.98**) |
| **A2 best+spat-cav** | **19.62 ± 0.52 (−0.49)** | 21.01 ± 0.41 (**−9.67**) | **16.57 ± 0.34 (−4.78)** |
| **A2 best+TTA** | 20.00 ± 0.68 (−0.11) | 20.77 ± 1.03 (**−9.91**) | 18.95 ± 1.10 (**−2.40**) |
| **A3 best+TTA** | 21.69 ± 0.37 (WORSE) | **20.45 ± 0.16 (−5.71)** | 19.64 ± 0.34 (**−2.26**) |
| **Fusion best+spat-cav (headline)** | **19.12 ± 0.49 (−0.99)** | **21.59 ± 0.60 (−4.57)** | **16.74 ± 0.29 (−4.61)** |
| **Fusion best+TTA (best Moldova)** | 20.22 ± 0.62 (within) | **19.98 ± 0.26 (−6.18)** | 18.84 ± 0.57 (**−2.51**) |

**Headline single config** = `Fusion · agentic_best_spatcav` (R3 retrieval + R6 conformal ensemble
+ R4b spatial cavity). **Best per country**: Rom 19.12 (Fusion+spat-cav), Mol 19.98 (Fusion+TTA),
Kaz 16.57 (A2+spat-cav). Vs Kantipudi A2 paper (18.70 / 18.85 / 19.62): **beats Kazakhstan by 3.05
points; within 0.42 on Romania; within 1.13 on Moldova**.

Paired-bootstrap on A2 and A3 (2000 reps, paired on `image_id`, reference R1 MSE).

A2: `agentic_best` is **significant on all three countries** (Δ −1.11 Rom *** / −1.29 Mol *** /
−0.93 Kaz ***). Adding spatial cavity bumps Kazakhstan further (Δ −1.59 **). TTA is significant
on Moldova (Δ −1.17 ***) but **significantly hurts Kazakhstan** (Δ +0.84 *).

A3 (amplifies the country-conditional pattern): R3 retrieval is highly significant on Moldova
(Δ −1.48 ***) and Kazakhstan (Δ −2.16 ***). `agentic_best_tta` on A3 delivers the **largest
single-mode Moldova reduction in the entire run** (Δ −3.27 CI [−3.96, −2.55]) but **significantly
hurts Romania** (Δ +2.34 *). R6 conformal on A3 helps Kazakhstan (Δ −2.62 ***) but hurts Romania
(Δ +1.02 *). The A3 mode therefore exposes the country-conditional trade-off most starkly — the
deployment recommendation of Fusion mode (which averages A2 and A3) comes directly from the
A3-Romania regressions being significant.

**Per-rung findings (each is a full ablation row):**
- **R1 backbone:** biggest single lift — backbone, not loss.
- **R2 TTA:** Moldova-only win (best Moldova number, 19.98 in Fusion+TTA); significantly degrades
  Kazakhstan (paired bootstrap Δ +0.84). Refines diagnosis: Moldova = label + minor covariate
  shift.
- **R3 retrieval:** most consistent positive rung; significant on Moldova/Kazakhstan in paired
  bootstrap; never a regression.
- **R4 cavity-focal:** **negative result** — focal harms AUC on the already-balanced cavity split.
  Romania cavity bottleneck is intrinsic feature quality on Romania-specific cavity morphology.
- **R4b spatial cavity:** **the correct fix** — attention-pooled per-patch scoring closes the
  cavity-AUC gap (Romania 0.679 → 0.721, Moldova 0.851 → 0.872, Kazakhstan 0.846 → 0.877). Most
  importantly, *also* lifts the regression slope (A2 Rom 0.596 → 0.666), the only intervention
  that does.
- **R5 MoE:** **null result** — MoE on frozen features ≈ MLP with more parameters.
- **R6 ensemble M=10 + conformal:** small consistent MAE bump + calibrated intervals (marginal
  coverage 0.88–0.92 under domain shift). Moldova under-coverage on Fusion (0.83) is the
  residual covariate-shift TTA cannot fully address.
- **R7 isotonic slope calibration:** **null on real data** — synthetic test promised +0.10–0.18
  slope; real data delivers +0.00–0.02. Honest negative finding.
- **A1 mode:** spatial RAD-DINO ALP head crushes A1 locked by 7–12 Timika points; beats
  Kantipudi's A1 paper numbers on all three countries.

### 6.3 DA-MoE — ❌ ABANDONED (the negative result that motivated the pivot)

The original Domain-Adversarial MoE (shared DenseNet121 trunk + K experts + gate + DANN
gradient-reversal + critic + load-balance, two-phase training) was run on all four modes and
**failed**, worse than the locked single-network baselines on every mode except a marginal A1 gain;
the direct-Timika mode (a3) **collapsed to near-constant output (Pearson ≈ 0)**.

Scoreboard (3-seed mean Timika MAE / Pearson):

| Mode | Country | DA-MoE | Locked baseline | Verdict |
|------|---------|--------|-----------------|---------|
| a2 | Moldova | 33.34 / 0.458 | 30.68 / 0.697 | worse |
| a2 | Kazakhstan | 28.72 / 0.588 | 21.35 / 0.648 | worse (+7.4) |
| a3 | Romania | 29.58 / **0.168** | 20.26 / 0.699 | **collapsed** |
| a3 | Moldova | 36.46 / **0.024** | 26.16 / 0.759 | **collapsed** |
| a3 | Kazakhstan | 32.62 / **0.184** | 21.90 / 0.692 | **collapsed** |
| fusion | Moldova | 33.39 / 0.434 | 30.68 / 0.697 | worse |

**Root cause (proven by ablation):** DANN at λ_max = 1.0 destroys the trunk's regression features.
(a) Validation error rises *monotonically* as λ ramps up. (b) The `--no-dann` ablation recovers
signal to near-baseline (e.g. A3 Moldova 36.38/0.05 → **30.81/0.74**; A2 Moldova 33.33/0.44 →
**27.27/0.74**). The critic alone (DANN still on) stays collapsed, so the critic is innocent.
Mechanism: because severity correlates with country, an over-strong country adversary suppresses
exactly the severity signal needed for regression — the label-shift theorem in action. A1 was the
only survivor because its prediction came from fixed detection geometry, barely using the trunk.

**The one glimmer that survived into the new plan:** `A2 --no-dann` on Moldova improved over the
baseline (27.27 / 0.74 vs 30.68 / 0.697) from the *mixture + critic alone, with the adversary
disabled* — direct motivation for keeping a **DANN-free** severity MoE as Rung 5.

**For the paper:** DA-MoE is best written as a **cautionary negative result** ("when *not* to use
domain-adversarial training for medical severity regression") and as the empirical motivation for
the frozen-foundation + label-calibration pivot. The four implementation bugs in the original
`train_da_moe.py` (λ default 1.0; GRL ramp starting in phase 1; phase-2-only checkpoint selection;
MSE selection rewarding mean-collapse) compounded the failure but are *secondary* to the
conceptual mismatch — fixing them would at best recover baseline-level performance, which RAD-DINO
already exceeds.

---

## 7. Why Moldova Fails — Root-Cause Analysis (and how Rung 1 resolves it)

### 7.1 Label-distribution shift (dominant cause, confirmed; now largely mitigated)

Moldova test ALP mean ≈ 39.8 vs train-pool ≈ 26.0 (gap +13.8). A regressor trained on the pool
regresses toward the training mean and under-predicts Moldova. Smoking gun on the **locked
DenseNet baseline**: Moldova ALP MAE ≈ 24.7 vs a naïve "always predict the pool mean" ALP MAE
≈ 25.5 — nearly identical, i.e. the head had **collapsed to the prior** with essentially zero
genuine generalisation. Romania/Kazakhstan sat far below their naïve baselines, confirming the
model genuinely worked there. **This is not a code bug; it is the cross-country generalisation
failure the project targets.**

**Resolution (✅ RUN):** RAD-DINO + light head drops Moldova ALP MAE to ≈20.9 and Timika MAE to
21.79 — well below the prior-collapse floor. The label shift still exists, but the foundation
features are robust enough that the head no longer collapses to the mean. Rung 3 (retrieval) is
designed to push this further by anchoring predictions to retrieved high-severity training labels.

### 7.2 Apex-clipping hypothesis (plausible secondary, unconfirmed)

Kantipudi notes TB lesions cluster at the lung apex. Our MedSAM lung crop (used by the *baseline*
pipeline) may crop the apex tighter than their ResNet18-UNet, under-feeding apex lesions to the ALP
regressor — worst for the most severe (Moldova) cases. Consistent with the data, not directly
confirmed (would need crop-level visual comparison). **Note:** the agentic pipeline feeds RAD-DINO
*whole images* (no MedSAM crop), so if apex-clipping contributed to the baseline's Moldova failure,
the agentic pipeline sidesteps it — another reason Rung 1 helps.

### 7.3 Unknown split (secondary)

Kantipudi never published patient-level splits. A different random patient-disjoint split shifts
which images land in test and can move the effective test ALP mean by several points. Our LOCO
split is honest but not theirs.

### 7.4 Cavity AUC offset (now nuanced)

The locked DenseNet baseline's cavity AUC ran ~0.07 below the paper uniformly (≈0.73/0.81/0.78 vs
0.80/0.88/0.85). With RAD-DINO the picture *changes*: cavity AUC improves on Moldova (0.849) and
Kazakhstan (0.842) but **drops on Romania (0.676)**. So cavity is no longer a uniform offset — it
is now a Romania-specific weakness, which Rung 4 targets directly.

---

## 8. Architecture Details (for the Methods Section)

### 8.1 Frozen backbones — `src/components/backbones.py`
- `RadDinoBackbone` — HF `AutoImageProcessor` + `AutoModel`, frozen; `embed()` → 768-d
  `pooler_output` (CLS); `embed_grid(grid=7)` → pooled patch-token grid `[N, 49, 768]` for A1
  (drops CLS, reshapes patch tokens to their native square grid, `adaptive_avg_pool2d` to 7×7).
- `TorchXRVBackbone` (1024-d) and `DenseNetBackbone` (1024-d, = Kantipudi backbone) — fallbacks /
  the "backbone-only" control; txrv also supports `embed_grid`.
- `build_backbone(name, device, model_id)`; all expose `embed(list[PIL.Image]) → [N, dim]`.

### 8.2 Feature heads — `src/components/feature_heads.py`
- `RegressionHead`: MLP → sigmoid → [0,1] (×100 = ALP%, ×140 = Timika).
- `ClassifierHead`: MLP → 2 logits (cavity).
- `MoERegressionHead` (Rung 5): K experts (MLP→1 each) + gate (MLP→K, softmax); output =
  `sigmoid(Σ_k w_k · expert_k)`; `forward(return_gate=True)` exposes gate weights. **No trunk, no
  DANN.**
- `CriticHead` (Rung 5): MLP → sigmoid reliability in [0,1], trained to regress `1 − |error|`.
- `SpatialALPHead` (A1): shared scorer over a patch grid `[B, P, D]` → per-cell involvement in
  [0,1] → mean over cells → ALP (a differentiable analogue of lesion-area / lung-area).

### 8.3 Losses — `src/training/losses.py`
- `BMCLoss` (Balanced MSE, Ren et al. 2022): batch-Monte-Carlo, learnable `noise_sigma`; add its
  parameters to the optimiser. `make_reg_loss("mse"|"bmc")`.
- `lds_weights` (LDS, Yang et al. 2021): inverse smoothed-density per-sample weights (optional
  weighted-MSE variant).
- `class_balanced_weights` (Cui et al. 2019, effective-number) + `focal_ce` (focal cross-entropy)
  for the Rung 4 cavity upgrade.

### 8.4 Retrieval — `src/components/retrieval.py`
- `RetrievalCalibrator(k, temperature)`: `fit(feats, labels)`; `retrieve()` = softmax-over-cosine
  kNN label estimate; `calibrate(feats, head_pred, α)`; `select_alpha(val…)` picks α on validation
  by MAE. α=0 = pure head (ablation).

### 8.5 Test-time adaptation — `src/training/tta.py`
- `FeatureStandardizer` (inductive/transductive z-score) and `coral_align(source, target)` (CORAL
  whitening + recolouring).

### 8.6 Conformal + ensemble — `src/components/conformal.py`
- `ensemble_mean(preds)` → (mean, inter-member std); `conformal_quantile(cal_true, cal_pred, α)`
  (finite-sample corrected); `split_conformal(...)` → coverage + mean width on test.

### 8.7 Orchestration — `src/training/train_agentic.py`
Mode-aware (a2/a3/fusion/a1) and rung-aware trainer on cached features. Per (mode, country, seed):
trains the base predictors, then evaluates each requested rung as an ablation row (with bootstrap
CIs and a verdict vs the locked baseline), writes per-image predictions for later paired
significance tests. Reuses `make_country_split`, `make_balanced_cavity_split`, `evaluate_split`,
`LOCKED_BASELINE`, `KANTIPUDI_A1/A2/A3`, `paired_bootstrap_delta`.

### 8.8 Evaluation — `src/evaluation/eval_tbportals.py`
Timika/ALP regression metrics (MAE, MAE%, RMSE, Pearson, Spearman, R²), cavity metrics
(AUC/F1/precision/recall), 1000-sample paired percentile bootstrap CIs, `paired_bootstrap_delta`
for model-vs-model significance, and the per-mode `LOCKED_BASELINE` / `KANTIPUDI_*` reference
dicts. `Timika = ALP_pred·100 + 40·(cavity_prob > τ)`.

---

## 9. Data Pipeline
- **Manifest** (`src/data/tbportals.py`): join CXRs + Annotations on `imagingstudy_id`, aggregate
  sextants, normalise cavity tokens, seed-42 subsample to 5,010; `make_country_split`,
  `make_balanced_cavity_split`, `assert_no_patient_leakage`.
- **Feature cache** (`scripts/cache_features.py`): runs a frozen backbone over the manifest, writes
  `.npz` (`image_id`, `features`, `backbone`, `patch_grid`); `--patch-grid N` for A1; idempotent.
  `load_features()` handles both CLS `[N,D]` and grid `[N,P,D]` caches.
- **Image loading** (`src/data/tbportals_dataset.py` `_load_image`): PNG/JPEG, extracted DICOM, and
  zip-resident DICOM.
- **Baseline pipeline only** still uses MedSAM lung crops (Dice 0.886 Montgomery / 0.959 Shenzhen
  vs paper 0.96±0.02); **the agentic pipeline does not crop** — RAD-DINO sees whole images.

---

## 10. File Map (current, agentic direction)

```
dl-project-codebase/
├── src/
│   ├── components/
│   │   ├── baseline_paper.py     # ALPRegressor/CavityClassifier/TimikaRegressor (baseline)
│   │   ├── da_moe.py             # ❌ DA-MoE (abandoned; negative result)
│   │   ├── backbones.py          # 🆕 RAD-DINO / txrv / densenet frozen backbones (+patch grid)
│   │   ├── feature_heads.py      # 🆕 Regression/Classifier/MoE/Critic/SpatialALP heads
│   │   ├── retrieval.py          # 🆕 Rung 3 retrieval calibrator
│   │   └── conformal.py          # 🆕 Rung 6 conformal + ensemble
│   ├── data/
│   │   ├── tbportals.py          # manifest build + LOCO splits
│   │   └── tbportals_dataset.py  # image loading
│   ├── training/
│   │   ├── train_baseline_paper.py / train_a3_direct.py / train_a1_detect.py  # baselines
│   │   ├── train_da_moe.py       # ❌ DA-MoE trainer (abandoned)
│   │   ├── losses.py             # 🆕 BMC, LDS, focal/class-balanced (Rung 1/4)
│   │   ├── tta.py                # 🆕 Rung 2 feature standardization + CORAL
│   │   └── train_agentic.py      # 🆕 mode/rung-aware orchestrator
│   └── evaluation/
│       └── eval_tbportals.py     # metrics, bootstrap CIs, LOCKED_BASELINE + Kantipudi refs
├── scripts/cache_features.py     # 🆕 frozen-feature cache (CLS + patch-grid)
├── notebooks/
│   ├── tbportals_agentic.ipynb            # Rung 1 (RUN; logs in repo, superseded)
│   └── tbportals_agentic_{a2,a3,fusion,a1}.ipynb  # ✅ full rung stack (RUN; logs in repo)
└── baseline_runs/
    ├── BASELINE_COMPARISON.md             # locked 3-seed baselines
    ├── MoE/MOE_RESULTS_REPORT.md          # ❌ DA-MoE negative result + ablations
    ├── a2_rungs/agentic_rung1.zip         # ✅ Rung 1 results (the initial validation)
    └── agentic_runs/                      # ✅ full 6-rung run (A1/A2/A3/Fusion zips)
        └── AGENTIC_RESULTS.md             # full per-mode rung scoreboard + theory
```

---

## 11. Summary of Contributions vs Gaps

### Delivered (✅)
1. **Faithful Kantipudi replication** (A1/A2/A3); Romania/Kazakhstan match within noise. **Our best
   approach is A3, not A2** — the flip is driven entirely by Moldova (A2 inherits the ALP head's
   collapse; A3 degrades more gracefully).
2. **Quantified the Moldova failure** as label-distribution shift (+13.8 ALP gap; baseline ALP head
   collapses to the prior).
3. **DA-MoE negative result** with a clean ablation proving the cause (over-strong DANN destroys
   severity features under label shift) — a publishable cautionary finding.
4. **Pivot to a frozen CXR foundation backbone** and **Rung 1 RUN**: RAD-DINO + balanced regression
   closes most of the Moldova gap (30.68 → 21.79), beats the locked baseline on 2/3 countries, and
   beats Kantipudi outright on Kazakhstan.
5. **Full 6-rung agentic pipeline RUN on all 4 modes** (3 seeds × 3 countries × 6–8 rungs). Best
   single config: **fusion · rung6_conformal → Rom 19.96 / Mol 22.42 / Kaz 16.86**. Beats locked
   baseline on Mol/Kaz with bootstrap-CI significance across every rung in fusion mode; ties
   locked on Romania.
6. **Beat Kantipudi on Kazakhstan by 2.76 points** (16.86 vs 19.62); within 1.25/1.22 on
   Romania/Moldova.
7. **A1 spatial RAD-DINO head** crushed A1 locked baseline by 7–11 Timika points and **beat
   Kantipudi A1 on all 3 countries** — a novel detection-free regional ALP estimator.
8. **Two publishable negative results** from rung ablations: R4 cavity-focal harms an
   already-balanced split; R5 MoE on frozen features is a null result (DomainBed transfer).

### Open / honest gaps
1. **Romania cavity is the only remaining bottleneck** (RAD-DINO Romania cavity AUC 0.675).
   R4 was the wrong intervention (focal+CB on already-balanced split). **Next attack: spatial
   cavity head over the RAD-DINO patch grid** — cavities are localized, so global CLS pooling
   washes them out; per-patch scoring + attention pooling should preserve the signal.
2. **Headline configuration is "agentic_best" = R1_bmc + R3_retrieval + R6_ensemble** (drops R4
   to avoid the cavity-focal harm). To be run on 5 seeds × M=10 ensemble for tight final CIs +
   paired bootstrap significance vs locked baseline for every claimed gain.
3. **We have not beaten Kantipudi on Romania (19.95 vs 18.70) or Moldova (20.07 vs 18.85)** —
   only Kazakhstan. The 1.2-point residual on Moldova is *probably* split/dataset-version-driven
   (their exact patient split is unpublished), not a failure of method.
4. **Qualitative + quantitative analysis for the paper** still to do: per-mode paired-bootstrap
   Δ-MAE grids with CI95, conformal coverage decomposition (per-country + per-severity-band),
   cavity calibration (Brier, ECE, reliability diagrams), spatial ALP heatmaps for A1, retrieval
   neighbour panels, error-vs-severity scatter, failure-case panels.

---

## 12. Honest Paper Narrative (recommended)

**Primary thesis (defensible, evidence-backed today):** *"On a documented cross-country TB-severity
generalisation failure, we show that domain-adversarial training (DA-MoE) actively harms
performance — because the shift is in label space (plus minor covariate components), not style —
whereas a frozen chest-X-ray foundation model with a balanced-regression head closes most of the
gap. On top of this strong foundation we add an agentic stack — retrieval-based label calibration,
test-time feature adaptation, a severity mixture-of-experts, and conformal severity intervals —
and report, with paired bootstrap significance, exactly which components add lift and which do
not. Our best fusion configuration achieves Timika MAE 19.96/22.42/16.86 on Romania/Moldova/
Kazakhstan, beating our locked DenseNet baseline (20.11/30.68/21.35) on Moldova/Kazakhstan with
bootstrap-CI significance and beating Kantipudi (18.70/18.85/19.62) outright on Kazakhstan, with
conformal coverage 0.87–0.90 under domain shift. Negative results — Rung 4 class-balanced focal
on an already-balanced cavity split, and Rung 5 MoE on frozen features — are reported as such,
and motivate a follow-up spatial cavity head as the principled fix for the remaining Romania
cavity bottleneck."*

This narrative is honest about (a) the negative DA-MoE result, (b) the backbone (not the loss)
being the main lift, (c) not yet beating the paper everywhere, and (d) reporting rung nulls as
nulls. It positions the contribution as a *method + rigorous empirical study* rather than an
over-claimed single-number win. **Do not let the paper claim the agentic rungs beat the paper
until those experiments are RUN and the CIs support it.**

---

## 13. Published-method baselines (2026-05-30 addition for ICONIP)

Dr. Taj round-3 feedback was: *"baseline comparisons need to be against
published works, not internal backbone ablations"*. We now report three
families of published baselines on our exact LOCO splits — all numbers
are Timika MAE.

| Method | Romania | Moldova | Kazakhstan | Source |
|---|---:|---:|---:|---|
| TXV-only (no cavity)              | 37.15 | 37.47 | 31.08 | Cohen 2020 (TorchXRayVision) |
| CheXzero-only (cavity × 40)       | 32.00 | 35.03 | 27.22 | Tiu 2022 (CheXzero) |
| TXV + CheXzero (composite plug-in)| 24.02 | 26.37 | 27.76 | Cohen 2020 + Tiu 2022 |
| GroupDRO on TXV (5 seeds)         | 23.54 ± 1.85 | 27.58 ± 1.14 | 25.30 ± 0.25 | Sagawa 2020 |
| Importance-Weighted reg. (5 seeds)| 22.82 ± 1.82 | 28.18 ± 1.37 | 25.25 ± 0.78 | Shimodaira 2000 |
| K24 method, our replication (5s)  | 26.74 / 20.11 / 19.96 (A1/A2/A3) | 32.86 / 30.68 / 28.18 | 27.69 / 21.35 / 19.46 | Kantipudi 2024† |
| **Ours: Agentic Fusion + SpatCav**| **19.12** | **21.59** | **16.74** | this work |

† We replicate K24's three approach families locally with 5 seeds because
the original paper does not release patient-disjoint splits. Asterisked
numbers in the paper are our trained reproductions, not K24's published
table.

**Cohort context** (see `iconips_Paper/tables/cohort_summary.csv`):

| Country | Images | Patients | ALP μ/σ | Cavity % | Timika μ/σ |
|---|---:|---:|---:|---:|---:|
| Georgia | 1414 | 1412 | 26.6/18.3 | 50.4 | 46.8/29.9 |
| Ukraine | 1316 | 1306 | 29.8/23.5 | 38.0 | 45.0/35.1 |
| Belarus | 1052 | 798  | 20.3/23.1 | 24.1 | 30.0/34.3 |
| **Moldova†**   |  589 | 589 | 39.8/29.9 | 32.8 | 52.9/41.8 |
| **Kazakhstan†**|  399 | 399 | 22.6/22.8 | 39.8 | 38.5/36.7 |
| **Romania†**   |  220 | 169 | 31.9/22.9 | 65.0 | 57.9/32.3 |
| **TOTAL**      | 5010 | 4691 | 27.6/23.6 | 39.3 | 43.3/35.3 |

The cohort table makes the shift explicit: Romania has **65 %** cavity
prevalence vs Kazakhstan's 40 %; Moldova has the highest ALP mean
(39.8 vs the 27.6 cohort average). These are exactly the label-shift
quantities §3 (shift-decomposition framework) decomposes empirically.

All outputs are in `baseline_runs/agentic_runs/paper/`; aggregated tables
in `baseline_runs/agentic_runs/paper/_extract/aggregated_baselines.md`
and `iconips_Paper/tables/cohort_summary.csv`.

---

*End of context document. Rewritten 2026-05-28 to reflect the agentic 6-rung pipeline (MoE = Rung 5,
DANN removed). Rung 1 numbers are RUN; Rungs 2–6 and modes a3/fusion/a1 are implemented but unrun —
labelled accordingly above. §13 added 2026-05-30 with published-baseline numbers + cohort breakdown
in response to Dr. Taj feedback round 3.*
