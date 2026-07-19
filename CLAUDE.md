# CLAUDE.md — TB Portals Timika Scoring (active workstream: WS-C)

Project working notes for Claude Code. The active task is **WS-C: Cavity Grounding**
(plan: `../ws-c.md`, also in `refinement.md` / `plan.md` Sprint 1). This file records
decisions already made so they are not re-derived each session.

## Project one-liner
Cross-country TB severity (Timika) scoring from CXRs on the **TB Portals** corpus, under a
patient-disjoint **Leave-One-Country-Out** protocol. Paper: *"Decomposing the Cross-Country
Generalisation Gap in TB Severity Scoring"* (ICONIP 2026). Method = frozen **RAD-DINO**
encoder + composable rungs (R1 balanced-MSE, R2 CORAL TTA, R3 retrieval calibration,
R4b spatial cavity head, R6 ensemble+conformal; R5 MoE / R7 isotonic are diagnostic nulls).
`Timika = ALP + 40·1[cavity]`, ALP∈[0,100], score∈[0,140].

## WS-C goal
Turn Fig. 4 (cavity attention) from an *illustrative* panel into a *measured, zone-level
localization* claim, and add the cavity-probability calibration the paper omits (M7).
Deliverables: **D1** calibration (ECE/Brier/reliability, global CLS vs spatial head),
**D2** zone-level localization (zone-AUC, pointing game, energy concentration vs sextant
labels), **D3** robustness/controls, **D4** updated Fig. 4 + §5.5 paragraph.

## C1.0 SCHEMA GATE — RESOLVED: **PASS** (verified locally 2026-06-26)
The raw annotation CSV **does** carry a per-sextant location column, so D2 is alive.
- `TB_Portals_CXR_Manual_Annotations_August_2023.csv` (local, 25,335 rows, 8,764 images):
  column **`sextant`** ∈ {Upper,Middle,Lower}×{Left,Right} (+ `None` for image-level rows).
  Cavity per row from `smallcavities`/`mediumcavities`/`largecavities` > 0.
- Recovery = per `(imagingstudy_id, sextant)` take `max` cavity across raters → 6-dim
  binary vector (zone order: 0 UL, 1 UR, 2 ML, 3 MR, 4 LL, 5 LR).
- **Validated against real data:** any-sextant-cavity vs manifest `cavity` flag = **100%
  agreement** (n=1117 held-out). Coverage Romania 97.3% / Moldova 98.1% / Kazakhstan 81.5%.
  Per-zone pos-rate UL 0.22, UR 0.28, ML 0.12, MR 0.13, LL 0.03, LR 0.04 — **confirms the
  apex prediction** (pre-registration §8). Pointing-game chance baseline ≈ 1.85/6 ≈ 0.31.

## The join chain (non-obvious — do not re-derive)
Manifest `image_id` is the **basename of `series_instance_content_url`** (final DICOM UID,
strip directories and `.dcm`), NOT `imagingstudy_id`. All 5,010 manifest rows match.
```
manifest.image_id  ==  basename(CXRs.series_instance_content_url).replace('.dcm','')
CXRs.imagingstudy_id  →  Annotations.imagingstudy_id  →  per-image 6-dim sextant matrix
```
Files (all local): manifest `notebooks/tbportals_manifest_paper.csv` (5,010 rows, 8 countries);
bridge `TB_Portals_CXRs_August_2023.csv` (`imagingstudy_id`↔`series_instance_content_url`);
labels `TB_Portals_CXR_Manual_Annotations_August_2023.csv`.

## Radiological L/R flip (D2 correctness)
CXRs are displayed facing the patient: the **patient's right lung is on the image's left**.
The 7×7 → 6-zone map (3 rows × 2 cols, patch centre falls in a zone) must flip image-columns
vs sextant L/R. C4 control: flipping the map must *lower* zone-AUC; getting it backwards
inverts the result.

## Local vs Kaggle (what runs where)
- **Local (python3.11 — `python3` is 3.14 and lacks pandas):** sextant recovery + manifest
  join (done/validated); all D1 calibration metrics; D2 zone-map + metrics + C4 controls on
  **synthetic attention** for unit tests. No GPU needed.
- **Kaggle only (heavy artefacts not in repo):** RAD-DINO **patch-grid** feature cache
  (`scripts/cache_features.py --patch-grid 7`), trained `SpatialCavityHead` weights
  (`train_agentic.py --save-heads`, `cavity_head=spatial`), and therefore the real
  **attention vectors** → the final zone-AUC numbers. `SpatialCavityHead.forward(return_attn=True)`
  already returns `attn [B,49]` (`src/components/feature_heads.py`).

## Conventions / discipline (never skip)
- Every localization number printed **with its chance baseline** and a **bootstrap 95% CI**
  (reuse `bootstrap_ci` from `src/evaluation/eval_tbportals.py`).
- **Pre-registration** (ws-c.md §8): zone-AUC > chance on ≥2/3 countries, strongest on
  Romania; spatial ≥ global on Brier/ECE; attention concentrates in upper lobes.
- **Report nulls as nulls.** If zone-AUC ≈ 0.5 or the L/R-flip control doesn't drop AUC,
  the localization claim is reported unsupported and Fig. 4 reverts to illustrative.
- Claims are **zone-level, segmenter-free, approximate** — never claim pixel localization.
- Run env: `python3.11`.

## WS-C file map
- New: `scripts/ws_c_cavity_grounding.py` — recovery + zone map + D1/D2 metrics + C4 controls + figure.
- New: `tests/test_ws_c_cavity_grounding.py` — synthetic-data unit tests for the pure fns.
- Touch: `src/data/tbportals.py` — `aggregate_sextant_cavities()` (keep zones, don't collapse).
- Outputs: `ws_c/cavity_grounding_metrics.{json,csv}`, `ws_c/reliability.pdf`, updated `iconips_Paper/figures/fig_cavity_attention.*`.
