# XAI × Edge Deployment Survey — TB Timika Severity Pipeline

Three-part literature survey + expansion plan for taking the **frozen RAD-DINO / Timika** TB severity model (ALP regression + 49-token spatial-attention cavity head + deep-ensemble/split-conformal) to **explainable, on-device deployment** in LMIC TB clinics — **without compromising the cross-country (LOCO) generalization novelty**.

> Written against the **paper architecture** (RAD-DINO ViT-B), not this repo's MoE/MedSAM code. Reconcile file paths once the RAD-DINO repo is copied over.

## Contents

| File | What |
|---|---|
| [01_survey_xai.md](01_survey_xai.md) | **Survey A — Explainable AI for medical imaging** (attribution, transformer/attention explainability, prototype & concept models, faithfulness/localization evaluation, UQ-as-explanation, regulatory frame). |
| [02_survey_edge.md](02_survey_edge.md) | **Survey B — Edge / low-compute deployment** (quantization, pruning + ViT token reduction, distillation, efficient backbones, runtimes across iPad/Android/browser/CPU, real TB point-of-care deployments, federated + system efficiency). |
| [03_survey_integration_and_plan.md](03_survey_integration_and_plan.md) | **Survey C v2 — Integration + expansion plan, LOCO-centered** (XAI↔edge↔cross-country-robustness; white-space **WS1–WS6**; technique→pipeline mapping with a per-country-coverage column; P0/P1/P2 roadmap with LOCO re-validation gates; **6 candidate contributions**, flagship = compression that preserves the cross-country gap). |
| [03_survey_integration_and_plan_v1.md](03_survey_integration_and_plan_v1.md) | Superseded first-pass Survey C (kept for reference). |
| [04_gaps_and_contributions.md](04_gaps_and_contributions.md) | First-pass research gaps + 5 candidate contributions + completeness-critic findings. |
| [05_references.md](05_references.md) | ~242 citations grouped by section — adversarially fact-checked (incl. 60 compression×robustness refs from the follow-up). |
| [06_corrections_and_errata.md](06_corrections_and_errata.md) | Factual errata for Surveys A/B — **6 corrections applied** to the files; full list with sources here. |
| [00_provenance_and_verification.md](00_provenance_and_verification.md) | Verification flags the synthesis agents handled (refuted/corrected claims). |

## How it was produced
29-agent workflow (run `wf_fc5c9a5f-79e`): 12 web-research specialists → 12 adversarial citation fact-checkers → 3 synthesis authors → completeness critic + novelty-gap finder. 12 sections, all verified.

## Status — COMPLETE
- ✅ Surveys A, B (XAI, edge) + first-pass gaps/contributions/references.
- ✅ Follow-up: compression × cross-country-robustness research thread (4 verified sections); **Survey C v2** reframed around the no-compromise-LOCO constraint (WS1–WS6, distillation falsification protocol, CORAL×quantization, composite-score conformal, energy axis); **6 factual corrections applied** to A/B.
- 🔜 Open when the RAD-DINO repo is copied over: reconcile the C3 roadmap's concrete steps to real file paths.
