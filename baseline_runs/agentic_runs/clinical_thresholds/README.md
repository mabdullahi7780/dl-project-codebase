# Clinical Threshold Evaluation

This folder contains a clinical threshold evaluation for Timika severity prediction.

## Purpose

The main paper reports continuous Timika regression metrics such as MAE and Pearson correlation.
This analysis asks whether the model can also be used as a severe TB triage score.

For each held-out CXR, continuous Timika predictions were converted into severe-case decisions at:

- Timika >= 80
- Timika >= 100
- Timika >= 120

## Countries

The analysis covers the six adequately powered LOCO countries:

| Country | Test images |
|---|---:|
| Romania | 220 |
| Moldova | 589 |
| Kazakhstan | 399 |
| Georgia | 1,414 |
| Ukraine | 1,316 |
| Belarus | 1,052 |

Total held-out evaluation pool: 4,990 CXRs.

## Methods Compared

| Code name | Paper-friendly label |
|---|---|
| `rung1_mse` | R1: MSE baseline |
| `rung1_bmc` | R1: Balanced-MSE |
| `rung4b_spatial_cavity` | R4b: spatial cavity head |
| `agentic_best_spatcav` | Spatial cavity-aware agentic stack |

## Metrics

For each threshold, country, method, and seed, the analysis reports:

- sensitivity
- specificity
- precision
- F1
- false-negative rate
- AUROC
- AUPRC
- MAE

Country-macro summaries average each country equally.

## Main Interpretation

The spatial cavity-aware agentic stack improves continuous Timika MAE and provides comparable severe-case triage discrimination.

At raw Timika decision thresholds, the gains are clinically useful but modest:

- At Timika >= 80 and Timika >= 100, the final method slightly improves precision/specificity and F1 over the MSE baseline.
- AUROC/AUPRC are similar to the baseline, suggesting the model retains severe-case ranking performance while improving continuous severity error.
- Timika >= 120 is very rare in several countries, so it should be treated as supplementary rather than a primary endpoint.

This analysis is useful for the paper because it translates regression results into a clinically interpretable severe-disease triage setting.

## Files

| File | Description |
|---|---|
| `clinical_threshold_metrics_per_seed.csv` | Per-country, per-seed threshold metrics |
| `clinical_threshold_summary_by_country.csv` | Mean metrics per country/method/threshold |
| `clinical_threshold_country_macro_summary.csv` | Six-country macro summary for selected methods |
| `paper_ready_clinical_thresholds.csv` | Selected paper-ready table |
| `clinical_threshold_prevalence_by_country.csv` | Severe-case prevalence by country and threshold |
| `clinical_threshold_all_rungs_macro.csv` | Macro threshold metrics for all rungs |
| `clinical_threshold_final_vs_mse_delta.csv` | Final method minus MSE baseline deltas |
| `clinical_threshold_metadata.json` | Analysis metadata and caveats |

## Recommended Paper Wording

In a clinical threshold analysis, we evaluated whether continuous Timika predictions could identify severe disease at thresholds of 80, 100, and 120 points. Across six held-out countries, the spatial cavity-aware agentic stack retained strong severe-case discrimination while improving continuous severity error. At the Timika >= 100 threshold, it achieved country-macro AUROC around 0.92 and slightly improved F1 relative to the MSE baseline. Timika >= 120 results should be interpreted cautiously because very severe cases were rare in several held-out countries.
