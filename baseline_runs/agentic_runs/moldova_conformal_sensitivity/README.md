# Moldova Conformal Coverage Sensitivity

This folder contains a Moldova-specific conformal coverage sensitivity analysis.

## Motivation

The original conformal rung targeted nominal 90% uncertainty coverage, but Moldova undercovered:

| Method | Coverage | Width | MAE |
|---|---:|---:|---:|
| rung6_conformal | 0.829 | 83.678 | 22.344 |

This indicates that the model was too confident under Moldova country shift.

## Main Sensitivity Finding

The TTA-adapted predictor improved both predictive accuracy and uncertainty coverage:

| Method | Coverage | Width | MAE |
|---|---:|---:|---:|
| agentic_best_tta | 0.881 | 82.346 | 19.980 |
| agentic_best_tta + 1.10 interval multiplier | 0.914 | 90.581 | 19.980 |

Compared with the original conformal rung, the selected sensitivity setting improves coverage from 0.829 to 0.914 and reduces MAE from 22.344 to 19.980.

## Interpretation

This is a sensitivity analysis, not a locked conformal method.

The 1.10 interval multiplier was selected after evaluating Moldova residuals, so it should be reported honestly as post-hoc evidence that Moldova coverage can be recovered with modest interval widening.

Recommended manuscript wording:

> Moldova showed undercoverage under the original conformal rung. A Moldova-specific sensitivity analysis showed that using the TTA-adapted predictor increased coverage from 0.829 to 0.881 at similar interval width and lower MAE. Applying a modest 10% interval inflation increased coverage to 0.914 with mean width 90.6. Because the inflation factor was selected post hoc, we report this as a sensitivity analysis rather than a locked calibration rule.

## Files

| File | Description |
|---|---|
| `results_agentic_fusion.csv` | Moldova rerun aggregate results |
| `moldova_conformal_width_sensitivity_per_seed.csv` | Per-seed interval multiplier sweep |
| `moldova_conformal_width_sensitivity_summary.csv` | Mean coverage/width/MAE by multiplier |
| `moldova_conformal_selected_sensitivity.csv` | Paper-relevant selected rows |
| `moldova_conformal_sensitivity_metadata.json` | Metadata and caveat |
