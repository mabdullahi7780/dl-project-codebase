# WS-SP — Paper draft (fill placeholders `<...>` from ws_sp/selective_prediction.csv)

## Subsection: Selective Prediction / Abstention

> **Does the uncertainty carry actionable information?** A conformal interval is only
> clinically useful if its per-image width tracks error. Because our split-conformal
> width is constant across images, we instead rank test cases by the deep-ensemble
> inter-member standard deviation (Sec.~\ref{sec:method}, R6) — a genuine per-image
> epistemic signal — and simulate a triage policy that defers the most-uncertain cases
> to a radiologist. Figure~\ref{fig:risk_coverage} plots retained-case Timika MAE against
> the retained fraction. Deferring the most-uncertain **20\%** of cases lowers retained
> MAE from `<full_MAE>` to `<mae@80>` on `<country>` (a `<drop>`-point reduction), and the
> uncertainty policy beats random deferral at every operating point on
> `<k/3>` of the three held-out countries. Crucially, deferral also removes a
> disproportionate share of cavity false negatives — the clinically costly error carrying
> the 40-point Timika penalty (Eq.~\ref{eq:timika}) — reducing the retained cavity-FN rate
> from `<fn@100>` to `<fn@80>`. This confirms that the ensemble uncertainty is not merely
> a reported number but an actionable abstention signal.
>
> [If flat on a country:] On `<country>`, deferral by uncertainty tracks random deferral,
> indicating that its residual error is dominated by label shift rather than
> reducible epistemic uncertainty — consistent with the shift decomposition (Sec.~3).

## Figure caption

> \caption{\textbf{Selective prediction / abstention (Hybrid mode, R6 ensemble).}
> Retained-case Timika MAE as a function of the retained fraction, deferring the
> most-uncertain cases (ranked by deep-ensemble inter-member std) to a radiologist.
> Solid lines: defer by uncertainty; dashed: random deferral (matched deferral rate).
> Uncertainty-based deferral lowers retained MAE monotonically and beats random on
> `<k/3>` held-out countries, showing the ensemble uncertainty is actionable.}
> \label{fig:risk_coverage}

## How to fill

Run:
```
python3.11 scripts/selective_prediction.py \
    --preds-dir <dir with patched preds_*.csv> \
    --mode fusion --rung rung6_conformal --out-dir ws_sp
```
Then read the numbers off `ws_sp/selective_prediction.csv`:
- `<full_MAE>` = `mae_uncertainty_mean` at `retained_fraction = 1.0`
- `<mae@80>`  = `mae_uncertainty_mean` at `retained_fraction = 0.8`
- `<drop>`    = `<full_MAE>` − `<mae@80>`
- `<fn@100>` / `<fn@80>` = `cavity_fn_rate` at 1.0 / 0.8
- `<k/3>` = # countries where the console sanity check prints `beats-random=True`

Figure: `ws_sp/fig_risk_coverage.pdf` (copy into `iconips_Paper/figures/`).
