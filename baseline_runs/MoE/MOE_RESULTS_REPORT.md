# DA-MoE Results — Brutally Honest Report

Source: `baseline_runs/MoE/{MoE_A1,MoE_A2,MoE_A3,MoE_Fusion}` (3 countries × 3 seeds × 30 ep)
plus the ablation traces recovered from the executed notebooks' logs.
Comparison anchors: our **locked baselines** (`baseline_runs/BASELINE_COMPARISON.md`) and Kantipudi.

---

## 1. Verdict (read this first)

**The MoE, as configured, failed.** It is *worse* than the locked single-network baselines on
every mode except a marginal gain on A1 (the paper's weakest arm). The flagship `fusion` regressed,
`a2` regressed, and `a3` **collapsed to near-zero correlation** (Pearson ≈ 0). This is a negative
result — but a *diagnosable* one, and the ablations point to a single dominant cause.

## 2. The scoreboard — what we tried to beat vs what we got

Timika MAE (lower better) / Pearson (higher better), 3-seed mean. "OUR base" = locked baseline = the bar.

| Mode | Country | **MoE (got)** | OUR baseline (target) | Paper | Verdict |
|------|---------|--------------:|----------------------:|------:|---------|
| **a1** | Romania | 23.93 / 0.563 | 26.84 / 0.570 | 23.83 / 0.59 | beats base (MAE) |
| | Moldova | 29.94 / 0.668 | 32.76 / 0.643 | 24.44 / 0.80 | beats base (MAE) |
| | Kazakhstan | 22.46 / 0.658 | 21.87 / 0.651 | 22.13 / 0.68 | worse |
| **a2** | Romania | 24.92 / 0.591 | **20.11 / 0.684** | 18.70 / 0.70 | **worse** (+4.8) |
| | Moldova | 33.34 / 0.458 | **30.68 / 0.697** | 18.85 / 0.84 | **worse** (Pearson −0.24) |
| | Kazakhstan | 28.72 / 0.588 | **21.35 / 0.648** | 19.62 / 0.70 | **worse** (+7.4) |
| **a3** | Romania | 29.58 / **0.168** | **20.26 / 0.699** | 19.67 / 0.70 | **collapsed** |
| | Moldova | 36.46 / **0.024** | **26.16 / 0.759** | 18.98 / 0.85 | **collapsed** |
| | Kazakhstan | 32.62 / **0.184** | **21.90 / 0.692** | 22.12 / 0.74 | **collapsed** |
| **fusion** | Romania | 25.71 / 0.542 | 20.11 / 0.684 | — | **worse** |
| | Moldova | 33.39 / 0.434 | 30.68 / 0.697 | — | **worse** |
| | Kazakhstan | 29.27 / 0.568 | 21.35 / 0.648 | — | **worse** |

Overall 3-country mean Timika MAE: a1 25.44 (base 27.16, **−1.7**) · a2 28.99 (base 24.05, **+4.9**)
· a3 32.89 (base 22.77, **+10.1**) · fusion 29.46 (base 24.05, **+5.4**).

**a3 Pearson ≈ 0 means the model is predicting essentially noise / a near-constant** — the single
biggest red flag. The A3 *baseline* (one direct regressor) got Pearson ~0.70; adding our MoE
machinery destroyed it.

## 3. Root cause — the ablations make it unambiguous

The per-epoch logs and the `--no-dann` ablation (single seed) isolate the culprit. Two facts:

**(a) Validation error rises monotonically as the DANN strength λ ramps up.** A3, every run:

| epoch | λ (GRL) | val Timika-MSE |
|------:|--------:|---------------:|
| 0  | 0.00 | **0.042** (best) |
| 9  | 0.91 | 0.071–0.123 |
| 29 | 1.00 | 0.063 |

The model is at its best at λ=0 (epoch 0) and degrades as the adversary kicks in.

**(b) Turning DANN OFF recovers the signal completely.** From the ablation logs (seed 0):

| Mode / country | full (λ→1) | **`--no-dann` (λ=0)** | locked baseline |
|---|---:|---:|---:|
| A3 Romania | 29.30 / **0.19** | 23.39 / **0.72** | 20.26 / 0.70 |
| A3 Moldova | 36.38 / **0.05** | 30.81 / **0.74** | 26.16 / 0.76 |
| A3 Kazakhstan | 32.75 / **0.17** | 20.83 / **0.70** | 21.90 / 0.69 |
| A2 Romania | 25.76 / **0.54** | 20.79 / **0.70** | 20.11 / 0.68 |
| A2 Moldova | 33.33 / **0.44** | **27.27 / 0.74** | 30.68 / 0.70 |

The `--no-critic` ablation (DANN still ON) stays collapsed (A3 Pearson 0.05/0.08/−0.03); the
`--no-both` ablation (DANN OFF) recovers (A3 Pearson 0.65/0.75). So **the critic is not the
problem — DANN at λ_max = 1.0 is.**

**Conclusion: the gradient-reversal domain adversary at λ=1.0 is collapsing the shared trunk's
regression-relevant features (a classic over-strong-adversary failure).** Supporting evidence:
`a1` — whose prediction comes from *fixed* detection geometry + the frozen cavity agent and barely
uses the trunk — is the *only* mode that did **not** collapse under DANN. The trunk-dependent
modes (a2/a3/fusion) all did; a3 (purely trunk→Timika) collapsed hardest.

## 4. Compounding design errors (smaller, but real)

1. **The λ schedule ramps over the *whole* run, so it is already λ≈0.91 by the end of "phase 1."**
   The experts never get a clean pretraining — they fight the adversary from epoch 1. The two-phase
   design is effectively defeated.
2. **Model selection only saves checkpoints from phase 2** (`if phase2 and vmse < best_val`). The
   lowest validation error occurs at epoch 0–1 (phase 1, λ≈0), but those are ignored. We select the
   best of the *most-corrupted* regime.
3. **MSE-based selection can reward a "predict-the-mean" collapse** (low variance ⇒ low-ish MSE ⇒
   ~0 Pearson). Consistent with a3's high MAE + zero Pearson.
4. **The mixture + critic, even with DANN off, give ~no gain over the single baseline** — they roughly
   tie it (A3 −no-dann ≈ baseline; A2 −no-dann ≈ baseline). The extra machinery is not, by itself,
   buying accuracy.

## 5. The one glimmer worth noting

`A2 --no-dann` on **Moldova** (the target): **27.27 / 0.74 vs baseline 30.68 / 0.697** — a real
improvement on the country that matters, from the mixture + critic *without* the broken adversary.
It's a single seed, so treat it as a hypothesis, not a result — but it's the one signal that the
core idea isn't dead.

## 6. How to fix — options to brainstorm (NOT yet implemented)

Ordered roughly by expected impact / effort.

**A. Fix DANN tuning (highest priority — it's the headline mechanism).**
- Drop `λ_max` by 1–2 orders of magnitude (try 0.05–0.1, not 1.0). DANN on small medical datasets
  typically needs small λ.
- Confine the λ ramp to **phase 2 only**, after experts have converged at λ=0 in phase 1.
- Add a **warm-up** (λ=0 for the first N epochs regardless of phase).
- Consider a **separate, lower LR** for the trunk vs the domain head, or detaching DANN to a
  shallower branch so it can't dominate the whole trunk.

**B. Fix model selection (cheap, do regardless).**
- Track best val across **all** epochs (including phase 1), not just phase 2.
- Select by **validation Pearson** (or MAE), not MSE, to avoid rewarding mean-collapse.

**C. Re-examine whether the mixture is doing anything (cheap, diagnostic).**
- Log the **gate weights** per country at test time. If the gate is near-uniform, the "experts"
  aren't specialising and the MoE reduces to an ensemble average — we'd need a diversity mechanism
  or per-expert specialisation signal.

**D. Reconsider the DANN target (conceptual).**
- DANN enforces *country invariance*, but Moldova fails because it's *out-of-distribution in label
  space* (sicker), not just a style shift. Invariance to country may be removing exactly the
  severity signal we need. Alternatives: importance weighting / distribution alignment on the
  *label* (e.g. CORAL on features conditioned on severity), or quantile/balanced sampling so high-ALP
  cases aren't drowned out — these attack the actual shift more directly than adversarial invariance.

**E. Decide the honest contribution if DANN can't be made to help.**
- If, after tuning, DANN gives no significant lift, the paper pivots to "a calibrated mixture +
  critic for severity regression" (anchored to the A2-Moldova glimmer) — a smaller but honest claim.

## 7. Open questions for you

1. **Headline mechanism:** is DANN (cross-country invariance) the contribution we're committed to,
   or are we open to a different domain-shift remedy (importance weighting / balanced sampling /
   CORAL) if DANN keeps fighting the regression?
2. **Compute budget:** how many more Kaggle GPU-hours can we spend on the λ sweep + re-runs? (One
   mode × 3 seeds ≈ 2–3 h; a proper λ sweep is several of those.)
3. **Target:** are we trying to beat the **paper** (likely infeasible — they report Moldova 18.85,
   we can't even replicate that) or our **locked baseline** (the honest, defensible target)?
4. **Scope:** focus all effort on getting **one** mode right first (I'd argue **A2**, given the
   Moldova glimmer), or keep all four in play?
