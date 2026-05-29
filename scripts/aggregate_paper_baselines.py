"""Aggregate published + shift-robust baseline outputs into the paper tables.

Reads:
  baseline_runs/agentic_runs/paper/_extract/preds_published_cavity_baselines.csv
  baseline_runs/agentic_runs/paper/_extract/preds_shift_robust_baselines.csv
  baseline_runs/agentic_runs/paper/_extract/summary_published_baselines.csv
  baseline_runs/agentic_runs/paper/_extract/summary_shift_robust.csv

Emits:
  baseline_runs/agentic_runs/paper/_extract/aggregated_baselines.csv
  baseline_runs/agentic_runs/paper/_extract/aggregated_baselines.md
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "baseline_runs" / "agentic_runs" / "paper" / "_extract"

OUR_LOCKED = {  # Our locally-trained K24 replication (DenseNet, 5 seeds)
    "Romania":    {"A1": 26.74, "A2": 20.11, "A3": 19.96},
    "Moldova":    {"A1": 32.86, "A2": 30.68, "A3": 28.18},
    "Kazakhstan": {"A1": 27.69, "A2": 21.35, "A3": 19.46},
}
OUR_AGENTIC = {  # Fusion-best + spatial-cavity, our best agentic
    "Romania": 19.12, "Moldova": 21.59, "Kazakhstan": 16.74,
}


def aggregate():
    pub_sum = pd.read_csv(ROOT / "summary_published_baselines.csv")
    sr_sum  = pd.read_csv(ROOT / "summary_shift_robust.csv")

    # Published cavity baselines are deterministic (single value per country/method); no seeds.
    pub_table = pub_sum.pivot_table(index="method", columns="held_out",
                                    values="timika_mae", aggfunc="mean").round(2)

    # Shift-robust: 5 seeds; report mean +/- std.
    sr_mean = sr_sum.groupby(["method", "held_out"])["timika_mae"].mean().unstack().round(2)
    sr_std  = sr_sum.groupby(["method", "held_out"])["timika_mae"].std().unstack().round(2)

    print("=== Published Cavity Baselines (deterministic) ===")
    print(pub_table.to_string())
    print()
    print("=== Shift-Robust Baselines (5 seeds, mean +/- std) ===")
    for m in sr_mean.index:
        print(f"\n{m}")
        for c in ["Romania", "Moldova", "Kazakhstan"]:
            mu, sd = sr_mean.loc[m, c], sr_std.loc[m, c]
            print(f"  {c:12s}  {mu:6.2f} +/- {sd:.2f}")

    # Write a combined CSV
    rows = []
    for m in pub_table.index:
        for c in ["Romania", "Moldova", "Kazakhstan"]:
            rows.append({"method": m, "country": c, "timika_mae_mean": pub_table.loc[m, c],
                         "timika_mae_std": np.nan, "n_seeds": 1})
    for m in sr_mean.index:
        for c in ["Romania", "Moldova", "Kazakhstan"]:
            rows.append({"method": m, "country": c,
                         "timika_mae_mean": sr_mean.loc[m, c],
                         "timika_mae_std":  sr_std.loc[m, c],
                         "n_seeds": 5})
    # Add our locked baseline + agentic for context
    for c in ["Romania", "Moldova", "Kazakhstan"]:
        for k, name in [("A1", "Ours: A1 (Single-Head ALP)"),
                        ("A2", "Ours: A2 (Dual-Head)"),
                        ("A3", "Ours: A3 (Dual-Task)")]:
            rows.append({"method": name, "country": c,
                         "timika_mae_mean": OUR_LOCKED[c][k],
                         "timika_mae_std": np.nan, "n_seeds": 5})
        rows.append({"method": "Ours: Agentic Fusion+SpatCav", "country": c,
                     "timika_mae_mean": OUR_AGENTIC[c],
                     "timika_mae_std": np.nan, "n_seeds": 5})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "aggregated_baselines.csv", index=False)
    print(f"\nWrote {ROOT / 'aggregated_baselines.csv'}")

    # Pivot to a paper-friendly table
    pivot = df.pivot_table(index="method", columns="country",
                           values="timika_mae_mean", aggfunc="mean")
    pivot = pivot[["Romania", "Moldova", "Kazakhstan"]]
    pivot.to_csv(ROOT / "aggregated_paper_table.csv")
    print(f"Wrote {ROOT / 'aggregated_paper_table.csv'}")

    md = ["# Aggregated Baselines for ICONIP 2026\n",
          "All numbers are Timika MAE (lower is better). Held-out country = test set.\n",
          "| Method | Romania | Moldova | Kazakhstan |",
          "|---|---:|---:|---:|"]
    method_order = [
        "TXV-only (no cavity)",
        "CheXzero-only (cavity x40)",
        "TXV+CheXzero",
        "GroupDRO (TXV)",
        "Importance-Weighted (TXV)",
        "Ours: A1 (Single-Head ALP)",
        "Ours: A2 (Dual-Head)",
        "Ours: A3 (Dual-Task)",
        "Ours: Agentic Fusion+SpatCav",
    ]
    for m in method_order:
        if m not in pivot.index:
            continue
        row_vals = []
        for c in ["Romania", "Moldova", "Kazakhstan"]:
            v = pivot.loc[m, c]
            if pd.isna(v):
                row_vals.append("--")
            else:
                row_vals.append(f"{v:.2f}")
        md.append(f"| {m} | {row_vals[0]} | {row_vals[1]} | {row_vals[2]} |")
    md_text = "\n".join(md) + "\n"
    (ROOT / "aggregated_baselines.md").write_text(md_text, encoding="utf-8")
    print(f"Wrote {ROOT / 'aggregated_baselines.md'}")
    print()
    print(md_text)


if __name__ == "__main__":
    aggregate()
