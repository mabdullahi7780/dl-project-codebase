#!/usr/bin/env python3
"""
Reproduce the six-country extra LOCO summary tables from committed result CSVs.

This script does not require:
- image files
- RAD-DINO feature caches
- private per-image prediction files
- model checkpoints

It only reads the aggregate CSVs in:
baseline_runs/agentic_runs/extra_loco/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COUNTRIES = {
    "Romania",
    "Moldova",
    "Kazakhstan",
    "Georgia",
    "Ukraine",
    "Belarus",
}

SELECTED_RUNGS = [
    "rung1_mse",
    "rung1_bmc",
    "rung4b_spatial_cavity",
    "agentic_best_spatcav",
    "agentic_v2_iso",
]


def require_columns(df: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def build_country_summary(six: pd.DataFrame) -> pd.DataFrame:
    return (
        six.groupby(["held_out", "rung"], as_index=False)
        .agg(
            n_test=("n_test", "max"),
            seed_count=("seed", "nunique"),
            mae_mean=("timika_mae", "mean"),
            mae_std=("timika_mae", "std"),
            pearson_mean=("timika_pearson", "mean"),
            pearson_std=("timika_pearson", "std"),
        )
        .sort_values(["held_out", "rung"])
        .reset_index(drop=True)
    )


def build_macro_summary(country_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for rung, group in country_summary.groupby("rung", sort=True):
        weights = group["n_test"].to_numpy(dtype=float)

        rows.append(
            {
                "rung": rung,
                "n_countries": group["held_out"].nunique(),
                "n_images_total": int(group["n_test"].sum()),
                "mae_country_macro": group["mae_mean"].mean(),
                "mae_image_weighted": np.average(
                    group["mae_mean"],
                    weights=weights,
                ),
                "pearson_country_macro": group["pearson_mean"].mean(),
                "pearson_image_weighted": np.average(
                    group["pearson_mean"],
                    weights=weights,
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["mae_country_macro", "rung"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="baseline_runs/agentic_runs/extra_loco",
        help="Folder containing the extra LOCO aggregate CSV files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require the expected 450 rows and 75 rows per country.",
    )

    args = parser.parse_args()
    root = Path(args.root)

    six_path = root / "results_fusion_six_country.csv"
    bootstrap_path = root / "paired_cluster_bootstrap_extra_loco.csv"

    if not six_path.exists():
        raise FileNotFoundError(f"Could not find {six_path}")

    six = pd.read_csv(six_path)

    require_columns(
        six,
        {
            "held_out",
            "rung",
            "seed",
            "n_test",
            "timika_mae",
            "timika_pearson",
        },
        six_path,
    )

    countries = set(six["held_out"].unique())

    if countries != EXPECTED_COUNTRIES:
        raise ValueError(
            "Unexpected country set.\n"
            f"Expected: {sorted(EXPECTED_COUNTRIES)}\n"
            f"Found:    {sorted(countries)}"
        )

    if args.strict:
        if len(six) != 450:
            raise ValueError(f"Expected 450 rows, found {len(six)}")

        country_rows = six.groupby("held_out").size()
        bad = country_rows[country_rows != 75]

        if len(bad):
            raise ValueError(
                "Expected 75 rows per country. Bad counts:\n"
                f"{bad.to_string()}"
            )

    country_summary = build_country_summary(six)
    macro_summary = build_macro_summary(country_summary)

    country_summary_path = root / "summary_fusion_six_country_by_country.csv"
    macro_summary_path = root / "summary_fusion_six_country_macro.csv"
    selected_macro_path = root / "paper_ready_selected_macro.csv"
    selected_country_path = root / "paper_ready_selected_by_country.csv"

    country_summary.to_csv(country_summary_path, index=False, float_format="%.6f")
    macro_summary.to_csv(macro_summary_path, index=False, float_format="%.6f")

    selected_macro = macro_summary[
        macro_summary["rung"].isin(SELECTED_RUNGS)
    ].copy()

    selected_country = country_summary[
        country_summary["rung"].isin(SELECTED_RUNGS)
    ].copy()

    selected_macro.to_csv(selected_macro_path, index=False, float_format="%.6f")
    selected_country.to_csv(selected_country_path, index=False, float_format="%.6f")

    print("\nSix-country selected macro summary:")
    print(
        selected_macro[
            [
                "rung",
                "mae_country_macro",
                "mae_image_weighted",
                "pearson_country_macro",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    if bootstrap_path.exists():
        bootstrap = pd.read_csv(bootstrap_path)
        bootstrap_out = root / "paper_ready_bootstrap_extra_loco.csv"
        bootstrap.to_csv(bootstrap_out, index=False, float_format="%.6f")

        print("\nExtra LOCO bootstrap significance:")
        print(bootstrap.round(4).to_string(index=False))

    print("\nWrote:")
    for path in [
        country_summary_path,
        macro_summary_path,
        selected_macro_path,
        selected_country_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
