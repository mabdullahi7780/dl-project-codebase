from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.app.infer import run_single_image_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline TB CXR pipeline on a CSV manifest.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--component4-decoder-ckpt",
        default=None,
        help="Path to a fine-tuned Component 4 mask decoder checkpoint (overrides config).",
    )
    parser.add_argument(
        "--component1-adapter",
        default=None,
        help="Path to Component 1 LoRA+DANN adapters (overrides config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "dataset"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Batch CSV must contain at least `image` and `dataset` columns.")

        for index, row in enumerate(reader):
            image_path = row["image"]
            dataset = row["dataset"]
            sample_name = row.get("sample_id") or Path(image_path).stem or f"sample_{index:04d}"
            sample_outdir = outdir / sample_name
            bundle = run_single_image_inference(
                image_path=image_path,
                dataset=dataset,
                outdir=sample_outdir,
                config_path=args.config,
                view=row.get("view") or None,
                pixel_spacing_cm=float(row["pixel_spacing_cm"]) if row.get("pixel_spacing_cm") else None,
                seed=args.seed,
                component4_decoder_ckpt=args.component4_decoder_ckpt,
                component1_adapter_path=args.component1_adapter,
            )
            results.append(
                {
                    "sample_id": sample_name,
                    "image": image_path,
                    "dataset": dataset,
                    "timika_score": bundle.timika_score,
                    "severity": bundle.severity,
                    "report_path": str(sample_outdir / "report.txt"),
                    "json_path": str(sample_outdir / "evidence.json"),
                    "overlay_path": str(sample_outdir / "overlay.png"),
                }
            )

    (outdir / "batch_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"processed": len(results), "summary_path": str(outdir / "batch_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
