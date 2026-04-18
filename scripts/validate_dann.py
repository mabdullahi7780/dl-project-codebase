"""Validate the retrained Component 1 DANN head against per-domain collapse.

Loads the trained LoRA + DANN adapter, runs a balanced N-per-domain forward
pass in eval mode (gradient-reversal lambda forced to 0), then prints a 4-way
confusion matrix, per-domain accuracy, and a collapse verdict.

Typical Kaggle usage::

    !python /kaggle/working/repo/scripts/validate_dann.py \
        --config /kaggle/working/component1_runs/component1_dann.kaggle.yaml \
        --paths  /kaggle/working/component1_runs/paths.kaggle.yaml \
        --adapter /kaggle/working/component1_runs/component1_adapters.safetensors \
        --limit-per-domain 20

The verdict is derived from the most-predicted-class fraction:
    > 0.80 -> COLLAPSED      (one domain dominates > 80% of predictions)
    > 0.50 -> PARTIAL         (skewed toward one domain)
    else   -> HEALTHY         (predictions spread across all 4 domains)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from src.components.component1_encoder import load_trainable_state_dict
from src.core.device import pick_device
from src.training.train_component1_dann import (
    DOMAIN_TO_ID,
    Component1DomainDataset,
    build_component1_manifest,
    build_model,
    collate_component1_batch,
    load_yaml_config,
    maybe_limit_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Component 1 DANN against domain collapse.")
    p.add_argument("--config", required=True, help="Path to component1_dann*.yaml used for retraining.")
    p.add_argument("--paths", required=True, help="Path to paths.yaml (dataset roots).")
    p.add_argument("--adapter", required=True, help="Path to component1_adapters.safetensors (or .pt).")
    p.add_argument("--limit-per-domain", type=int, default=20, help="Images per domain in the balanced eval.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None, help="cpu/cuda/mps; autodetect if omitted.")
    p.add_argument("--output", default=None, help="Optional path to write the result as JSON.")
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:
    cfg = load_yaml_config(args.config)["component1_dann"]
    samples = build_component1_manifest(paths_config=args.paths, component1_config=args.config)
    samples = maybe_limit_manifest(samples, int(args.limit_per_domain))
    if not samples:
        raise RuntimeError("Manifest is empty. Check paths.yaml and limit-per-domain.")

    device = pick_device(args.device or cfg["training"].get("device"))
    model = build_model(cfg).to(device)
    load_trainable_state_dict(model, args.adapter)
    model.eval()

    loader = DataLoader(
        Component1DomainDataset(samples, apply_augmentation=False),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_component1_batch,
    )

    id_to_domain = {v: k for k, v in DOMAIN_TO_ID.items()}
    n_domains = len(DOMAIN_TO_ID)
    confusion = torch.zeros(n_domains, n_domains, dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            x = batch["x_3ch"].to(device)
            y = batch["domain_id"]
            _, dom_logits = model(x, lambda_=0.0)
            y_hat = dom_logits.argmax(dim=1).detach().cpu()
            for t, p in zip(y.tolist(), y_hat.tolist()):
                confusion[t, p] += 1

    per_class_acc = {}
    for i in range(n_domains):
        row = confusion[i]
        denom = max(row.sum().item(), 1)
        per_class_acc[id_to_domain[i]] = row[i].item() / denom

    col_sums = confusion.sum(dim=0).tolist()
    total = confusion.sum().item()
    max_pred_frac = max(col_sums) / max(total, 1)
    dominant_domain = id_to_domain[int(torch.tensor(col_sums).argmax().item())]

    if max_pred_frac > 0.80:
        verdict = "COLLAPSED"
    elif max_pred_frac > 0.50:
        verdict = "PARTIAL"
    else:
        verdict = "HEALTHY"

    # Pretty-print
    label = "true\\pred"
    header = f"{label:<13}" + "".join(f"{id_to_domain[i]:>12}" for i in range(n_domains)) + "   per-class"
    print(header)
    for i in range(n_domains):
        row = confusion[i]
        acc = per_class_acc[id_to_domain[i]]
        print(
            f"{id_to_domain[i]:<13}"
            + "".join(f"{int(v.item()):>12}" for v in row)
            + f"   {acc:.1%}"
        )
    print()
    print(f"Total evaluated     : {total}")
    print(f"Per-domain counts   : {dict((id_to_domain[i], int(confusion[i].sum().item())) for i in range(n_domains))}")
    print(f"Mean per-class acc  : {sum(per_class_acc.values()) / n_domains:.1%}")
    print(f"Max pred fraction   : {max_pred_frac:.1%}  (dominant = {dominant_domain})")
    print(f"Verdict             : {verdict}")

    result = {
        "confusion": confusion.tolist(),
        "per_class_accuracy": per_class_acc,
        "mean_per_class_accuracy": sum(per_class_acc.values()) / n_domains,
        "max_pred_fraction": max_pred_frac,
        "dominant_domain": dominant_domain,
        "verdict": verdict,
        "total": int(total),
        "limit_per_domain": int(args.limit_per_domain),
        "adapter": str(args.adapter),
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Wrote JSON result to: {out_path}")
    return result


if __name__ == "__main__":
    run(parse_args())
