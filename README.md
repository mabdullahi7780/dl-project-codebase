# TB CXR Pipeline

This repo is no longer just `Component 0 + Component 1`.

Current state:

- Baseline path is implemented end to end: `C0 -> C1 -> C2 -> C4 -> baseline lesion proposer -> C7 heuristics -> C8 -> C9 -> C10`
- MoE path is implemented in code: `C3 routing -> C5 experts -> C6 fusion -> upgraded C7 -> upgraded C8`
- Component 1 LoRA+DANN and Component 4 lung decoder already have real training scripts and checkpoint hooks
- The MoE stack now supports a grounded 3-dataset cache workflow for `TBX11K + Montgomery + Shenzhen`

## What Is Implemented

- `src/components/component0_qc.py`
  Harmonisation, CLAHE policy, `x_1024`, `x_224`, `x_3ch`
- `src/components/component1_encoder.py`, `src/components/component1_dann.py`
  MedSAM encoder wrapper, LoRA, DANN
- `src/components/component2_txv.py`
  TXV pathology features plus optional trainable routing head
- `src/components/component4_lung.py`
  Lung segmentation with fine-tunable decoder
- `src/components/component3_routing.py`
  MoE routing gate, now with optional `domain_ctx` conditioning
- `src/components/component5_experts.py`
  Four pathology experts: `consolidation`, `cavity`, `fibrosis`, `nodule`
- `src/components/component6_fusion.py`
  Weighted expert fusion and variance map
- `src/components/component7_verification.py`
  Boundary critic and reprompt refiner
- `src/app/infer.py`
  Dual baseline/MoE inference entry point

## MoE Training — `gate_only` Mode

The default `configs/moe.yaml` ships with `moe_training.joint.gate_only: true`.

**What this means:**
When `gate_only: true`, the joint-training phase loads `moe_best.pt` and
**freezes the expert bank and fusion module**.  Only the routing gate's
parameters are updated.  This is the default because:

- The expert decoders are trained in Phase 1 (expert pretraining) and are
  considered production-quality.  Re-exposing them to gradient updates risks
  catastrophic forgetting of the fine-grained pathology boundaries each expert
  has learned.
- The fusion module is a small learned bias layer; retraining it jointly with
  a freshly initialised gate causes instability.
- Retraining only the gate converges in roughly 5 epochs instead of 15,
  making it practical on a single T4.

**How to run a full joint retrain (gate + experts + fusion):**
If you want to retrain all components simultaneously — for example after a
major data expansion — set `gate_only: false` in `configs/moe.yaml`:

```yaml
moe_training:
  joint:
    gate_only: false   # unlocks experts + fusion for gradient updates
```

Be prepared to increase `epochs` to at least 15 and monitor for
mode collapse (all routing weight converging to one expert).

## MoE Training Order

1. Train `Component 1` adapters if needed.
2. Train `Component 4` lung decoder if needed.
3. Build the MoE cache with `scripts/cache_moe_embeddings.py`.
4. Pretrain experts with `python -m src.training.train_experts --config configs/moe.yaml --cache-dir <cache> --all`.
5. Joint-train the MoE with `python -m src.training.train_moe_joint --config configs/moe.yaml --cache-dir <cache>`.
6. Train the boundary critic with `python -m src.training.train_boundary_critic --config configs/moe.yaml --cache-dir <cache>`.

## Kaggle

Use `scripts/kaggle_moe_train.py` for the Kaggle bootstrap. It expects the same dataset mount style already used for Component 1 training and runs:

- cache
- expert pretraining
- joint MoE training
- boundary critic training

The cache path intentionally excludes `NIH` for now and focuses on `TBX11K`, `Montgomery`, and `Shenzhen`.
