---
name: Component Implementation Status Snapshot
description: Confirmed implementation status per Component as of 2026-04-21
type: project
---

- Component 0 (QC + harmonisation): Fully implemented. canonicalise_dataset_id + validate_view + validate_image_shape + dataset-aware CLAHE + HarmonisedCXR contract. Tested.
- Component 1 (MedSAM ViT-B + LoRA + DANN): Fully implemented and trained. LoRA rank=4/alpha=16 on qkv, 4-way DANN head with GRL; adapters .safetensors + snapshot.pt present.
- Component 2 (TXV soft domain ctx + pathology + 7×7 features): Fully implemented and trained. Routing head .pt present. 18-class TXV logits exposed to proposer + auditor.
- Component 3 (routing gate): Implemented AND trained (plan.md Priority C). GAP→MLP→softmax with temperature + top_k + optional domain_ctx; moe_best.pt includes its state.
- Component 4 (MedSAM lung): Fully implemented and trained. Decoder checkpoint present; returns LungMaskOutput at 256 and 1024.
- Component 5 (ExpertBank, 4 experts): Implemented and trained. EXPERT_NAMES=("consolidation","cavity","fibrosis","nodule"); per-expert *_best.pt files.
- Component 6 (fusion): Implemented. Weighted-logit default, weighted-prob alternative; emits fused mask + variance + per-expert masks.
- Baseline lesion proposer: Implemented (TXV Grad-CAM surrogate + lung intersect + Otsu + morphology). Intentional stand-in — do not remove.
- Component 7 (boundary + FP + refine + verification + reprompt refiner): Fully implemented. Boundary-critic ResNet18 trained (boundary_critic.pt present). Expert-3 reprompt refiner with Dice arbiter wired for MoE mode.
- Component 8 (Timika): Fully implemented. Deterministic baseline compute_baseline_timika (cavity=0, confidence not-assessed-baseline); compute_moe_timika uses Expert-2 cavity mask when present (confidence expert2-radiographic).
- Component 9 (structured JSON): Fully implemented with routing-weights extension for MoE.
- Component 10 (report): Template path fully implemented. BioGPT path implemented with FaithfulnessChecker and fall-back; `use_mock: false` by default in moe.yaml, template backend selected by default.

**Why:** Gives a one-line truth table for future audit runs so I don't re-read 50 files to confirm status.

**How to apply:** Before any audit/status answer, cross-check this snapshot against the checkpoints_inventory.md file — if a trained checkpoint is missing, downgrade the corresponding status from "trained" to "implemented but untrained".
