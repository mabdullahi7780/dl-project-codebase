---
name: Checkpoints Inventory
description: Exact filenames and sizes of pretrained artifacts confirmed present in the repo's checkpoints/ tree as of 2026-04-21
type: project
---

Confirmed present artifacts (as of 2026-04-21):

- checkpoints/medsam/medsam_vit_b.pth (~358 MB) — shared MedSAM ViT-B backbone weights, loaded by BOTH Component 1 (encoder) and Component 4 (lung decoder)
- checkpoints/component1/component1_adapters.safetensors — LoRA adapters + DANN head (trainable-only state dict)
- checkpoints/component1/component1_dann_full.pt — full state dict (includes frozen encoder)
- checkpoints/component1/last_component1_snapshot.pt (~348 MB) — Kaggle mid-run resume snapshot
- checkpoints/component2/component2_routing_head.pt (~1.3 MB) — trained soft-domain routing head for TXV DenseNet121
- checkpoints/component4/component4_mask_decoder.pt (~46 MB) — MedSAM mask-decoder weights fine-tuned for lung segmentation
- checkpoints/component_moe/ (~72 MB total):
    - expert_{cavity,consolidation,fibrosis,nodule}_{best,final}.pt  (~640 KB each)
    - moe_best.pt (~2.9 MB)  — routing_gate + expert_bank + fusion joint checkpoint
    - moe_checkpoint.pt (~2.9 MB) — final epoch
    - moe_epoch{1,3,5,7,9,11,13}.pt — periodic snapshots
    - boundary_critic.pt (~44 MB) — ResNet18 boundary critic (Component 7 verification)
    - moe_joint_history.jsonl, expert_*_history.jsonl — training curves
- checkpoints/txv/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt (~28 MB) — TorchXRayVision DenseNet121 public weights

**Why:** These files drive all inference paths; knowing what's present saves re-training and tells us which configs can be exercised end-to-end.

**How to apply:** When asked about what the pipeline can run today, trust this inventory first, then re-verify a specific file's presence with Glob if the user is about to act on the info.
