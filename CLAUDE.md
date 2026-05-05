# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

End-to-end TB chest X-ray pipeline targeting SOTA performance. The repo is **CXR-only** (TB Portals removed), domain classifier is **4-way** (montgomery=0, shenzhen=1, tbx11k=2, nih_cxr14=3), target hardware is a single 8 GB GPU in fp16. Python 3.11, PyTorch.

**Read `sprint_plan.md` before touching any training script.** It contains sprint history, confirmed metrics, and exact root-cause analyses for every known bug. `plan.md` is the architectural design brief. `README.md` tracks implemented components.

## Sprint Status

| Sprint | Target | Status | Key Metric |
|---|---|---|---|
| Sprint 1 (C4 lung) | Dice ≥ 0.85 | ✅ Complete | Montgomery 0.886, Shenzhen 0.959 |
| Sprint 2 (C2 TB head) | AUROC ≥ 0.85 | ✅ Complete | Overall 0.9842, TBX11K 0.9828 |
| Sprint 3 (MoE training) | Timika AUROC ≥ 0.70 | ❌ Not started | MoE ALP = 0 (gate/experts untrained) |

## Common Commands

```bash
# Install deps
pip install -r requirements.txt

# Run all tests
pytest

# End-to-end inference (baseline path)
python -m src.app.infer \
  --image path/to/file.png \
  --dataset montgomery \
  --outdir outputs/demo \
  --config configs/baseline.yaml

# End-to-end inference (MoE path — requires trained MoE checkpoint)
python -m src.app.infer \
  --image path/to/file.png \
  --dataset montgomery \
  --outdir outputs/demo \
  --config configs/moe.yaml

# Component 2 (TB head) training
python -m src.training.train_component2_txv --config configs/component2_txv.yaml

# Component 4 (lung) training
python -m src.training.train_component4_lung --config configs/component4_lung.yaml

# MoE Phase 0 — build embedding cache (PREREQUISITE for Phase 1/2/3)
python scripts/cache_moe_embeddings.py --config configs/moe.yaml --cache-dir /path/to/cache

# MoE Phase 1 — pretrain all 4 expert decoders
python -m src.training.train_experts --config configs/moe.yaml --cache-dir /path/to/cache --all

# MoE Phase 2 — joint MoE training (gate-only by default in configs/moe.yaml)
python -m src.training.train_moe_joint --config configs/moe.yaml --cache-dir /path/to/cache

# MoE Phase 3 — boundary critic
python -m src.training.train_boundary_critic --config configs/moe.yaml --cache-dir /path/to/cache

# Kaggle bootstrap (runs all 4 phases sequentially)
python scripts/kaggle_moe_train.py --mode full
```

Dataset roots and checkpoint paths are resolved from `configs/paths.yaml` + environment variables from `.env.example`. Configs use `${VAR:-default}` substitution — no hardcoded paths.

## Architecture

### Dual inference paths

The pipeline supports two lesion segmentation paths:

**Baseline path** (`configs/baseline.yaml`):
`C0 → C1 → C2 → C4 → baseline_lesion_proposer → C7 → C8 → C9 → C10`

**MoE path** (`configs/moe.yaml`):
`C0 → C1 → C2 → C4 → C3 (gate) → C5 (4 experts) → C6 (fusion) → C7 (upgraded) → C8 → C9 → C10`

### Component-by-component

1. **Component 0** (`component0_qc.py`) — QC + normalisation. Produces `HarmonisedCXR` (`x_1024`, `x_224`, `x_3ch`, meta).
2. **Component 1** (`component1_encoder.py` + `component1_dann.py`) — MedSAM ViT-B backbone with LoRA adapters + 4-way DANN domain head. Returns `(img_emb [B,256,64,64], dom_logits [B,4])`.
3. **Component 2** (`component2_txv.py`) — TorchXRayVision DenseNet121 (frozen) + trainable routing head + TB binary head. Returns `(domain_ctx [B,256], pathology_logits [B,18], tb_logit [B,1])`. **Requires `torchxrayvision` installed** — without it, the mock backend activates and TB AUROC is exactly 0.5000. The training script (`train_component2_txv.py`) guards against this with a `RuntimeError` on startup.
4. **Component 4** (`component4_lung.py`) — MedSAM ViT-B lung mask (frozen encoder, trainable decoder). Returns `lung_mask_256 [B,1,256,256]` + `lung_mask_1024 [B,1,1024,1024]`.
5. **`baseline_lesion_proposer.py`** — Baseline-only lesion source (Grad-CAM from TXV TB-mimic classes, masked by lung, morphology-cleaned). Used when MoE not available. **Will be superseded by the MoE path once Sprint 3 completes.**
6. **Component 3** (`component3_routing.py`) — Routing gate. Takes `img_emb [B,256,64,64]` + optional `domain_ctx [B,256]` → `expert_weights [B,4]`. **Code implemented; weights never trained.**
7. **Component 5** (`component5_experts.py`) — Four lightweight CNN expert decoders (consolidation, cavity, fibrosis, nodule). Each: `img_emb [B,256,64,64]` → `mask_logits [B,1,256,256]`. **Code implemented; weights never trained.**
8. **Component 6** (`component6_fusion.py`) — Logit-space weighted fusion. `expert_logits + routing_weights` → `FusionOutput` (fused mask + inter-expert variance). **Code implemented; weights never trained.**
9. **Component 7** (`component7_boundary.py`, `component7_fp_auditor.py`, `component7_refine.py`, `component7_verification.py`) — Boundary quality score, FP auditor, and mask refinement. Heuristic boundary scorer active in baseline; ResNet18 boundary critic exists in `train_boundary_critic.py` but not trained.
10. **Component 8** (`component8_timika.py`) — Deterministic ALP, cavity flag, Timika score, severity. **ALP = 0 for all images until Sprint 3 completes** (MoE masks are empty).
11. **Component 9** (`component9_json_output.py`) — Structured evidence JSON.
12. **Component 10** (`component10_report.py`, `component10_biogpt.py`) — Deterministic template report (BioGPT variant is low priority).

### MoE training order (Sprint 3)

The three MoE modules must train in this specific order. Violating it produces empty masks:

```
Phase 0: cache_moe_embeddings.py   ← PREREQUISITE, builds .pt cache on disk
Phase 1: train_experts.py          ← pretrain 4 expert decoders independently
Phase 2: train_moe_joint.py        ← train routing gate (experts frozen, gate_only=true)
Phase 3: train_boundary_critic.py  ← optional; trains ResNet18 boundary critic
```

### Data contracts

`src/core/types.py` defines:
- `HarmonisedCXR` — Component 0 output contract that every downstream component consumes.
- `BaselineInferenceBundle` — shared end-to-end state accumulated field-by-field through inference. Add new fields here; do not invent parallel return structures.

Field names are load-bearing — the plan, tests, and JSON schema assume them.

### Backends and graceful fallback

Components 1, 2, and 4 each accept `backend="auto"` and expose `active_backend`. They fall back to lightweight stubs when checkpoints or optional deps are missing. **The mock backend for Component 2 is dangerous:** `MockTXVDenseNet.features()` returns zeros, making TB AUROC exactly 0.5000. The training script raises `RuntimeError` if the mock backend is active. Never disable this guard.

### Training

Training scripts in `src/training/train_componentN_*.py` are standalone. Each has its own YAML config in `configs/`. Component1DomainDataset reads from raw dataset dirs or tar archives on external storage, keyed by canonical dataset IDs.

### Utilities

- `src/core/` — `types`, `constants`, `device` (CPU/CUDA/MPS picker), `seed`.
- `src/data/transforms_qc.py`, `src/data/harmonise.py` — low-level preprocessing (CLAHE, resize, grayscale→3ch).
- `src/utils/` — `checkpoints`, `morphology` (connected components), `visualization` (overlay/mask PNG writers).

Notebooks in `notebooks/` are for Kaggle training and manual evaluation only. Core logic must live in `src/`.

## Known Bugs (Fix Before Reporting Metrics)

1. **TBX11K AUROC = N/A in eval_baseline.ipynb** — eval manifest with `limit_per_domain=200` draws only TB-positive TBX11K images (directory walk hits `imgs/tb/` first). Fix: stratified sampling.

2. **MoE checkpoint missing → silent fallback** — `src/app/infer.py` may silently fall back to `baseline_lesion_proposer` when `moe_best.pt` is missing. Always log which lesion path is active.

3. **Montgomery TB AUROC high variance** — only 10–15 negative val images means one misranked pair = ±0.08 AUROC. Do not over-interpret small fluctuations in Montgomery AUROC.

## Rules (enforce when editing)

- Every component must have an isolated class/module with a clean input/output contract — no hidden global state.
- No hardcoded absolute paths; go through `configs/paths.yaml` + env vars.
- Inference steps must run under `torch.no_grad()` and support CPU fallback.
- Core logic lives in `src/`, not in notebooks.
- Meaningful errors on missing metadata (see `canonicalise_dataset_id` for the pattern).
- Never disable the `active_backend != "xrv"` guard in `train_component2_txv.py::main()`.
- MoE training phases must execute in order: Phase 0 → 1 → 2 → (3 optional). Never skip Phase 0 or 1.
