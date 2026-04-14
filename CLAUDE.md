# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Baseline-first implementation of the TB chest X-ray pipeline described in `plan.md`. The repo is **CXR-only** (TB Portals removed), the domain classifier is **4-way** (montgomery, shenzhen, tbx11k, nih_cxr14), and the target hardware is a single 8 GB GPU in fp16. Python 3.11, PyTorch.

`plan.md` is the authoritative design brief — read it before making architectural changes. `README.md` tracks which components are currently implemented vs. scaffolded.

## Common commands

```bash
# Install deps
pip install -r requirements.txt

# Run all tests (pytest config in pyproject.toml: testpaths=tests, pythonpath=.)
pytest

# Run one test file / one test
pytest tests/test_component1.py
pytest tests/test_component1.py::test_name

# End-to-end baseline inference on a single image
python -m src.app.infer \
  --image path/to/file.png \
  --dataset montgomery \
  --outdir outputs/demo \
  --config configs/baseline.yaml

# Component 1 (SAM+LoRA+DANN) training
python -m src.training.train_component1_dann --config configs/component1_dann.yaml

# Other training entrypoints live in src/training/ (component2, component4, component7...)
```

Dataset roots and checkpoint paths are resolved from `configs/paths.yaml` + environment variables from `.env.example` (e.g. `EXTERNAL_DATA_ROOT`, `MONTGOMERY_ROOT`, `SAM_VIT_H_CKPT`). Configs use `${VAR:-default}` substitution — do not hardcode absolute paths.

## Architecture

### Component pipeline (baseline)

Inference flows through numbered Components, each isolated in `src/components/componentN_*.py`. The end-to-end path in `src/app/infer.py::run_single_image_inference` is the canonical wiring:

1. **Component 0** (`component0_qc.py`) — QC + normalisation. Produces `HarmonisedCXR` (`x_1024`, `x_224`, `x_3ch`, meta).
2. **Component 1** (`component1_encoder.py` + `component1_dann.py`) — SAM ViT-H backbone with LoRA adapters + DANN domain head. Returns `(img_emb, dom_logits)`.
3. **Component 2** (`component2_txv.py`) — TorchXRayVision DenseNet121 soft domain context + pathology logits + 7×7 features.
4. **Component 4** (`component4_lung.py`) — MedSAM ViT-B lung mask (256 and 1024 resolutions).
5. **`baseline_lesion_proposer.py`** — baseline-only stand-in for the missing expert branch (Components 3/5/6). Builds a Grad-CAM suspiciousness map from TXV TB-mimic classes, masks by lung, thresholds + cleans morphology → coarse lesion mask. Scheduled to be replaced by the real expert path later; keep its interface stable.
6. **Component 7** (`component7_boundary.py`, `component7_fp_auditor.py`, `component7_refine.py`) — boundary-quality score, FP-probability audit, and refinement that uses both to shrink / suppress regions.
7. **Component 8** (`component8_timika.py`) — deterministic ALP, cavity flag, Timika score, severity.
8. **Component 9** (`component9_json_output.py`) — structured evidence JSON.
9. **Component 10** (`component10_report.py`, `component10_biogpt.py`) — deterministic template report (BioGPT variant is Priority C).

Components 3, 5, 6 are intentionally **not** implemented in the baseline. Do not add stubs for them unless explicitly asked.

### Data contracts

`src/core/types.py` defines the two dataclasses that tie the pipeline together:

- `HarmonisedCXR` — the Component 0 output contract that every downstream component consumes.
- `BaselineInferenceBundle` — the shared end-to-end state, accumulated field-by-field through inference. New components should add fields here rather than inventing parallel return structures.

Field names in these dataclasses are load-bearing — the plan, tests, and JSON schema assume them.

### Backends and graceful fallback

Components 1, 2, and 4 each accept `backend="auto"` and expose an `active_backend` attribute. They attempt the real pretrained model (SAM, TorchXRayVision, MedSAM) and fall back to a lightweight stub when checkpoints or optional deps are missing, so tests and CPU smoke runs work without weights. When modifying these components, preserve the fallback path and the `active_backend` reporting.

### Training

Training scripts live in `src/training/train_componentN_*.py` and are standalone (`python -m src.training.<name> --config ...`). Each has its own YAML config in `configs/`. `Component1DomainDataset` is set up to read from raw dataset dirs **or** tar archives on an external drive, keyed by canonical dataset IDs in `DOMAIN_TO_ID` (montgomery=0, shenzhen=1, tbx11k=2, nih_cxr14=3).

### Utilities

- `src/core/` — `types`, `constants`, `device` (CPU/CUDA/MPS picker), `seed`.
- `src/data/transforms_qc.py`, `src/data/harmonise.py` — low-level preprocessing (CLAHE, resize, grayscale→3ch).
- `src/utils/` — `checkpoints`, `morphology` (connected components), `visualization` (overlay/mask PNG writers).

Notebooks in `notebooks/` are for manual inspection against datasets on an external HDD; they must not contain core logic.

## Rules from plan.md §7 (enforce when editing)

- Every component must have an isolated class/module with a clean input/output contract — no hidden global state.
- No hardcoded absolute paths; go through `configs/paths.yaml` + env vars.
- Inference steps must run under `torch.no_grad()` and support CPU fallback.
- Core logic lives in `src/`, not in notebooks.
- Meaningful errors on missing metadata (see `canonicalise_dataset_id` for the pattern).
