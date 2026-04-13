# TB CXR Baseline

This repository follows the baseline-first structure described in `plan.md`.

Current implementation status:

- full directory scaffold is in place
- Component 0 (`QC + normalisation`) is implemented
- Component 1 (`shared image encoder + LoRA + DANN head`) is implemented
- a companion notebook lives in `notebooks/component0_qc.ipynb`
- a companion notebook lives in `notebooks/component1_dann.ipynb`
- tests for Component 0 live in `tests/test_component0.py`
- tests for Component 1 live in `tests/test_component1.py`

The core preprocessing logic is intentionally kept in `src/components/component0_qc.py`.
The core Component 1 logic is intentionally kept in `src/components/component1_encoder.py`,
`src/components/component1_dann.py`, and `src/training/train_component1_dann.py`.
The notebooks are for inspection and manual runs against datasets stored on an external HDD.

## Environment

- Python `3.11`
- PyTorch-based preprocessing pipeline
- OpenCV is used for CLAHE

Install the dependencies from `requirements.txt`, then open the notebook or import the module directly.

## External datasets

The notebook is written so dataset roots can point to an external drive, for example `E:\datasets`.
You can also store those paths in environment variables or in `configs/paths.yaml`.

## Implemented files

- `src/core/types.py`
- `src/core/constants.py`
- `src/data/transforms_qc.py`
- `src/data/harmonise.py`
- `src/components/component0_qc.py`
- `src/components/component1_encoder.py`
- `src/components/component1_dann.py`
- `src/training/train_component1_dann.py`
- `tests/test_component0.py`
- `tests/test_component1.py`
- `notebooks/component0_qc.ipynb`
- `notebooks/component1_dann.ipynb`
