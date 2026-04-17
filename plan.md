# TB CXR Baseline-First Implementation Spec

This document is the coding brief for the **baseline-first implementation** of the TB chest X-ray pipeline.
It is written for coding agents such as Codex and Claude so they can work from the same contract.

The source pipeline is the revised **CXR-only** implementation with **TB Portals removed**. The revised pipeline keeps the core ideas of domain routing, multi-agent verification, and deterministic Timika scoring, but the current coding target is a **practical baseline** that can be built, tested, and delivered first.

---

## 1. Goal of this implementation

Build a **working end-to-end baseline** first.

The baseline must cover these components from the revised pipeline:

- **Component 0** — QC + normalisation
- **Component 1** — shared image encoder (**MedSAM ViT-B** + LoRA + DANN head)
  - *Note: downgraded from SAM ViT-H to MedSAM ViT-B so the whole pipeline fits on a Kaggle free-tier T4 16 GB. The `[B, 256, 64, 64]` embedding contract is unchanged, so nothing downstream cares which ViT produced it.*
- **Component 2** — soft domain context (TorchXRayVision DenseNet121)
- **Component 4** — lung mask module (MedSAM ViT-B)
- **Component 7** — verification/refinement
- **Component 8** — ALP + cavity flag + Timika score
- **Component 9** — structured JSON output
- **Component 10** — constrained report generation

The baseline must be **complete and deliverable** even though it does **not** yet include:

- **Component 3** — routing gate
- **Component 5** — MoE experts 1–4
- **Component 6** — expert merge/fuse

Those remaining components will be added later without breaking the baseline code structure.

---

## 2. Critical design decision for the baseline

The full pipeline expects lesion masks from the expert branch, but the current baseline intentionally skips Components 3, 5, and 6.
That means the baseline needs a **temporary lesion source** so that Components 7, 8, 9, and 10 can still run.

### Baseline replacement for missing Components 3, 5, 6

Implement a new baseline-only module:

- `baseline_lesion_proposer`

This module is **not** part of the final paper architecture.
It is a practical bridge so the baseline remains end-to-end.

### What `baseline_lesion_proposer` does

It produces a **coarse lesion mask** from:

- the **lung mask** from Component 4
- the **TXV pathology logits/features** from Component 2
- the **normalised image** from Component 0

### Recommended baseline strategy

Use a **Grad-CAM based suspicious-region map** built from TB-mimic classes available in TorchXRayVision.

Use the following classes as a suspicious set:

- consolidation
- infiltration
- fibrosis
- pleural_thickening
- effusion
- mass
- nodule
- pneumonia

Then:

1. compute Grad-CAM heatmaps on the TXV backbone for the suspicious classes
2. weight each heatmap by its sigmoid probability
3. sum the weighted maps into one suspiciousness map
4. resize to lung-mask resolution
5. multiply by lung mask
6. threshold adaptively
7. clean with morphology
8. return a binary coarse lesion mask

This is the baseline stand-in for the expert-produced `mask_fused / mask_refined` path.

### Why this choice

This keeps the baseline:

- image-grounded
- explainable
- cheap to implement
- compatible with Components 7, 8, 9, 10
- easy to replace later with the real expert branch

---

## 3. Non-negotiable baseline outputs

The baseline must produce all of the following for a single image and for a batch:

1. harmonised tensors from Component 0
2. SAM image embedding from Component 1
3. domain context + pathology logits from Component 2
4. lung mask from Component 4
5. coarse lesion mask from `baseline_lesion_proposer`
6. verification outputs from Component 7
7. final lesion mask after baseline refinement
8. deterministic metrics from Component 8
9. structured JSON from Component 9
10. human-readable report text from Component 10

The baseline is considered complete only when `python -m app.infer --image path/to/file.png` can generate:

- saved overlay image
- saved masks
- saved JSON
- saved report text

---

## 4. Scope assumptions from the revised pipeline

The implementation must follow these revised-scope facts:

- the project is now **CXR-only**
- **TB Portals is removed**
- the domain classifier is **4-way**, not 5-way
- the working datasets are:
  - Montgomery
  - Shenzhen
  - TBX11K
  - NIH ChestX-ray14

The full revised pipeline runs through Components 0–10 and computes ALP and Timika from the refined lesion mask and lung mask. The domain head is 4-class, and the pipeline is explicitly CXR-only. The model registry also states that the non-BioGPT GPU stack fits on a single 8 GB GPU in fp16, which is aligned with the intended hardware constraints for this project. fileciteturn2file0

---

## 5. Implementation priorities

### Priority A — baseline that runs end-to-end

Codex should implement this first:

- Component 0
- Component 2 inference
- Component 4 inference or training-ready module
- `baseline_lesion_proposer`
- baseline version of Component 7
- Component 8
- Component 9
- report generator with deterministic template first

### Priority B — add Component 1 training hooks cleanly

Then add:

- SAM ViT-H loader
- LoRA injection hooks
- DANN head module
- training script skeleton for domain adaptation

### Priority C — only after the baseline runs

Add support for the remaining full-pipeline modules later:

- Component 3 routing gate
- Component 5 experts
- Component 6 fusion
- full Component 7 refiner with Expert 3 integration
- BioGPT backed report generation

---

## 6. Repository structure to create

Codex should generate code using this structure.

```text
repo_root/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
│
├── configs/
│   ├── paths.yaml
│   ├── baseline.yaml
│   ├── component1_dann.yaml
│   ├── component4_lung.yaml
│   ├── component7_fp_auditor.yaml
│   └── reporting.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── splits/
│   └── cache/
│
├── checkpoints/
│   ├── sam/
│   ├── medsam/
│   ├── txv/
│   ├── baseline/
│   └── reports/
│
├── src/
│   ├── app/
│   │   ├── infer.py
│   │   ├── batch_infer.py
│   │   └── serve_contract.py
│   │
│   ├── core/
│   │   ├── types.py
│   │   ├── constants.py
│   │   ├── io.py
│   │   ├── seed.py
│   │   ├── logging_utils.py
│   │   └── device.py
│   │
│   ├── data/
│   │   ├── dataset_registry.py
│   │   ├── montgomery.py
│   │   ├── shenzhen.py
│   │   ├── tbx11k.py
│   │   ├── nih_cxr14.py
│   │   ├── transforms_qc.py
│   │   ├── harmonise.py
│   │   ├── split_builders.py
│   │   └── collate.py
│   │
│   ├── components/
│   │   ├── component0_qc.py
│   │   ├── component1_encoder.py
│   │   ├── component1_dann.py
│   │   ├── component2_txv.py
│   │   ├── component4_lung.py
│   │   ├── baseline_lesion_proposer.py
│   │   ├── component7_boundary.py
│   │   ├── component7_fp_auditor.py
│   │   ├── component7_refine.py
│   │   ├── component8_timika.py
│   │   ├── component9_json.py
│   │   ├── component10_report.py
│   │   └── faithfulness.py
│   │
│   ├── training/
│   │   ├── train_component1_dann.py
│   │   ├── train_component4_lung.py
│   │   ├── train_fp_auditor.py
│   │   ├── common.py
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── metrics_seg.py
│   │   ├── metrics_cls.py
│   │   ├── metrics_reporting.py
│   │   ├── evaluate_baseline.py
│   │   └── overlays.py
│   │
│   └── utils/
│       ├── gradcam.py
│       ├── morphology.py
│       ├── bbox.py
│       ├── json_schema.py
│       └── visualization.py
│
├── scripts/
│   ├── download_sam.sh
│   ├── download_medsam.sh
│   ├── prepare_splits.py
│   ├── cache_txv_features.py
│   └── run_baseline_demo.sh
│
└── tests/
    ├── test_component0.py
    ├── test_component2.py
    ├── test_component4.py
    ├── test_baseline_lesion_proposer.py
    ├── test_component8_timika.py
    ├── test_component9_json.py
    ├── test_infer_contract.py
    └── test_smoke_batch.py
```

---

## 7. Coding rules

Codex and Claude must follow these rules.

### General rules

- use **Python 3.11**
- use **PyTorch**
- use **typed dataclasses / pydantic models** where useful
- avoid notebooks for core logic
- notebooks may exist only for quick inspection, not as the main pipeline
- every component must have an isolated class or module
- every component must expose a clean input/output contract
- no hidden global state
- no hardcoded absolute paths

### Reliability rules

- every inference step must work with `torch.no_grad()`
- support CPU fallback where possible
- support mixed precision only if safe
- save intermediate artifacts optionally through a flag
- produce meaningful errors when metadata is missing

### Testing rules

- unit tests for deterministic modules
- smoke tests for model wrappers
- one end-to-end inference test with mock or tiny tensors

---

## 8. Central data contracts

### 8.1 HarmonisedCXR

Implement this first.

```python
from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class HarmonisedCXR:
    x_1024: torch.Tensor      # [1, 1024, 1024], float32 in [0,1]
    x_224: torch.Tensor       # [1, 224, 224], TXV-ready scale
    x_3ch: torch.Tensor       # [3, 1024, 1024]
    meta: dict[str, Any]      # dataset_id, scanner_domain, pixel_spacing_cm, image_id, etc.
```

This matches the revised plan, which explicitly uses a `HarmonisedCXR` object with `x_1024`, `x_224`, `x_3ch`, and metadata containing dataset and pixel spacing information. fileciteturn2file0

### 8.2 Baseline inference bundle

Implement a second dataclass used across inference:

```python
@dataclass
class BaselineInferenceBundle:
    harmonised: HarmonisedCXR
    img_emb: torch.Tensor | None
    dom_logits: torch.Tensor | None
    domain_ctx: torch.Tensor | None
    pathology_logits: torch.Tensor | None
    lung_mask_256: torch.Tensor | None
    lung_mask_1024: torch.Tensor | None
    lesion_mask_coarse_256: torch.Tensor | None
    lesion_mask_refined_256: torch.Tensor | None
    lesion_mask_refined_1024: torch.Tensor | None
    boundary_score: float | None
    fp_prob: float | None
    alp: float | None
    cavity_flag: int | None
    timika_score: float | None
    severity: str | None
    evidence_json: dict | None
    report_text: str | None
```

### 8.3 Dataloader sample contract

Each dataset reader should emit a dictionary with standard keys.

```python
{
    "image": np.ndarray | torch.Tensor,
    "image_id": str,
    "dataset_id": str,
    "domain_id": int,
    "view": str | None,
    "pixel_spacing_cm": float | None,
    "lung_mask": optional,
    "lesion_mask": optional,
    "bbox": optional,
    "labels": dict,
    "path": str,
}
```

Never make downstream code parse raw dataset-specific CSV columns directly.
Do that only once inside dataset adapters.

---

## 9. Baseline implementation details by component

# 9.1 Component 0 — QC + normalisation

Implement `src/components/component0_qc.py`.

### Required behavior

- validate image shape
- reject obviously invalid samples
- normalise by dataset-specific bit depth
- create `x_1024`, `x_224`, `x_3ch`
- attach metadata
- optionally apply CLAHE to `x_1024` only

### Rules from the revised pipeline

Implement these exact baseline-compatible rules:

- accept only PA if metadata exists
- reject images smaller than 512×512
- reject aspect ratio outside `[0.7, 1.4]`
- Montgomery 12-bit: divide by `4095.0`
- Shenzhen/TBX11K/NIH CXR14 8-bit: divide by `255.0`
- apply CLAHE to `x_1024` only
- CLAHE mandatory for Montgomery
- skip CLAHE for TBX11K and NIH CXR14 by default
- use TXV-style crop/resize for `x_224`
- repeat grayscale channel to create `x_3ch`

These preprocessing and harmonisation rules, including the `x_1024` / `x_224` / `x_3ch` split and CLAHE restrictions, are explicitly defined in the revised pipeline document. fileciteturn2file0

### Functions to implement

- `validate_view(...)`
- `normalise_by_dataset(...)`
- `make_x1024(...)`
- `make_x224_txv(...)`
- `apply_clahe_x1024(...)`
- `harmonise_sample(...) -> HarmonisedCXR`

### Tests

- test bit-depth scaling
- test shapes
- test CLAHE only touches `x_1024`
- test metadata pass-through

---

## 9.2 Component 1 — shared image encoder

Implement `src/components/component1_encoder.py` and `component1_dann.py`.

### Backbone choice: MedSAM ViT-B (not SAM ViT-H)

The revised baseline uses **MedSAM ViT-B** as the frozen backbone, not SAM ViT-H. Reasons:

- MedSAM ViT-B is already pretrained on medical imagery (including CXRs), so the frozen features are a stronger starting point for TB than natural-image SAM.
- ViT-B at 1024² with gradient checkpointing + LoRA fits on a Kaggle free-tier T4 (16 GB). SAM ViT-H does not.
- The neck contract (`[B, 256, 64, 64]`) is identical between ViT-B and ViT-H, so every downstream component (lesion proposer, lung masker, C7/C8/C9/C10) is unchanged.

Everywhere below that still says "SAM ViT-H" should be read as "MedSAM ViT-B" for the current baseline.

### Baseline requirement

For the first delivered baseline, **inference support is required**.
Training support should be scaffolded cleanly, but the first end-to-end demo may run with:

- frozen MedSAM ViT-B without trained LoRA
- DANN head class instantiated but optional during inference

### What must exist in code

- MedSAM ViT-B loader (via `segment_anything`'s `sam_model_registry["vit_b"]`, fed the MedSAM checkpoint)
- wrapper that returns image embedding `[B,256,64,64]`
- LoRA injection hooks (targeting `qkv` Linear layers inside the ViT blocks)
- DANN head class with 4-way output
- checkpoint loading hooks

### Exact shape contract

- input: `[B, 3, 1024, 1024]`
- output embedding: `[B, 256, 64, 64]`
- domain logits: `[B, 4]`

The revised plan fixes the shared encoder to **frozen MedSAM ViT-B** with LoRA adapters and a 4-class DANN head, with `img_emb` of shape `[B,256,64,64]` and `dom_logits` of shape `[B,4]`. Only the LoRA matrices and the DANN head are trainable; the MedSAM backbone weights are never updated.

### Training scaffold

Codex should add a training script for later use:

- gradient reversal layer
- weighted 4-domain sampling
- config-driven training loop
- save LoRA weights separately
- never resave the full frozen backbone unless explicitly requested

---

## 9.3 Component 2 — soft domain context

Implement `src/components/component2_txv.py`.

### Required behavior

- load TXV DenseNet121
- expose frozen backbone features
- expose pathology logits
- expose trainable domain routing head
- support Grad-CAM target extraction for the baseline lesion proposer

### Exact outputs

- penultimate pooled features: `[B,1024]`
- pathology logits: `[B,18]`
- domain context: `[B,256]`

The revised plan uses a frozen TorchXRayVision DenseNet121 backbone, pooled `[B,1024]` features, pathology logits, and a trainable projection head that outputs `domain_ctx [B,256]`. fileciteturn2file0

### Important coding note

The baseline should expose a single class such as:

```python
class TXVBackboneWrapper(nn.Module):
    def forward(self, x_224):
        return {
            "features_7x7": ..., 
            "pooled_features": ..., 
            "pathology_logits": ..., 
            "domain_ctx": ...
        }
```

This avoids running the same backbone twice.

---

## 9.4 Component 4 — lung mask module

Implement `src/components/component4_lung.py`.

### Required behavior

- load MedSAM ViT-B
- support fine-tuning mask decoder later
- support inference now
- use whole-image box prompt by default
- return both 256 and 1024 versions of the lung mask

### Exact contract

- input: `[B, 3, 1024, 1024]`
- raw output: `[B, 1, 256, 256]`
- upsampled output: `[B, 1, 1024, 1024]`

The revised pipeline defines Component 4 as MedSAM ViT-B with a frozen image encoder, a trainable mask decoder, and lung mask outputs at 256 then upsampled to 1024 for later ALP use. fileciteturn2file0

### Minimum training script

Implement `train_component4_lung.py` with:

- BCE + Dice
- Montgomery + Shenzhen
- early stopping on val Dice

---

## 9.5 Baseline lesion proposer

Implement `src/components/baseline_lesion_proposer.py`.

This is the baseline-only bridge module.

### Inputs

- `x_224`
- TXV intermediate feature maps
- pathology logits
- lung mask
- optional `x_1024` for edge-aware cleanup

### Outputs

- `lesion_mask_coarse_256`
- optional confidence map
- optional bounding box

### Algorithm

1. choose suspicious classes from TXV outputs
2. compute sigmoid probabilities
3. select classes with probability above a configurable threshold
4. compute Grad-CAM map per selected class
5. probability-weight and sum maps
6. normalise to `[0,1]`
7. resize to `256×256`
8. intersect with lung mask
9. threshold using either:
   - Otsu on lung-only values, or
   - fixed threshold from config
10. clean with morphology:
   - opening
   - closing
   - remove tiny regions
11. return binary mask

### Why this module must exist

Without Components 3/5/6, there is no lesion segmentation path in the real architecture.
This module is therefore required so that Components 7, 8, 9, and 10 can be built and tested now.

### Acceptance condition

For a single image, this module must return a non-empty mask on at least some abnormal TB-like cases and mostly empty masks on clear negatives.
It does not need paper-level accuracy.
It only needs to support a coherent baseline pipeline.

---

## 9.6 Component 7 — baseline verification and refinement

Implement three files:

- `component7_boundary.py`
- `component7_fp_auditor.py`
- `component7_refine.py`

### Important baseline adaptation

The revised full pipeline expects:

- a ResNet18 boundary critic
- a DenseNet-based FP auditor
- a re-prompted refinement step tied to the expert boundary decoder

But because the baseline skips expert decoders, Component 7 must be split into:

### 7A. Baseline boundary quality scorer

For the **first deliverable**, implement a **heuristic boundary scorer** first, then a trainable ResNet18 version second.

#### Heuristic score features

Compute a score from:

- lesion area fraction inside lung
- edge alignment with Sobel/Canny image gradients
- number of connected components
- compactness / perimeter-area ratio
- whether mask spills outside lungs

Return `boundary_score in [0,1]`.

#### Later optional replacement

Keep the file/class boundary so the heuristic version can later be replaced by the true ResNet18 critic.

### 7B. FP auditor

This one **should be implemented as trainable** because it is feasible now.

Use:

- pooled TXV features
- pathology logits
- simple MLP head

That matches the revised pipeline idea closely.

Required output:

- `fp_prob in [0,1]`

The revised plan explicitly uses the TXV DenseNet backbone plus pathology logits to train a small MLP FP head, with NIH ChestX-ray14 as the main source of hard negatives and a recommendation to cache frozen features to disk. fileciteturn2file0

### 7C. Baseline refinement

For the baseline, do **not** depend on Expert 3.

Implement refinement as:

1. clip lesion mask to lung mask
2. remove tiny components
3. fill small holes
4. optionally sharpen boundary using edge-aware erosion/dilation
5. if `fp_prob` is high, suppress or downweight the mask
6. if `boundary_score` is poor, run one stronger morphology cleanup pass

Return:

- `lesion_mask_refined_256`

### Interface

```python
refined = refine_mask(
    lesion_mask_coarse_256,
    lung_mask_256,
    x_1024,
    boundary_score,
    fp_prob,
    config,
)
```

---

## 9.7 Component 8 — ALP + cavity flag + Timika

Implement `src/components/component8_timika.py`.

### Baseline note

The revised pipeline computes these metrics deterministically from:

- refined lesion mask
- lung mask
- cavity signal

For the baseline, because Expert 2 is not implemented yet, create a baseline cavity path:

- either derive a cavity candidate map from the lesion proposer using high-intensity ring-like regions inside lesions
- or return `cavity_flag = 0` by default in baseline v1 and expose the code path for later upgrade

### Recommended baseline v1

Use this staged behavior:

- baseline v1.0: `cavity_flag = 0`, `cavitation_confidence = "not-assessed-baseline"`
- baseline v1.1: add a simple cavity heuristic

This keeps the pipeline honest and prevents fake cavitation claims.

### Deterministic outputs

- `ALP`
- `cavity_flag`
- `timika_score`
- `severity`

The revised plan defines ALP as lesion-in-lung area percentage, cavity flag via connected components, and Timika as `ALP + 40 * cavity_flag`, with severity bands derived from ALP. It also stresses that this step is deterministic and CPU-based. fileciteturn2file0

### Safety rule

Do not hallucinate cavity detection in the baseline.
If cavity logic is not implemented yet, state that clearly in JSON and report text.

---

## 9.8 Component 9 — structured JSON output

Implement `src/components/component9_json.py`.

### Requirement

This module defines the contract between vision outputs and report generation.

Create:

- a pydantic schema
- serializer
- validator

### Minimum JSON fields

```json
{
  "patient_id": "...",
  "modality": "CXR-PA",
  "scanner_domain": "...",
  "segmentation": {
    "n_distinct_lesions": 0,
    "lesion_area_cm2": 0.0,
    "boundary_quality_score": 0.0,
    "fp_probability": 0.0
  },
  "scoring": {
    "ALP": 0.0,
    "cavity_flag": 0,
    "timika_score": 0.0,
    "severity": "mild",
    "cavitation_confidence": "not-assessed-baseline"
  },
  "pathology_flags": {}
}
```

The revised pipeline defines Component 9 as the grounding contract for report generation and includes fields such as `boundary_quality_score`, `fp_probability`, `ALP`, `cavity_flag`, `timika_score`, `severity`, and `cavitation_confidence`. fileciteturn2file0

### Baseline rule

Only include fields that are truly supported.
If a value is approximate, label it as approximate in either a companion metadata field or the report text.

---

## 9.9 Component 10 — constrained report generator

Implement `src/components/component10_report.py`.

### Important baseline rule

Do **not** block the baseline on BioGPT.

Implement this in two layers:

### Layer 1 — deterministic template report

Must exist first.

Generate a short report from JSON with rules like:

- mention projection/modality if known
- mention whether suspicious lung abnormalities are present
- mention whether findings are unilateral/bilateral if available
- mention ALP and severity
- mention that cavitation was not assessed in baseline if applicable
- mention confidence caveat if fp probability is high

### Layer 2 — optional BioGPT adapter

Create the interface for later use:

```python
class ReportGenerator:
    def generate(self, evidence_json: dict) -> str:
        ...
```

and

```python
class TemplateReportGenerator(ReportGenerator):
    ...

class BioGPTReportGenerator(ReportGenerator):
    ...
```

The revised pipeline uses JSON-grounded report generation and a faithfulness check that maps report claims back to JSON evidence. It also recommends few-shot first and optional fine-tuning only later. fileciteturn2file0

### Faithfulness requirement

A report generator must never mention:

- cavity unless JSON supports it
- severity beyond what Timika/ALP supports
- laterality if not computed
- disease certainty stronger than the pipeline can justify

Implement a lightweight faithfulness checker that validates the final text against JSON fields.

---

## 10. Training plan for the baseline codebase

This section tells Codex what training scripts must exist even if not all are run immediately.

### Baseline-trainable now

1. `train_component4_lung.py`
2. `train_component1_dann.py`
3. `train_fp_auditor.py`

### Optional later during baseline cycle

4. train heuristic threshold parameters for `baseline_lesion_proposer`
5. swap heuristic boundary score for ResNet18 boundary critic

### Not in current baseline training

- routing gate
- experts 1–4
- fusion
- expert-based refiner

The revised phase ordering places lung decoder first, then SAM LoRA + DANN, then expert pretraining, then joint MoE training, followed by boundary critic and FP auditor, with report tuning optional last. That phase plan is the source of the current baseline ordering even though the expert phases are intentionally deferred. fileciteturn2file0

---

## 11. CLI commands that must exist

Codex should create these commands.

### Single image inference

```bash
python -m src.app.infer \
  --image path/to/image.png \
  --dataset nih \
  --outdir outputs/demo
```

### Batch inference

```bash
python -m src.app.batch_infer \
  --csv path/to/list.csv \
  --outdir outputs/batch
```

### Train lung mask module

```bash
python -m src.training.train_component4_lung --config configs/component4_lung.yaml
```

### Train DANN head + LoRA scaffold

```bash
python -m src.training.train_component1_dann --config configs/component1_dann.yaml
```

### Train FP auditor

```bash
python -m src.training.train_fp_auditor --config configs/component7_fp_auditor.yaml
```

---

## 12. Acceptance criteria for baseline delivery

The baseline is delivered only if all of these are true.

### Functional acceptance

- can load one image and run end-to-end
- produces lung mask
- produces coarse lesion mask
- refines that lesion mask
- computes ALP
- computes Timika score
- writes JSON
- writes report text
- saves overlay visualisation

### Engineering acceptance

- code is modular
- configs are externalised
- path handling is clean
- no notebook dependence for core logic
- unit tests pass
- smoke test passes

### Honest-baseline acceptance

- no fake MoE logic is claimed in the baseline
- no cavity claims unless actually supported
- report text reflects baseline limitations
- JSON and report are consistent

---

## 13. What to defer until after baseline delivery

These are the remaining components and how they should be added later.

### Remaining Component 3 — routing gate

Add after the real expert masks exist.
Current baseline should not include a fake routing gate.

### Remaining Component 5 — experts 1 to 4

Add as separate decoders with shared interface:

```python
class ExpertDecoder(nn.Module):
    def forward(self, image_emb, prompts, dense_prompt=None):
        ...
```

### Remaining Component 6 — fuse expert outputs

Add logit-space weighted fusion after Component 3 and 5 are real.

### Future upgrade path for Component 7

Replace:

- heuristic boundary score → ResNet18 critic
- morphology refiner → expert-3 guided reprompt refiner

### Future upgrade path for Component 8

Replace:

- baseline cavity logic / default `0`
- with true cavity path from Expert 2

### Future upgrade path for Component 10

Replace or augment:

- template report generator
- with BioGPT grounded generation

---

## 14. Explicit implementation notes for Codex

Codex should follow these execution notes closely.

### Do first

1. create project skeleton
2. implement dataclasses and schemas
3. implement Component 0
4. implement TXV wrapper
5. implement MedSAM lung wrapper
6. implement baseline lesion proposer
7. implement baseline refinement
8. implement deterministic Timika module
9. implement JSON serializer
10. implement template report generator
11. add CLI
12. add tests

### Do not do yet

- do not pretend the baseline is the final MoE system
- do not hardwire expert outputs into baseline code
- do not tightly couple report generation to BioGPT availability
- do not hide unsupported fields in the JSON

### Use extension-friendly names

Use names like:

- `lesion_mask_source`
- `lesion_mask_coarse`
- `lesion_mask_refined`

Do not use names that assume expert fusion already exists.

---

## 15. Suggested milestone checklist

### Milestone 1 — runnable baseline core

- [ ] Component 0 works
- [ ] TXV wrapper works
- [ ] lung mask wrapper works
- [ ] baseline lesion proposer works
- [ ] single-image inference works

### Milestone 2 — deliverable outputs

- [ ] baseline refinement works
- [ ] ALP/Timika works
- [ ] JSON output works
- [ ] report generator works
- [ ] overlays saved

### Milestone 3 — training hooks

- [ ] lung training script exists
- [ ] DANN training scaffold exists
- [ ] FP auditor training script exists
- [ ] caching scripts exist

### Milestone 4 — future-ready architecture

- [ ] Component 3 placeholder interface exists
- [ ] expert interface exists
- [ ] fusion interface exists
- [ ] baseline modules can be swapped later

---

## 16. Final instruction to coding agents

Build the baseline as a **clean, honest, modular CXR-only pipeline**.

The main objective is not to imitate the final MoE pipeline prematurely.
The objective is to deliver a **credible baseline** that:

- runs end-to-end
- respects the revised no-TB-Portals plan
- preserves the same final output contract
- makes it easy to plug in Components 3, 5, and 6 later

The revised source document explicitly keeps the project CXR-only, uses 4 domains, preserves Components 0–10 as the overall structure, and computes final outputs through a refined mask, Timika engine, JSON serialisation, and report generation. This implementation brief keeps that structure while inserting a baseline lesion source until the MoE branch is added. fileciteturn2file0
