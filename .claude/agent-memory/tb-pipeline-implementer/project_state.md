---
name: project_state
description: Current implementation status, key interfaces, and architectural decisions for the TB CXR pipeline
type: project
---

## Component Implementation Status (as of 2026-04-22)

Baseline path fully implemented end-to-end:
  C0 → C1 (SAM ViT-B MedSAM + LoRA + DANN) → C2 (TXV DenseNet121) → C4 (MedSAM lung segmentation) → baseline_lesion_proposer → C7 heuristics → C8 Timika → C9 JSON → C10 report

MoE path fully implemented:
  C3 (routing gate) → C5 (4-expert bank: consolidation/cavity/fibrosis/nodule) → C6 (weighted fusion) → upgraded C7 (boundary critic + reprompt refiner) → upgraded C8 (real cavity from Expert 2)

Components 3, 5, 6 are intentionally only in MoE path — no baseline stubs.

## Key Interface Contracts

HarmonisedCXR fields (load-bearing, do not rename):
  x_1024, x_224, x_3ch, meta

BaselineInferenceBundle fields (load-bearing):
  harmonised, img_emb, dom_logits, domain_ctx, pathology_logits,
  lung_mask_256, lung_mask_1024, lesion_mask_coarse_256,
  lesion_mask_refined_256, lesion_mask_refined_1024,
  boundary_score, fp_prob, alp, cavity_flag, timika_score, severity,
  evidence_json, report_text,
  pipeline_mode, routing_weights, expert_masks_256, mask_fused_256,
  mask_variance_256, cavity_mask_256

Component9 JSON keys (load-bearing for BioGPT grounding):
  patient_id, modality, scanner_domain, segmentation{}, scoring{ALP, cavity_flag,
  timika_score, severity, cavitation_confidence}, pathology_flags{}

## Checkpoint Path Conventions

All checkpoint paths go through configs/paths.yaml + env vars.
Env var: MEDSAM_VIT_B_CKPT (replaces old SAM_VIT_H_CKPT — fixed in .env.example 2026-04-22).
MedSAM ViT-B is shared between C1 and C4 (same weights, different wrappers).

## Class Rename (2026-04-22)

MockSAMViTHImageEncoder → MockMedSAMViTBImageEncoder
Back-compat alias preserved: MockSAMViTHImageEncoder = MockMedSAMViTBImageEncoder
Location: src/components/component1_encoder.py

## Pydantic Schema (2026-04-22)

src/components/component9_schema.py — EvidenceReport Pydantic v2 model.
generate_structured_json now validates via EvidenceReport.from_component9_dict before returning.
ALP key normalisation: component9 writes "ALP" (uppercase), schema stores as "alp", re-applied on return.
cavity_flag: component9 writes int (0/1), schema coerces to bool, re-cast to int on return.

## Checkpoint Provenance (2026-04-22)

compute_file_sha256(path) added to src/utils/checkpoints.py.
run_single_image_inference now writes checkpoint_provenance: {name: sha256} into run_summary.json.
"file-not-found" sentinel used when a configured checkpoint path does not exist on disk.

## gate_only Training Default

configs/moe.yaml has moe_training.joint.gate_only: true by default.
This freezes experts + fusion and only trains the routing gate.
Why: prevents catastrophic forgetting of production-quality expert decoders.
Full joint retrain: set gate_only: false in moe.yaml.

## Domain ID Mapping

DOMAIN_TO_ID: montgomery=0, shenzhen=1, tbx11k=2, nih_cxr14=3

## Test Infrastructure Notes

System Python 3.14 is used for tests (not the venv Python).
pytest tmp_path requires --basetemp="C:/tmp/pytest_tb_pipeline" to avoid Windows permission errors.
transformers package is NOT installed — BioGPT tests must handle this gracefully.
All tests are CPU-only; no GPU required.
