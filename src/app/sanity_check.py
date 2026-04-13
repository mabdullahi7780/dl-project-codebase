from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _file_exists(relative_path: str) -> bool:
    return (REPO_ROOT / relative_path).exists()


def static_repo_checks() -> list[CheckResult]:
    required_files = [
        "src/components/component0_qc.py",
        "src/components/component1_encoder.py",
        "src/components/component1_dann.py",
        "src/components/component2_txv.py",
        "src/components/component4_lung.py",
        "src/components/baseline_lesion_proposer.py",
        "src/components/component7_boundary.py",
        "src/components/component7_fp_auditor.py",
        "src/components/component7_refine.py",
        "src/components/component8_timika.py",
        "src/components/component9_json_output.py",
        "src/components/component10_report.py",
        "src/app/infer.py",
        "src/app/batch_infer.py",
    ]

    checks: list[CheckResult] = []
    for path in required_files:
        checks.append(
            CheckResult(
                name=f"file:{path}",
                ok=_file_exists(path),
                detail="present" if _file_exists(path) else "missing",
            )
        )
    return checks


def dependency_checks() -> list[CheckResult]:
    required = ["torch", "numpy", "PIL", "yaml", "cv2", "scipy"]
    optional = ["torchvision", "transformers", "torchxrayvision", "segment_anything"]

    checks: list[CheckResult] = []
    for name in required:
        ok = _module_available(name)
        checks.append(CheckResult(name=f"dep:{name}", ok=ok, detail="available" if ok else "missing"))
    for name in optional:
        ok = _module_available(name)
        checks.append(CheckResult(name=f"optional-dep:{name}", ok=ok, detail="available" if ok else "not installed"))
    return checks


def dynamic_component_checks() -> list[CheckResult]:
    if not all(_module_available(name) for name in ("torch", "numpy")):
        return [CheckResult(name="dynamic-checks", ok=False, detail="skipped because torch/numpy are not installed")]

    import numpy as np
    import torch

    from src.components.component0_qc import harmonise_sample
    from src.components.component1_dann import Component1DANNModel
    from src.components.component1_encoder import Component1EncoderConfig, build_component1_encoder
    from src.components.component2_txv import Component2SoftDomainContext
    from src.components.baseline_lesion_proposer import BaselineLesionProposer
    from src.components.component4_lung import Component4MedSAM
    from src.components.component8_timika import compute_baseline_timika
    from src.components.component9_json_output import generate_structured_json
    from src.components.component10_report import generate_baseline_report

    checks: list[CheckResult] = []

    sample = {
        "image": np.linspace(0, 255, 700 * 700, dtype=np.uint8).reshape(700, 700),
        "dataset_id": "nih",
        "image_id": "sanity-demo",
        "view": "PA",
    }
    harmonised = harmonise_sample(sample, apply_clahe=False)
    x224_min = float(harmonised.x_224.min().item())
    x224_max = float(harmonised.x_224.max().item())
    ok_component0 = (
        tuple(harmonised.x_1024.shape) == (1, 1024, 1024)
        and tuple(harmonised.x_224.shape) == (1, 224, 224)
        and tuple(harmonised.x_3ch.shape) == (3, 1024, 1024)
        and x224_min >= -1024.1
        and x224_max <= 1024.1
    )
    checks.append(
        CheckResult(
            name="component0-contract",
            ok=ok_component0,
            detail=(
                f"x_1024={tuple(harmonised.x_1024.shape)} "
                f"x_224={tuple(harmonised.x_224.shape)} range=[{x224_min:.1f},{x224_max:.1f}] "
                f"x_3ch={tuple(harmonised.x_3ch.shape)}"
            ),
        )
    )

    encoder = build_component1_encoder(Component1EncoderConfig(backend="mock"))
    c1_model = Component1DANNModel(encoder=encoder)
    img_emb, dom_logits = c1_model(torch.rand(2, 3, 1024, 1024), lambda_=0.5)
    checks.append(
        CheckResult(
            name="component1-contract",
            ok=tuple(img_emb.shape) == (2, 256, 64, 64) and tuple(dom_logits.shape) == (2, 4),
            detail=f"img_emb={tuple(img_emb.shape)} dom_logits={tuple(dom_logits.shape)}",
        )
    )

    c2_model = Component2SoftDomainContext(backend="mock")
    txv_output = c2_model.forward_features(harmonised.x_224.unsqueeze(0))
    checks.append(
        CheckResult(
            name="component2-contract",
            ok=tuple(txv_output.domain_ctx.shape) == (1, 256) and tuple(txv_output.pathology_logits.shape) == (1, 18),
            detail=f"domain_ctx={tuple(txv_output.domain_ctx.shape)} pathology_logits={tuple(txv_output.pathology_logits.shape)}",
        )
    )

    c4_model = Component4MedSAM(backend="mock")
    lung_output = c4_model.predict_masks(harmonised.x_3ch.unsqueeze(0))
    checks.append(
        CheckResult(
            name="component4-contract",
            ok=tuple(lung_output.lung_mask_256.shape) == (1, 1, 256, 256)
            and tuple(lung_output.lung_mask_1024.shape) == (1, 1, 1024, 1024),
            detail=f"lung_mask_256={tuple(lung_output.lung_mask_256.shape)} lung_mask_1024={tuple(lung_output.lung_mask_1024.shape)}",
        )
    )

    proposer = BaselineLesionProposer()
    lesion_proposal = proposer.propose(
        x_224=harmonised.x_224.unsqueeze(0),
        features_7x7=txv_output.features_7x7,
        pathology_logits=txv_output.pathology_logits,
        lung_mask_256=lung_output.lung_mask_256,
        classifier_weight=txv_output.classifier_weight,
    )
    checks.append(
        CheckResult(
            name="baseline-lesion-proposer-contract",
            ok=tuple(lesion_proposal.lesion_mask_coarse_256.shape) == (1, 1, 256, 256),
            detail=f"lesion_mask_coarse_256={tuple(lesion_proposal.lesion_mask_coarse_256.shape)}",
        )
    )

    lesion_mask = torch.zeros((1, 1024, 1024), dtype=torch.float32)
    evidence_metrics = compute_baseline_timika(lesion_mask, lung_output.lung_mask_1024[0])
    evidence_json = generate_structured_json(
        patient_id="sanity-demo",
        modality="CXR-PA",
        scanner_domain="baseline-mock",
        n_distinct_lesions=0,
        lesion_area_cm2=0.0,
        expert_routing=None,
        boundary_quality_score=0.0,
        fp_probability=0.0,
        alp=evidence_metrics.ALP,
        cavity_flag=evidence_metrics.cavity_flag,
        timika_score=evidence_metrics.timika_score,
        severity=evidence_metrics.severity,
        pathology_flags={},
        cavitation_confidence=evidence_metrics.cavitation_confidence,
    )
    report_text = generate_baseline_report(evidence_json)
    routing_present = "expert_routing" in evidence_json.get("segmentation", {})
    report_mentions_routing = "routing" in report_text.lower() or "moe" in report_text.lower()
    checks.append(
        CheckResult(
            name="baseline-reporting-path",
            ok=(not routing_present) and (not report_mentions_routing),
            detail=f"routing_in_json={routing_present} routing_in_report={report_mentions_routing}",
        )
    )

    return checks


def summarize(checks: list[CheckResult]) -> dict[str, Any]:
    missing_baseline_bits = [
        check.name
        for check in checks
        if (not check.ok) and not check.name.startswith("optional-dep:")
    ]
    baseline_ready = len(missing_baseline_bits) == 0
    return {
        "baseline_ready": baseline_ready,
        "missing_baseline_bits": missing_baseline_bits,
        "checks": [asdict(check) for check in checks],
    }


def main() -> None:
    checks = []
    checks.extend(static_repo_checks())
    checks.extend(dependency_checks())
    checks.extend(dynamic_component_checks())
    summary = summarize(checks)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
