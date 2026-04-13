from __future__ import annotations

from src.components.component10_report import TemplateReportGenerator


def test_template_report_ignores_routing_fields_in_baseline() -> None:
    generator = TemplateReportGenerator()
    evidence_json = {
        "patient_id": "demo-001",
        "modality": "CXR-PA",
        "segmentation": {
            "n_distinct_lesions": 1,
            "lesion_area_cm2": 3.2,
            "expert_routing": {"E1": 0.7, "E3": 0.3},
            "fp_probability": 0.2,
        },
        "scoring": {
            "ALP": 8.5,
            "cavity_flag": 0,
            "severity": "mild",
            "cavitation_confidence": "not-assessed-baseline",
        },
    }

    report = generator.generate(evidence_json).lower()

    assert "routing" not in report
    assert "moe" not in report
    assert "cavitation was not assessed" in report
