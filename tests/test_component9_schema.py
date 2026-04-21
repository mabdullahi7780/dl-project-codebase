"""Tests for the Component 9 Pydantic v2 schema (component9_schema.py).

Covers:
- Valid reports constructed from the full generate_structured_json pipeline
- Valid reports constructed directly from EvidenceReport
- Invalid fields that must raise ValidationError
- EvidenceReport.from_component9_dict normalisation (ALP key, cavity_flag cast)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.components.component9_schema import (
    EvidenceReport,
    PathologyFlags,
    SegmentationOutput,
    ScoringOutput,
)


# ---------------------------------------------------------------------------
# Helper: minimal valid dicts
# ---------------------------------------------------------------------------

def _valid_seg(**overrides) -> dict:
    base = {
        "n_distinct_lesions": 2,
        "lesion_area_cm2": 8.5,
        "boundary_quality_score": 0.72,
        "fp_probability": 0.15,
        "expert_routing": None,
    }
    base.update(overrides)
    return base


def _valid_scoring(**overrides) -> dict:
    base = {
        "alp": 34.0,
        "cavity_flag": False,
        "timika_score": 34.0,
        "severity": "moderate",
        "cavitation_confidence": "radiographic-only",
    }
    base.update(overrides)
    return base


def _valid_report(**overrides) -> dict:
    base = {
        "patient_id": "TEST_001",
        "modality": "CXR-PA",
        "scanner_domain": "montgomery",
        "segmentation": _valid_seg(),
        "scoring": _valid_scoring(),
        "pathology_flags": {"consolidation": True, "cavitation": False},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# SegmentationOutput
# ---------------------------------------------------------------------------

class TestSegmentationOutput:
    def test_valid(self) -> None:
        seg = SegmentationOutput(**_valid_seg())
        assert seg.n_distinct_lesions == 2
        assert seg.lesion_area_cm2 == pytest.approx(8.5)
        assert seg.expert_routing is None

    def test_with_expert_routing(self) -> None:
        seg = SegmentationOutput(
            **_valid_seg(expert_routing={"consolidation": 0.4, "cavity": 0.3, "fibrosis": 0.2, "nodule": 0.1})
        )
        assert seg.expert_routing is not None
        assert len(seg.expert_routing) == 4

    def test_negative_n_lesions_raises(self) -> None:
        with pytest.raises(ValidationError):
            SegmentationOutput(**_valid_seg(n_distinct_lesions=-1))

    def test_boundary_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            SegmentationOutput(**_valid_seg(boundary_quality_score=1.5))

    def test_fp_probability_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            SegmentationOutput(**_valid_seg(fp_probability=-0.1))

    def test_expert_routing_weight_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            SegmentationOutput(**_valid_seg(expert_routing={"expert1": 2.0}))


# ---------------------------------------------------------------------------
# ScoringOutput
# ---------------------------------------------------------------------------

class TestScoringOutput:
    def test_valid(self) -> None:
        s = ScoringOutput(**_valid_scoring())
        assert s.severity == "moderate"
        assert s.cavity_flag is False

    def test_severity_normalised_to_lowercase(self) -> None:
        s = ScoringOutput(**_valid_scoring(severity="MILD"))
        assert s.severity == "mild"

    def test_invalid_severity_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScoringOutput(**_valid_scoring(severity="critical"))

    def test_invalid_cavitation_confidence_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScoringOutput(**_valid_scoring(cavitation_confidence="unknown-source"))

    def test_all_valid_confidence_strings(self) -> None:
        for conf in ("radiographic-only", "expert2-radiographic", "not-assessed-baseline"):
            s = ScoringOutput(**_valid_scoring(cavitation_confidence=conf))
            assert s.cavitation_confidence == conf

    def test_alp_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScoringOutput(**_valid_scoring(alp=101.0))


# ---------------------------------------------------------------------------
# PathologyFlags
# ---------------------------------------------------------------------------

class TestPathologyFlags:
    def test_valid(self) -> None:
        pf = PathologyFlags(
            top_classes=["consolidation"],
            probabilities={"consolidation": 0.82, "cavitation": 0.04},
        )
        assert "consolidation" in pf.top_classes

    def test_probability_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            PathologyFlags(
                top_classes=[],
                probabilities={"bad_class": 1.5},
            )


# ---------------------------------------------------------------------------
# EvidenceReport
# ---------------------------------------------------------------------------

class TestEvidenceReport:
    def test_valid_round_trip(self) -> None:
        report = EvidenceReport(**_valid_report())
        dumped = report.model_dump()
        assert dumped["patient_id"] == "TEST_001"
        assert dumped["segmentation"]["n_distinct_lesions"] == 2
        assert dumped["scoring"]["severity"] == "moderate"

    def test_missing_patient_id_raises(self) -> None:
        d = _valid_report()
        del d["patient_id"]
        with pytest.raises(ValidationError):
            EvidenceReport(**d)

    def test_empty_patient_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceReport(**_valid_report(patient_id=""))

    def test_pathology_flags_invalid_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceReport(**_valid_report(pathology_flags={"cls": "yes"}))

    def test_from_component9_dict_alp_key_normalisation(self) -> None:
        """from_component9_dict must handle 'ALP' key from component9 output."""
        raw = {
            "patient_id": "MCU_0034",
            "modality": "CXR-PA",
            "scanner_domain": "shenzhen",
            "segmentation": {
                "n_distinct_lesions": 1,
                "lesion_area_cm2": 5.0,
                "boundary_quality_score": 0.6,
                "fp_probability": 0.2,
            },
            "scoring": {
                "ALP": 25.0,       # uppercase — as written by component9
                "cavity_flag": 0,  # int — as written by component9
                "timika_score": 25.0,
                "severity": "mild",
                "cavitation_confidence": "not-assessed-baseline",
            },
            "pathology_flags": {"consolidation": False},
        }
        report = EvidenceReport.from_component9_dict(raw)
        assert report.scoring.alp == pytest.approx(25.0)
        assert report.scoring.cavity_flag is False  # coerced from int 0

    def test_from_component9_dict_cavity_flag_coercion(self) -> None:
        """cavity_flag=1 (int) must be coerced to True (bool)."""
        raw = _valid_report()
        raw["scoring"] = {
            "ALP": 50.0,
            "cavity_flag": 1,
            "timika_score": 50.0,
            "severity": "severe",
            "cavitation_confidence": "expert2-radiographic",
        }
        report = EvidenceReport.from_component9_dict(raw)
        assert report.scoring.cavity_flag is True

    def test_run_metadata_optional(self) -> None:
        """run_metadata should default to empty dict when not provided."""
        report = EvidenceReport(**_valid_report())
        assert report.run_metadata == {}

    def test_run_metadata_populated(self) -> None:
        report = EvidenceReport(**_valid_report(run_metadata={"model_version": "v1.2", "run_id": "abc"}))
        assert report.run_metadata["model_version"] == "v1.2"


# ---------------------------------------------------------------------------
# Integration: generate_structured_json now validates via schema
# ---------------------------------------------------------------------------

class TestComponent9Integration:
    def test_generate_structured_json_passes_schema(self) -> None:
        """generate_structured_json must return a schema-conforming dict."""
        from src.components.component9_json_output import generate_structured_json

        result = generate_structured_json(
            patient_id="MCU_0034",
            modality="CXR-PA",
            scanner_domain="US-CR",
            n_distinct_lesions=3,
            lesion_area_cm2=12.421,
            expert_routing={"consolidation": 0.6, "cavity": 0.2, "fibrosis": 0.1, "nodule": 0.1},
            boundary_quality_score=0.834,
            fp_probability=0.071,
            alp=34.72,
            cavity_flag=1,
            timika_score=74.7,
            severity="moderate",
            pathology_flags={"consolidation": True, "cavitation": True},
            cavitation_confidence="expert2-radiographic",
        )
        # Must still contain the top-level keys the rest of the pipeline reads
        assert result["patient_id"] == "MCU_0034"
        assert result["scoring"]["ALP"] == pytest.approx(34.7)
        assert result["scoring"]["cavity_flag"] == 1
        assert result["scoring"]["cavitation_confidence"] == "expert2-radiographic"

    def test_invalid_severity_from_timika_raises(self) -> None:
        """If Component 8 somehow produces an invalid severity, Component 9 must raise."""
        from src.components.component9_json_output import generate_structured_json

        with pytest.raises(ValueError, match="severity"):
            generate_structured_json(
                patient_id="BAD",
                modality="CXR",
                scanner_domain="test",
                n_distinct_lesions=0,
                lesion_area_cm2=0.0,
                expert_routing=None,
                boundary_quality_score=0.5,
                fp_probability=0.1,
                alp=0.0,
                cavity_flag=0,
                timika_score=0.0,
                severity="catastrophic",   # invalid severity
                pathology_flags={},
            )
