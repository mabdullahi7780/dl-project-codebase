"""Component 9 — Pydantic v2 schema models.

Defines the canonical structure of the structured evidence JSON produced
by ``component9_json_output.generate_structured_json``.  All downstream
consumers (BioGPT grounding, Component 10 report, external APIs) should
validate their input against ``EvidenceReport`` before processing.

Usage
-----
    from src.components.component9_schema import EvidenceReport

    validated = EvidenceReport(**raw_dict).model_dump()
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class SegmentationOutput(BaseModel):
    """Spatial lesion quantification from Components 4 + 7."""

    n_distinct_lesions: int = Field(..., ge=0, description="Number of connected lesion regions after refinement.")
    lesion_area_cm2: float = Field(..., ge=0.0, description="Total lesion area in square centimetres.")
    boundary_quality_score: float = Field(..., ge=0.0, le=1.0, description="Boundary quality score from Component 7 (0–1).")
    fp_probability: float = Field(..., ge=0.0, le=1.0, description="False-positive probability from Component 7 FP auditor.")
    expert_routing: Optional[dict[str, float]] = Field(
        default=None,
        description="Per-expert routing weights from Component 3 (MoE path only; None on baseline path).",
    )

    @field_validator("expert_routing")
    @classmethod
    def _routing_weights_in_range(
        cls, v: dict[str, float] | None
    ) -> dict[str, float] | None:
        if v is None:
            return v
        for key, weight in v.items():
            if not (0.0 <= weight <= 1.0):
                raise ValueError(
                    f"expert_routing[{key!r}] = {weight} is outside [0, 1]."
                )
        return v


class ScoringOutput(BaseModel):
    """Timika scoring result from Component 8."""

    alp: float = Field(..., ge=0.0, le=100.0, description="Affected Lung Percentage (0–100).")
    cavity_flag: bool = Field(..., description="True if radiographic cavitation is detected.")
    timika_score: float = Field(..., ge=0.0, description="Timika severity score.")
    severity: str = Field(..., description="Severity category: 'mild', 'moderate', or 'severe'.")
    cavitation_confidence: str = Field(
        ...,
        description=(
            "Source of cavity assessment: "
            "'radiographic-only', 'expert2-radiographic', "
            "'not-assessed-baseline'."
        ),
    )

    @field_validator("severity")
    @classmethod
    def _severity_must_be_valid(cls, v: str) -> str:
        allowed = {"mild", "moderate", "severe"}
        if v.lower() not in allowed:
            raise ValueError(f"severity must be one of {sorted(allowed)}, got {v!r}.")
        return v.lower()

    @field_validator("cavitation_confidence")
    @classmethod
    def _confidence_must_be_valid(cls, v: str) -> str:
        allowed = {"radiographic-only", "expert2-radiographic", "not-assessed-baseline"}
        if v not in allowed:
            raise ValueError(
                f"cavitation_confidence must be one of {sorted(allowed)}, got {v!r}."
            )
        return v


class PathologyFlags(BaseModel):
    """TXV pathology classification flags.

    Keys are TXV class names; values are boolean flag or probability.
    We accept ``dict[str, Any]`` because the downstream code writes
    booleans but older snapshots may store floats.
    """

    top_classes: list[str] = Field(
        default_factory=list,
        description="Class names whose predicted probability exceeded the flag threshold.",
    )
    probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Raw sigmoid probabilities keyed by class name.",
    )

    @field_validator("probabilities")
    @classmethod
    def _probs_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for name, prob in v.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(
                    f"probabilities[{name!r}] = {prob} is outside [0, 1]."
                )
        return v


class EvidenceReport(BaseModel):
    """Top-level evidence report validated before being passed to Component 10.

    Mirrors the dict schema produced by
    ``component9_json_output.generate_structured_json``.  The ``pathology_flags``
    field accepts the legacy flat-dict format (``{class: bool}``) OR the
    structured ``PathologyFlags`` sub-model.
    """

    patient_id: str = Field(..., min_length=1)
    modality: str = Field(..., description="Imaging modality, e.g. 'CXR-PA' or 'CXR'.")
    scanner_domain: str = Field(..., description="Dataset / scanner domain identifier.")
    segmentation: SegmentationOutput
    scoring: ScoringOutput
    pathology_flags: Any = Field(
        ...,
        description=(
            "Either a flat dict {class_name: bool} (legacy / current pipeline) "
            "or a PathologyFlags model."
        ),
    )
    run_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional key/value metadata (run timestamp, model versions, etc.).",
    )

    @field_validator("pathology_flags")
    @classmethod
    def _coerce_pathology_flags(cls, v: Any) -> Any:
        """Accept both flat {str: bool} dicts and PathologyFlags instances."""
        if isinstance(v, dict):
            # Flat dict — validate each value is bool or float-like
            for key, val in v.items():
                if not isinstance(val, (bool, int, float)):
                    raise ValueError(
                        f"pathology_flags[{key!r}] must be bool or numeric, got {type(val).__name__}."
                    )
            return v
        if isinstance(v, PathologyFlags):
            return v
        raise ValueError(
            f"pathology_flags must be a dict or PathologyFlags instance, got {type(v).__name__}."
        )

    @classmethod
    def from_component9_dict(cls, d: dict[str, Any]) -> "EvidenceReport":
        """Construct from the raw dict produced by ``generate_structured_json``.

        Handles the ALP capitalisation difference (component9 writes ``"ALP"``
        while the Pydantic field is ``alp``).
        """
        scoring_raw = dict(d.get("scoring", {}))
        # Normalise ALP key
        if "ALP" in scoring_raw and "alp" not in scoring_raw:
            scoring_raw["alp"] = scoring_raw.pop("ALP")
        # Normalise cavity_flag: component9 stores int (0/1); Pydantic expects bool
        if "cavity_flag" in scoring_raw:
            scoring_raw["cavity_flag"] = bool(scoring_raw["cavity_flag"])

        seg_raw = dict(d.get("segmentation", {}))

        return cls(
            patient_id=d["patient_id"],
            modality=d["modality"],
            scanner_domain=d["scanner_domain"],
            segmentation=SegmentationOutput(**seg_raw),
            scoring=ScoringOutput(**scoring_raw),
            pathology_flags=d.get("pathology_flags", {}),
            run_metadata=d.get("run_metadata", {}),
        )
