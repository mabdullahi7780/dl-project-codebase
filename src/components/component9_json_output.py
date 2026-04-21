import json
from typing import Dict, Any


def _validate_evidence_report(output: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the output dict against the EvidenceReport Pydantic schema.

    Returns the validated dict (re-serialised via model_dump) so callers
    always receive a schema-conforming structure.  Import is deferred to
    keep component9_json_output importable even if pydantic is missing
    (though pydantic>=2.0 is listed in requirements.txt).
    """
    try:
        from src.components.component9_schema import EvidenceReport
        validated = EvidenceReport.from_component9_dict(output).model_dump()
        # Re-apply original ALP capitalisation so downstream code is unaffected
        if "scoring" in validated and "alp" in validated["scoring"]:
            validated["scoring"]["ALP"] = validated["scoring"].pop("alp")
        # Restore int cavity_flag (Pydantic converts to bool; keep legacy int)
        if "scoring" in validated and "cavity_flag" in validated["scoring"]:
            validated["scoring"]["cavity_flag"] = int(validated["scoring"]["cavity_flag"])
        return validated
    except Exception as exc:  # pragma: no cover — only fires on schema mismatch
        raise ValueError(
            f"Component 9 JSON failed schema validation: {exc}\nPayload: {output}"
        ) from exc


def generate_structured_json(
    patient_id: str,
    modality: str,
    scanner_domain: str,
    n_distinct_lesions: int,
    lesion_area_cm2: float,
    expert_routing: Dict[str, float] | None,
    boundary_quality_score: float,
    fp_probability: float,
    alp: float,
    cavity_flag: int,
    timika_score: float,
    severity: str,
    pathology_flags: Dict[str, bool],
    *,
    cavitation_confidence: str = "radiographic-only",
) -> Dict[str, Any]:
    """
    Component 9: Structured JSON Output
    Serialiser - defines the grounding contract for BioGPT.
    The cavitation_confidence field is hardcoded to "radiographic-only" 
    to reflect that cavity detection is CXR-based, not CT-confirmed.
    """
    
    # Ensure float rounding matches the expected display precision where useful
    segmentation = {
        "n_distinct_lesions": n_distinct_lesions,
        "lesion_area_cm2": round(float(lesion_area_cm2), 2),
        "boundary_quality_score": round(float(boundary_quality_score), 2),
        "fp_probability": round(float(fp_probability), 2),
    }
    if expert_routing:
        segmentation["expert_routing"] = {k: round(float(v), 2) for k, v in expert_routing.items()}

    output = {
        "patient_id": patient_id,
        "modality": modality,
        "scanner_domain": scanner_domain,
        "segmentation": segmentation,
        "scoring": {
            "ALP": round(float(alp), 1),
            "cavity_flag": int(cavity_flag),
            "timika_score": round(float(timika_score), 1),
            "severity": severity,
            "cavitation_confidence": cavitation_confidence
        },
        "pathology_flags": pathology_flags
    }

    return _validate_evidence_report(output)

def save_structured_json(output_dict: Dict[str, Any], filepath: str) -> None:
    """Save the JSON payload to disk."""
    with open(filepath, 'w') as f:
        json.dump(output_dict, f, indent=2)
