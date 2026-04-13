import json
import tempfile
import os
from src.components.component9_json_output import generate_structured_json, save_structured_json

def test_component9_json_schema():
    """Verify Component 9 constructs the exact JSON dictionary as required by BioGPT Grounding."""
    result = generate_structured_json(
        patient_id="MCU_0034",
        modality="CXR-PA",
        scanner_domain="US-CR",
        n_distinct_lesions=3,
        lesion_area_cm2=12.421,
        expert_routing={"E1": 0.612, "E2": 0.221, "E3": 0.10, "E4": 0.07},
        boundary_quality_score=0.834,
        fp_probability=0.071,
        alp=34.72,
        cavity_flag=1,
        timika_score=74.7,
        severity="moderate",
        pathology_flags={
            "consolidation": True,
            "cavitation": True,
            "effusion": False,
            "fibrosis": False
        }
    )
    
    # Top-level keys
    assert "patient_id" in result
    assert result["patient_id"] == "MCU_0034"
    assert "segmentation" in result
    assert "scoring" in result
    assert "pathology_flags" in result
    
    # Check numeric rounding properties
    assert result["segmentation"]["lesion_area_cm2"] == 12.42
    assert result["segmentation"]["boundary_quality_score"] == 0.83
    assert result["scoring"]["ALP"] == 34.7
    
    # Check explicitly defined constraint from slide
    assert result["scoring"]["cavitation_confidence"] == "radiographic-only"
    
    # Check pathology keys
    assert result["pathology_flags"]["consolidation"] is True
    assert result["pathology_flags"]["fibrosis"] is False

def test_component9_json_saving():
    """Ensure it correctly outputs to a file without throwing exceptions."""
    result = generate_structured_json(
        patient_id="test",
        modality="CXR",
        scanner_domain="test",
        n_distinct_lesions=1,
        lesion_area_cm2=1.0,
        expert_routing={"E1": 1.0},
        boundary_quality_score=1.0,
        fp_probability=0.0,
        alp=10.0,
        cavity_flag=0,
        timika_score=10.0,
        severity="mild",
        pathology_flags={"cavitation": False}
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "output.json")
        save_structured_json(result, filepath)
        
        # Verify read file content matches
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        assert data["patient_id"] == "test"
        assert data["scoring"]["cavitation_confidence"] == "radiographic-only"
