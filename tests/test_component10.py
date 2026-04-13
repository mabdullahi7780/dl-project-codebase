import pytest
import json
from src.components.component10_biogpt import Component10ReportGenerator, FaithfulnessChecker

def test_model_freezing():
    """Ensure BioGPT-Large layers 1-10 are frozen, and 11-12+LM head are trainable."""
    generator = Component10ReportGenerator(use_mock=True)
    
    frozen_params = sum(p.numel() for p in generator.model.biogpt.layers[:10].parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in generator.model.biogpt.layers[10:].parameters() if p.requires_grad)
    lm_head_trainable = sum(p.numel() for p in generator.model.output_projection.parameters() if p.requires_grad)
    
    assert frozen_params > 0, "Lower layers must be frozen."
    assert trainable_params > 0, "Layers 11-12 must be trainable."
    assert lm_head_trainable > 0, "LM head must be trainable."

def test_few_shot_prompting():
    """Verify that the generated prompt includes the 3 required few-shot examples + correctly serialized json."""
    generator = Component10ReportGenerator(use_mock=True)
    evidence_json = {
        "patient_id": "TEST_001",
        "segmentation": {"n_distinct_lesions": 2, "lesion_area_cm2": 15.0},
        "scoring": {"ALP": 22.5, "cavity_flag": 1, "severity": "moderate"}
    }
    
    prompt = generator.format_prompt(evidence_json)
    
    # Needs 3 examples from Shenzhen/TBX11K in the prompt
    assert prompt.count("Input: {") == 4 # 3 examples + 1 actual
    assert "Example 1:" in prompt
    assert "Example 3:" in prompt
    assert "TEST_001" in prompt
    assert "Report:" in prompt  # The target instruction

def test_faithfulness_checker():
    """Verify ScispaCy NER logic strictly enforces consistency between claims and the input JSON."""
    checker = FaithfulnessChecker()
    evidence_json = {
        "scoring": {
            "severity": "mild",
            "cavity_flag": 0
        },
        "pathology_flags": {
            "cavitation": False
        }
    }
    
    # 1. Truthful Report
    faithful_report = "The lungs show a mild disease pattern. There is no cavitation visible."
    assert checker.verify_report(faithful_report, evidence_json) is True
    
    # 2. Hallucinated Severity
    hallucinated_severity = "The lungs show a moderate disease pattern."
    assert checker.verify_report(hallucinated_severity, evidence_json) is False
    
    # 3. Hallucinated Cavity (contradicts cavity_flag = 0)
    hallucinated_cavity = "The lungs show a mild disease pattern, with a 5cm apical cavity."
    assert checker.verify_report(hallucinated_cavity, evidence_json) is False
