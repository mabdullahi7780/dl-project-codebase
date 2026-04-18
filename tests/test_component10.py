from src.components.component10_biogpt import (
    BioGPTConfig,
    Component10BioGPTReport,
    FaithfulnessChecker,
)


def test_model_freezing() -> None:
    generator = Component10BioGPTReport(BioGPTConfig(use_mock=True))

    frozen_params = sum(
        p.numel() for p in generator.model.biogpt.layers[:10].parameters() if not p.requires_grad
    )
    trainable_params = sum(
        p.numel() for p in generator.model.biogpt.layers[10:].parameters() if p.requires_grad
    )
    lm_head_trainable = sum(
        p.numel() for p in generator.model.output_projection.parameters() if p.requires_grad
    )

    assert frozen_params > 0
    assert trainable_params > 0
    assert lm_head_trainable > 0


def test_few_shot_prompting() -> None:
    generator = Component10BioGPTReport(BioGPTConfig(use_mock=True))
    evidence_json = {
        "patient_id": "TEST_001",
        "segmentation": {"n_distinct_lesions": 2, "lesion_area_cm2": 15.0},
        "scoring": {"ALP": 22.5, "cavity_flag": 1, "severity": "moderate"},
    }

    prompt = generator.format_prompt(evidence_json)
    assert prompt.count("Input: {") == 4
    assert "Example 1:" in prompt
    assert "Example 3:" in prompt
    assert "TEST_001" in prompt
    assert "Report:" in prompt


def test_faithfulness_checker() -> None:
    checker = FaithfulnessChecker()
    evidence_json = {
        "scoring": {
            "severity": "mild",
            "cavity_flag": 0,
        },
        "pathology_flags": {
            "cavitation": False,
        },
    }

    faithful_report = "The lungs show a mild disease pattern. There is no cavitation visible."
    assert checker.verify_report(faithful_report, evidence_json) is True

    hallucinated_severity = "The lungs show a moderate disease pattern."
    assert checker.verify_report(hallucinated_severity, evidence_json) is False

    hallucinated_cavity = "The lungs show a mild disease pattern, with a 5cm apical cavity."
    assert checker.verify_report(hallucinated_cavity, evidence_json) is False
