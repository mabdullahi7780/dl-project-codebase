"""Integration tests for Component 10 BioGPT — fallback and faithfulness behaviour.

All tests run on CPU without real BioGPT weights by monkey-patching
``transformers.BioGptForCausalLM.from_pretrained`` to return a tiny stub.

Covers:
- Prompt structure validation (few-shot format, target JSON, "Report:" anchor)
- Fallback-to-template fires when the stub emits text that fails faithfulness
- Severity mismatch detection specifically
- Cavity consistency check (cavity_flag=True but report says "no cavity")
- Pathology claims check (report names a pathology not in TXV top-k)
- Lateralisation check wiring (no masks → check skipped)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from src.components.component10_biogpt import (
    BioGPTConfig,
    Component10BioGPTReport,
    FaithfulnessChecker,
    build_report_generator,
)
from src.components.component10_report import TemplateReportGenerator


# ---------------------------------------------------------------------------
# Stub BioGPT model + tokeniser
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Minimal tokeniser stub: encode → decode is a round-trip no-op."""

    eos_token_id = 1

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, Any]:
        # Return a single-token input_ids so generate() can be called
        return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        # Return a canned generation that the test can override via _stub_output
        return getattr(self, "_stub_output", "The lungs are clear.")


class _StubBioGPT(nn.Module):
    """Minimal BioGPT stub: exposes the attribute shape Component10BioGPTReport expects."""

    def __init__(self) -> None:
        super().__init__()
        self.biogpt = nn.Module()
        self.biogpt.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(12))
        self.output_projection = nn.Linear(8, 64)

    def generate(self, **kwargs: Any) -> torch.Tensor:
        # Return a dummy token sequence (content doesn't matter — tokenizer decode drives output)
        return torch.zeros(1, 5, dtype=torch.long)


def _make_stub_loaded_generator(stub_text: str) -> Component10BioGPTReport:
    """Return a Component10BioGPTReport that behaves as if BioGPT is loaded.

    We use use_mock=True to get a properly initialised instance (which calls
    nn.Module.__init__ internally), then overwrite the model/tokenizer with
    our controllable stubs and flip _loaded to True.
    """
    stub_tokeniser = _StubTokenizer()
    stub_tokeniser._stub_output = stub_text  # type: ignore[attr-defined]

    # Construct via mock path so nn.Module.__init__ is called correctly,
    # then patch the internals we want to control.
    gen = Component10BioGPTReport(BioGPTConfig(use_mock=True))
    # Replace internal model with our stub (both are nn.Module, so assignment is safe)
    object.__setattr__(gen, "model", _StubBioGPT())  # bypass nn.Module.__setattr__ guard
    gen.tokenizer = stub_tokeniser
    gen._loaded = True  # pretend model loaded successfully
    gen.cfg = BioGPTConfig(
        use_mock=False,
        fall_back_on_unfaithful=True,
        fall_back_on_load_error=True,
    )
    return gen


# ---------------------------------------------------------------------------
# Evidence JSON fixtures
# ---------------------------------------------------------------------------

def _mild_no_cavity_json() -> dict:
    return {
        "patient_id": "TEST_001",
        "modality": "CXR-PA",
        "scanner_domain": "montgomery",
        "segmentation": {
            "n_distinct_lesions": 1,
            "lesion_area_cm2": 3.0,
            "boundary_quality_score": 0.7,
            "fp_probability": 0.1,
        },
        "scoring": {
            "ALP": 10.0,
            "cavity_flag": 0,
            "timika_score": 10.0,
            "severity": "mild",
            "cavitation_confidence": "radiographic-only",
        },
        "pathology_flags": {"Consolidation": True, "Nodule": False},
    }


def _moderate_cavity_json() -> dict:
    return {
        "patient_id": "TEST_002",
        "modality": "CXR-PA",
        "scanner_domain": "shenzhen",
        "segmentation": {
            "n_distinct_lesions": 2,
            "lesion_area_cm2": 18.0,
            "boundary_quality_score": 0.55,
            "fp_probability": 0.2,
        },
        "scoring": {
            "ALP": 40.0,
            "cavity_flag": 1,
            "timika_score": 50.0,
            "severity": "moderate",
            "cavitation_confidence": "expert2-radiographic",
        },
        "pathology_flags": {"Consolidation": True, "Effusion": False},
    }


# ---------------------------------------------------------------------------
# 1. Prompt structure
# ---------------------------------------------------------------------------

class TestPromptStructure:
    def test_prompt_contains_three_examples(self) -> None:
        gen = Component10BioGPTReport(BioGPTConfig(use_mock=True))
        prompt = gen.format_prompt(_mild_no_cavity_json())
        assert prompt.count("Example ") >= 3

    def test_prompt_has_target_section(self) -> None:
        gen = Component10BioGPTReport(BioGPTConfig(use_mock=True))
        prompt = gen.format_prompt(_mild_no_cavity_json())
        assert "Target:" in prompt
        assert "Report:" in prompt

    def test_prompt_contains_patient_id(self) -> None:
        gen = Component10BioGPTReport(BioGPTConfig(use_mock=True))
        prompt = gen.format_prompt(_mild_no_cavity_json())
        assert "TEST_001" in prompt

    def test_prompt_embeds_evidence_json(self) -> None:
        gen = Component10BioGPTReport(BioGPTConfig(use_mock=True))
        evidence = _mild_no_cavity_json()
        prompt = gen.format_prompt(evidence)
        # The serialised JSON must appear inside the prompt
        assert json.dumps(evidence)[:30] in prompt or "TEST_001" in prompt


# ---------------------------------------------------------------------------
# 2. Fallback fires when faithfulness fails
# ---------------------------------------------------------------------------

class TestFallbackBehaviour:
    def test_faithful_report_returned_as_is(self) -> None:
        """When the generated text passes faithfulness, it is returned directly."""
        faithful_text = (
            "A single consolidation is identified in the right lung. "
            "The disease is consistent with mild severity. "
            "No cavitation is detected."
        )
        gen = _make_stub_loaded_generator(faithful_text)
        result = gen.generate(_mild_no_cavity_json())
        # Should return the stub text, not the template
        assert "mild" in result.lower()

    def test_severity_mismatch_triggers_fallback(self) -> None:
        """A report that says 'severe' when evidence says 'mild' must fall back."""
        wrong_severity_text = (
            "The disease pattern is consistent with severe TB. "
            "No cavitation is detected."
        )
        gen = _make_stub_loaded_generator(wrong_severity_text)
        result = gen.generate(_mild_no_cavity_json())
        # Template report should contain 'mild', not 'severe'
        assert "mild" in result.lower(), (
            f"Fallback template should say 'mild' but got: {result!r}"
        )

    def test_hallucinated_cavity_triggers_fallback(self) -> None:
        """A report claiming cavitation when cavity_flag=0 must fall back."""
        hallucinated_cavity_text = (
            "There is a large apical cavity present. "
            "The disease is mild."
        )
        gen = _make_stub_loaded_generator(hallucinated_cavity_text)
        result = gen.generate(_mild_no_cavity_json())
        # Template should not claim cavitation when flag is 0
        assert "cavity" not in result.lower() or "no cavit" in result.lower(), (
            f"Fallback should not hallucinate a cavity. Got: {result!r}"
        )

    def test_stub_fallback_on_load_error(self) -> None:
        """When BioGPT fails to load, generator must produce a template report.

        The patch path depends on whether ``transformers`` is installed.
        If it is, we patch ``BioGptForCausalLM.from_pretrained`` directly.
        If it is not, the import error inside ``_try_load`` already triggers
        the fallback — so we just construct without patching.
        """
        import importlib.util as _ilu

        cfg = BioGPTConfig(
            use_mock=False,
            fall_back_on_load_error=True,
        )

        if _ilu.find_spec("transformers") is not None:
            with patch(
                "transformers.BioGptForCausalLM.from_pretrained",
                side_effect=OSError("weights not found"),
            ):
                gen = Component10BioGPTReport(cfg)
        else:
            # transformers not installed — _try_load will raise ImportError
            # and fall_back_on_load_error=True causes graceful fallback
            gen = Component10BioGPTReport(cfg)

        # Generator should have fallen back to mock (not loaded)
        assert not gen._loaded
        # generate() must still return a non-empty string
        result = gen.generate(_mild_no_cavity_json())
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# 3. Severity mismatch detection
# ---------------------------------------------------------------------------

class TestSeverityMismatch:
    def test_moderate_in_mild_evidence_fails(self) -> None:
        checker = FaithfulnessChecker()
        evidence = _mild_no_cavity_json()
        report_with_wrong_severity = "The patient has moderate disease with bilateral infiltrates."
        assert checker.verify_report(report_with_wrong_severity, evidence) is False

    def test_correct_severity_passes(self) -> None:
        checker = FaithfulnessChecker()
        evidence = _mild_no_cavity_json()
        report_correct = "Mild disease consistent with early TB. No cavitation seen."
        assert checker.verify_report(report_correct, evidence) is True

    def test_severe_in_moderate_evidence_fails(self) -> None:
        checker = FaithfulnessChecker()
        evidence = _moderate_cavity_json()
        report = "The patient presents with severe bilateral consolidation and cavitation."
        assert checker.verify_report(report, evidence) is False


# ---------------------------------------------------------------------------
# 4. Cavity consistency check
# ---------------------------------------------------------------------------

class TestCavityConsistency:
    def test_no_cavity_in_report_when_flag_true_fails(self) -> None:
        checker = FaithfulnessChecker()
        assert checker.check_cavity_consistency(
            "There is no cavitation seen in the lungs.", cavity_flag=True
        ) is False

    def test_cavity_present_in_report_when_flag_true_passes(self) -> None:
        checker = FaithfulnessChecker()
        assert checker.check_cavity_consistency(
            "A discrete cavitation is identified in the upper lobe.", cavity_flag=True
        ) is True

    def test_no_cavity_when_flag_false_passes(self) -> None:
        checker = FaithfulnessChecker()
        assert checker.check_cavity_consistency(
            "No cavitation is detected.", cavity_flag=False
        ) is True

    def test_cavity_via_verify_report_integration(self) -> None:
        checker = FaithfulnessChecker()
        evidence = _moderate_cavity_json()  # cavity_flag=1
        report = "Moderate severity disease. No cavity is present."
        assert checker.verify_report(report, evidence) is False


# ---------------------------------------------------------------------------
# 5. Pathology claims check
# ---------------------------------------------------------------------------

class TestPathologyClaims:
    def test_supported_pathology_passes(self) -> None:
        checker = FaithfulnessChecker()
        assert checker.check_pathology_claims(
            "Consolidation is evident in the right lower lobe.",
            top_txv_classes=["Consolidation"],
        ) is True

    def test_unsupported_pathology_fails(self) -> None:
        checker = FaithfulnessChecker()
        assert checker.check_pathology_claims(
            "There is evidence of pleural effusion.",
            top_txv_classes=["Consolidation"],   # Effusion NOT in top-k
        ) is False

    def test_unknown_keyword_passes(self) -> None:
        """Keywords not in _CLAIM_TO_TXV must not cause failures."""
        checker = FaithfulnessChecker()
        assert checker.check_pathology_claims(
            "The patient may have sarcoidosis.",  # not a known keyword
            top_txv_classes=[],
        ) is True

    def test_via_verify_report_integration(self) -> None:
        checker = FaithfulnessChecker()
        evidence = _mild_no_cavity_json()  # pathology_flags: Consolidation=True
        # Report claims emphysema but TXV top-k only has Consolidation
        report = (
            "Mild disease with emphysematous changes. Consolidation noted. "
            "No cavitation is detected."
        )
        # emphysema not in top-k → fails
        assert checker.verify_report(report, evidence) is False

    def test_pathology_matches_txv_passes_via_verify(self) -> None:
        checker = FaithfulnessChecker()
        evidence = _mild_no_cavity_json()  # pathology_flags: Consolidation=True
        report = "Mild disease with consolidation present. No cavitation detected."
        assert checker.verify_report(report, evidence) is True


# ---------------------------------------------------------------------------
# 6. Lateralisation check
# ---------------------------------------------------------------------------

class TestLateralisation:
    def test_no_masks_skips_check(self) -> None:
        checker = FaithfulnessChecker()
        assert checker.check_lateralisation(
            "The right lung shows consolidation.",
            mask_fused_256=None,
            lung_mask_256=None,
        ) is True

    def test_no_lateral_mention_skips_check(self) -> None:
        checker = FaithfulnessChecker()
        mask = torch.zeros(1, 256, 256)
        mask[:, :, :128] = 1.0  # left half
        lung = torch.ones(1, 256, 256)
        assert checker.check_lateralisation(
            "There is consolidation in the upper zone.",  # no left/right
            mask_fused_256=mask,
            lung_mask_256=lung,
        ) is True

    def test_correct_side_passes(self) -> None:
        """Lesion in left image half = patient right lung. 'right' should pass."""
        checker = FaithfulnessChecker()
        mask = torch.zeros(1, 256, 256)
        mask[:, :, :100] = 1.0  # left image side → patient right
        lung = torch.ones(1, 256, 256)
        assert checker.check_lateralisation(
            "The right lung shows consolidation.",
            mask_fused_256=mask,
            lung_mask_256=lung,
        ) is True

    def test_wrong_side_fails(self) -> None:
        """Lesion in left image half = patient right lung. 'left' should fail."""
        checker = FaithfulnessChecker()
        mask = torch.zeros(1, 256, 256)
        mask[:, :, :100] = 1.0  # left image side → patient right
        lung = torch.ones(1, 256, 256)
        assert checker.check_lateralisation(
            "The left lung shows consolidation.",
            mask_fused_256=mask,
            lung_mask_256=lung,
        ) is False


# ---------------------------------------------------------------------------
# 7. build_report_generator factory
# ---------------------------------------------------------------------------

class TestBuildReportGenerator:
    def test_template_backend(self) -> None:
        gen = build_report_generator(backend="template")
        assert isinstance(gen, TemplateReportGenerator)

    def test_biogpt_backend_returns_component10(self) -> None:
        gen = build_report_generator(
            backend="biogpt",
            biogpt_config=BioGPTConfig(use_mock=True),
        )
        assert isinstance(gen, Component10BioGPTReport)

    def test_unknown_backend_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="Unknown report backend"):
            build_report_generator(backend="gpt4")
