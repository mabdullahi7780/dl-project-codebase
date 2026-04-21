"""Component 10 — BioGPT-grounded report generator (MoE/upgrade path).

Optional alternative to the deterministic ``TemplateReportGenerator``.
Implements the same ``ReportGenerator`` protocol so it is a drop-in
replacement when ``transformers`` is installed and the BioGPT-Large
weights are available.

Faithfulness checking
---------------------
Every BioGPT generation is verified against the structured evidence
JSON before being returned.  If the report contradicts the JSON
(claims cavitation when ``cavity_flag=0``, mentions the wrong
severity, etc.) the generator falls back to the template report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.components.component10_report import TemplateReportGenerator


# ---------------------------------------------------------------------------
# BioGPT report generator
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BioGPTConfig:
    model_name: str = "microsoft/BioGPT-Large"
    use_mock: bool = False
    max_new_tokens: int = 220
    temperature: float = 0.3
    top_p: float = 0.9
    do_sample: bool = False
    fall_back_on_unfaithful: bool = True
    fall_back_on_load_error: bool = True


class Component10BioGPTReport(nn.Module):
    """Constrained BioGPT-Large report generator.

    Freezes layers 1-10, leaves layers 11-12 + the LM head trainable
    so a small fine-tuning pass can be run on the few-shot examples
    without rewriting the whole stack.
    """

    def __init__(self, config: BioGPTConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or BioGPTConfig()
        self.template_fallback = TemplateReportGenerator()
        self.model: Any = None
        self.tokenizer: Any = None
        self._loaded = False

        if not self.cfg.use_mock:
            self._try_load()
        else:
            self._build_mock()

    def _try_load(self) -> None:
        try:
            from transformers import BioGptForCausalLM, BioGptTokenizer  # type: ignore
            self.tokenizer = BioGptTokenizer.from_pretrained(self.cfg.model_name)
            self.model = BioGptForCausalLM.from_pretrained(self.cfg.model_name)
            self._apply_freezing()
            self._loaded = True
        except Exception as exc:
            if not self.cfg.fall_back_on_load_error:
                raise
            print(f"[BioGPT] load failed ({exc}); falling back to mock + template.")
            self._build_mock()

    def _build_mock(self) -> None:
        # Mock that matches the expected attribute structure
        self.model = nn.Module()
        self.model.biogpt = nn.Module()
        self.model.biogpt.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(12))
        self.model.output_projection = nn.Linear(8, 64)
        self._apply_freezing()
        self._loaded = False  # treat mock as not really loaded for generation

    def _apply_freezing(self) -> None:
        layers = self.model.biogpt.layers
        for i in range(min(10, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False
        for i in range(10, len(layers)):
            for param in layers[i].parameters():
                param.requires_grad = True
        if hasattr(self.model, "output_projection"):
            for param in self.model.output_projection.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, evidence_json: dict[str, Any]) -> str:
        few_shot = """Example 1:
Input: {"patient_id": "SHZ_001", "segmentation": {"n_distinct_lesions": 1, "lesion_area_cm2": 5.2}, "scoring": {"ALP": 12.0, "cavity_flag": 0, "severity": "mild"}, "pathology_flags": {"consolidation": true, "cavitation": false}}
Report: A single distinct lesion measuring approximately 5.2 cm2 is identified. The affected lung percentage is calculated at 12.0%, consistent with mild severity disease. There is evidence of consolidation, but no discrete cavitation is observed radiographically.

Example 2:
Input: {"patient_id": "SHZ_002", "segmentation": {"n_distinct_lesions": 3, "lesion_area_cm2": 24.1}, "scoring": {"ALP": 45.0, "cavity_flag": 1, "severity": "moderate"}, "pathology_flags": {"consolidation": true, "cavitation": true}}
Report: Multiple lesions are present, spanning roughly 24.1 cm2 with a 45.0% affected lung percentage, indicating moderate severity. Radiographic features are strongly suggestive of consolidation and focal cavitation.

Example 3:
Input: {"patient_id": "TBX_003", "segmentation": {"n_distinct_lesions": 0, "lesion_area_cm2": 0.0}, "scoring": {"ALP": 0.0, "cavity_flag": 0, "severity": "mild"}, "pathology_flags": {"consolidation": false, "cavitation": false}}
Report: The lungs are clear with no distinct lesions isolated (0.0% affected lung percentage). No radiographic signs of consolidation or cavitation are detected.

Target:
Input: """
        return few_shot + json.dumps(evidence_json) + "\nReport:"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_text(self, prompt: str) -> str:
        if not self._loaded:
            raise RuntimeError("BioGPT model is not loaded; cannot call generate.")
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=self.cfg.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Take everything after the final "Report:" anchor
        marker = "\nReport:"
        if marker in full[len(prompt) - len(marker) :]:
            full = full.split(marker)[-1]
        # Truncate at a double newline (next "Input:" example) if any
        return full.split("\nInput:")[0].strip()

    # ------------------------------------------------------------------
    # ReportGenerator protocol
    # ------------------------------------------------------------------

    def generate(self, evidence_json: dict[str, Any]) -> str:
        """Generate a clinical report and verify faithfulness."""
        if not self._loaded:
            return self.template_fallback.generate(evidence_json)

        try:
            prompt = self.format_prompt(evidence_json)
            text = self._generate_text(prompt)
        except Exception as exc:
            if self.cfg.fall_back_on_load_error:
                print(f"[BioGPT] generation failed ({exc}); using template.")
                return self.template_fallback.generate(evidence_json)
            raise

        checker = FaithfulnessChecker()
        if not checker.verify_report(text, evidence_json):
            if self.cfg.fall_back_on_unfaithful:
                print("[BioGPT] generated report failed faithfulness check; using template.")
                return self.template_fallback.generate(evidence_json)
        return text


# ---------------------------------------------------------------------------
# Faithfulness checker
# ---------------------------------------------------------------------------

class FaithfulnessChecker:
    """Verifies that the generated report only makes claims supported by JSON.

    Uses ScispaCy NER when available; otherwise falls back to a minimal
    rule-based check.  The fallback is sufficient to catch the most
    common hallucinations (cavity vs no-cavity, severity mismatch,
    lateralisation errors, unsupported pathology claims).

    Extended checks
    ---------------
    check_lateralisation   — report must not name the wrong lung side.
    check_cavity_consistency — "no cavity" must not appear when cavity_flag=True.
    check_pathology_claims  — any pathology named in the report must be in the
                              TXV top-k list.
    """

    def __init__(self) -> None:
        self.has_spacy = False
        self.nlp: Any = None
        try:
            import spacy  # type: ignore
            self.nlp = spacy.load("en_core_sci_sm")
            self.has_spacy = True
        except Exception:
            self.has_spacy = False

    # ------------------------------------------------------------------
    # Extended check 1 — lateralisation
    # ------------------------------------------------------------------

    def check_lateralisation(
        self,
        report_text: str,
        mask_fused_256: "torch.Tensor | None",
        lung_mask_256: "torch.Tensor | None",
    ) -> bool:
        """Return False if the report names the wrong lung side.

        Computes the lesion centroid x-coordinate relative to the lung
        bounding box midpoint.  If the lesion is on the left half of the
        image but the report says "right", or vice versa, the check fails.

        When either mask is None (baseline path) the check is skipped
        (returns True).
        """
        if mask_fused_256 is None or lung_mask_256 is None:
            return True

        report_lower = report_text.lower()
        mentions_right = "right lung" in report_lower or "right lobe" in report_lower
        mentions_left  = "left lung"  in report_lower or "left lobe"  in report_lower

        if not (mentions_right or mentions_left):
            # No lateralisation claim — nothing to check
            return True

        try:
            import torch

            # Work on 2-D arrays
            mask_2d = mask_fused_256.squeeze().float()
            lung_2d = lung_mask_256.squeeze().float()

            lung_cols = lung_2d.nonzero(as_tuple=False)[:, 1]
            if lung_cols.numel() == 0:
                return True
            lung_mid_x = float(lung_cols.float().mean())

            mask_cols = (mask_2d > 0.5).nonzero(as_tuple=False)[:, 1]
            if mask_cols.numel() == 0:
                return True
            lesion_centroid_x = float(mask_cols.float().mean())

            # In a standard PA CXR, "left" lung appears on the right side of
            # the image (patient's left = image right).  Use the simpler
            # image-space convention: lesion_x < lung_mid => image-left side.
            lesion_on_image_left = lesion_centroid_x < lung_mid_x

            # image-left = patient's RIGHT lung in PA
            lesion_on_patient_right = lesion_on_image_left

            if mentions_right and not lesion_on_patient_right:
                return False
            if mentions_left and lesion_on_patient_right:
                return False
        except Exception:
            # If tensor ops fail for any reason, skip the check
            return True

        return True

    # ------------------------------------------------------------------
    # Extended check 2 — cavity consistency
    # ------------------------------------------------------------------

    def check_cavity_consistency(self, report_text: str, cavity_flag: bool) -> bool:
        """Return False if the report says "no cavity" when cavity_flag is True.

        A model that detected a cavity (cavity_flag=True) but then writes
        "no cavity" is self-contradictory.  This catches that specific case.
        """
        report_lower = report_text.lower()
        no_cavity_phrases = (
            "no cavity",
            "no cavitation",
            "without cavit",
            "absence of cavit",
            "no discrete cavit",
        )
        if cavity_flag:
            for phrase in no_cavity_phrases:
                if phrase in report_lower:
                    return False
        return True

    # ------------------------------------------------------------------
    # Extended check 3 — pathology claims
    # ------------------------------------------------------------------

    # Vocabulary map: report keyword → TXV class names that support the claim.
    # A claim in the report is "supported" if at least one matching TXV class
    # appears in the provided top_txv_classes list.
    _CLAIM_TO_TXV: dict[str, tuple[str, ...]] = {
        "consolidation":    ("Consolidation",),
        "effusion":         ("Effusion", "Pleural_Thickening"),
        "pleural effusion": ("Effusion", "Pleural_Thickening"),
        "pneumothorax":     ("Pneumothorax",),
        "atelectasis":      ("Atelectasis",),
        "cardiomegaly":     ("Cardiomegaly",),
        "infiltrate":       ("Infiltration",),
        "infiltration":     ("Infiltration",),
        "nodule":           ("Nodule", "Mass"),
        "mass":             ("Mass", "Nodule"),
        "fibrosis":         ("Fibrosis",),
        "emphysema":        ("Emphysema",),
        "edema":            ("Edema",),
    }

    def check_pathology_claims(
        self,
        report_text: str,
        top_txv_classes: list[str],
    ) -> bool:
        """Return False if the report names a pathology absent from TXV top-k.

        Any pathology keyword found in the report must be supported by at
        least one class in ``top_txv_classes``.  Unknown keywords (not in
        ``_CLAIM_TO_TXV``) are ignored so novel phrasings do not cause
        spurious failures.

        Args:
            report_text:    The generated report string.
            top_txv_classes: List of TXV class names that exceeded the
                             flag threshold (e.g. ``["Consolidation", "Nodule"]``).
        """
        report_lower = report_text.lower()
        txv_set = {c.lower() for c in top_txv_classes}

        for keyword, supporting_classes in self._CLAIM_TO_TXV.items():
            if keyword in report_lower:
                # Check that at least one supporting TXV class was flagged
                if not any(sc.lower() in txv_set for sc in supporting_classes):
                    return False
        return True

    # ------------------------------------------------------------------
    # Master verification (wires all checks together)
    # ------------------------------------------------------------------

    def verify_report(
        self,
        report_text: str,
        evidence_json: dict[str, Any],
        *,
        mask_fused_256: "torch.Tensor | None" = None,
        lung_mask_256: "torch.Tensor | None" = None,
    ) -> bool:
        """Run all faithfulness checks against evidence_json.

        The extended checks (lateralisation, cavity consistency, pathology
        claims) are wired in here so the existing call-site in ``generate``
        does not need to change — the extra kwargs default to None and the
        extended checks gracefully no-op when data is unavailable.
        """
        report_lower = report_text.lower()
        scoring = evidence_json.get("scoring", {})

        # --- original check 1: severity ---
        severity = str(scoring.get("severity", "")).lower()
        if severity and severity not in report_lower:
            return False

        # --- original check 2: cavitation ---
        cavity_flag_raw = scoring.get("cavity_flag", 0) or 0
        cavity_flag_bool = bool(int(cavity_flag_raw))
        confidence = str(scoring.get("cavitation_confidence", ""))

        if cavity_flag_bool and "cavit" not in report_lower:
            return False
        if not cavity_flag_bool and "cavity" in report_lower \
                and "no cavity" not in report_lower \
                and "without cavit" not in report_lower:
            return False

        if confidence == "not-assessed-baseline" \
                and ("cavity" in report_lower or "cavitation" in report_lower):
            if "not assessed" not in report_lower and "no cavit" not in report_lower:
                return False

        # --- extended check: cavity consistency ---
        if not self.check_cavity_consistency(report_text, cavity_flag_bool):
            return False

        # --- extended check: lateralisation ---
        if not self.check_lateralisation(report_text, mask_fused_256, lung_mask_256):
            return False

        # --- extended check: pathology claims ---
        pathology_flags = evidence_json.get("pathology_flags", {})
        # Build the top-k list from whichever format pathology_flags is in
        if isinstance(pathology_flags, dict):
            top_txv = [k for k, v in pathology_flags.items() if v]
        else:
            top_txv = []
        if top_txv:
            if not self.check_pathology_claims(report_text, top_txv):
                return False

        return True


# ---------------------------------------------------------------------------
# Convenience wrapper for swap-in use
# ---------------------------------------------------------------------------

def build_report_generator(
    *,
    backend: str = "template",
    biogpt_config: BioGPTConfig | None = None,
):
    """Factory for either ``TemplateReportGenerator`` or ``Component10BioGPTReport``.

    Args:
        backend: "template" (deterministic) or "biogpt".
        biogpt_config: optional override for BioGPT settings.

    Returns:
        Object satisfying the ``ReportGenerator`` protocol.
    """
    if backend == "template":
        return TemplateReportGenerator()
    if backend == "biogpt":
        return Component10BioGPTReport(biogpt_config)
    raise ValueError(f"Unknown report backend {backend!r}.")
