from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ReportGenerator(Protocol):
    def generate(self, evidence_json: dict[str, Any]) -> str:
        ...


@dataclass(slots=True)
class TemplateReportGenerator:
    """Deterministic baseline report generator.

    This generator intentionally ignores MoE/routing details even if they are
    present in the evidence JSON. The baseline report should describe findings,
    not internal expert allocation machinery.
    """

    fp_caveat_threshold: float = 0.5

    def generate(self, evidence_json: dict[str, Any]) -> str:
        segmentation = evidence_json.get("segmentation", {})
        scoring = evidence_json.get("scoring", {})
        modality = evidence_json.get("modality", "CXR")
        lesion_count = int(segmentation.get("n_distinct_lesions", 0) or 0)
        lesion_area_cm2 = float(segmentation.get("lesion_area_cm2", 0.0) or 0.0)
        fp_probability = float(segmentation.get("fp_probability", 0.0) or 0.0)
        alp = float(scoring.get("ALP", 0.0) or 0.0)
        severity = str(scoring.get("severity", "undetermined"))
        cavity_flag = int(scoring.get("cavity_flag", 0) or 0)
        cavitation_confidence = str(
            scoring.get("cavitation_confidence", "not-assessed-baseline")
        )

        lines: list[str] = [f"{modality}: baseline CXR analysis."]

        if lesion_count <= 0:
            lines.append("No suspicious focal lung abnormality was isolated by the baseline lesion pipeline.")
        elif lesion_count == 1:
            lines.append(
                f"One suspicious lesion region was identified, covering approximately {lesion_area_cm2:.2f} cm2."
            )
        else:
            lines.append(
                f"{lesion_count} suspicious lesion regions were identified, covering approximately {lesion_area_cm2:.2f} cm2 in total."
            )

        lines.append(f"Affected lung percentage is {alp:.1f}%, corresponding to {severity} estimated severity.")

        if cavitation_confidence == "not-assessed-baseline":
            lines.append("Cavitation was not assessed in this baseline configuration.")
        elif cavity_flag == 1:
            lines.append("Radiographic cavitation is flagged by the current scoring path.")
        else:
            lines.append("No radiographic cavitation flag is raised by the current scoring path.")

        if fp_probability >= self.fp_caveat_threshold:
            lines.append(
                "Caution: the false-positive auditor score is elevated, so these findings should be reviewed manually."
            )

        return " ".join(lines)


def generate_baseline_report(evidence_json: dict[str, Any]) -> str:
    return TemplateReportGenerator().generate(evidence_json)
