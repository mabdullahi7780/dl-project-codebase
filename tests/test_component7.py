import torch

from src.components.component7_fp_auditor import estimate_fp_probability
from src.components.component7_verification import Component7BoundaryCritic


def test_component7_boundary_critic() -> None:
    model = Component7BoundaryCritic()
    x = torch.randn(2, 3, 224, 224)

    frozen = [param.requires_grad for param in next(iter(model.features.children())).parameters()]
    assert frozen and not any(frozen)
    assert next(model.head.parameters()).requires_grad

    out = model(x)
    assert out.shape == (2, 1)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_component7_fp_auditor() -> None:
    lesion_mask = torch.zeros(1, 256, 256)
    lesion_mask[:, 100:170, 120:180] = 1.0
    lung_mask = torch.ones(1, 256, 256)
    pathology_logits = torch.randn(18)

    result = estimate_fp_probability(
        lesion_mask,
        lung_mask,
        pathology_logits,
        class_names=(
            "atelectasis",
            "consolidation",
            "infiltration",
            "pneumothorax",
            "edema",
            "emphysema",
            "fibrosis",
            "effusion",
            "pneumonia",
            "pleural_thickening",
            "cardiomegaly",
            "nodule",
            "mass",
            "hernia",
            "lung_lesion",
            "fracture",
            "lung_opacity",
            "enlarged_cardiomediastinum",
        ),
    )

    assert 0.0 <= result.fp_probability <= 1.0
    assert isinstance(result.suspicious_probability, float)
