from __future__ import annotations

import torch

from src.components.baseline_lesion_proposer import BaselineLesionProposer


def test_baseline_lesion_proposer_returns_expected_shapes() -> None:
    proposer = BaselineLesionProposer()
    x_224 = torch.zeros((2, 1, 224, 224))
    x_224[:, :, 80:140, 90:150] = 512.0
    features = torch.rand((2, 1024, 7, 7))
    pathology_logits = torch.zeros((2, 18))
    pathology_logits[:, 1] = 2.5
    lung_mask_256 = torch.ones((2, 1, 256, 256))

    output = proposer.propose(
        x_224=x_224,
        features_7x7=features,
        pathology_logits=pathology_logits,
        lung_mask_256=lung_mask_256,
        classifier_weight=torch.rand((18, 1024)),
    )

    assert tuple(output.lesion_mask_coarse_256.shape) == (2, 1, 256, 256)
    assert tuple(output.confidence_map_256.shape) == (2, 1, 256, 256)
    assert len(output.selected_classes) == 2
