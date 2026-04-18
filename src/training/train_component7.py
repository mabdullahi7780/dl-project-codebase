"""Legacy Component 7 smoke helpers.

This file remains as a lightweight compatibility entry point, but the real
training path lives in ``src.training.train_boundary_critic`` for the
ResNet18 boundary critic and in the inference-time heuristic FP auditor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.components.component7_fp_auditor import estimate_fp_probability
from src.components.component7_verification import Component7BoundaryCritic


def smoke_train_boundary_critic() -> None:
    model = Component7BoundaryCritic()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=2e-4,
    )
    criterion = nn.BCELoss()

    x = torch.randn(8, 3, 224, 224)
    y = torch.rand(8, 1)

    model.train()
    optimizer.zero_grad()
    boundary_score = model(x)
    loss = criterion(boundary_score, y)
    loss.backward()
    optimizer.step()

    print(f"Boundary Critic Loss: {loss.item()}")


def smoke_run_fp_auditor() -> None:
    lesion_mask = torch.zeros(1, 256, 256)
    lesion_mask[:, 96:160, 96:176] = 1.0
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
    print(f"FP Auditor Probability: {result.fp_probability}")


if __name__ == "__main__":
    smoke_train_boundary_critic()
    smoke_run_fp_auditor()
