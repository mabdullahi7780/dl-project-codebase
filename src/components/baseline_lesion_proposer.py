from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.components.component2_txv import TXV_CLASS_NAMES
from src.utils.morphology import otsu_threshold, postprocess_binary_mask


SUSPICIOUS_CLASS_SET = {
    "consolidation",
    "infiltration",
    "fibrosis",
    "pleural_thickening",
    "effusion",
    "mass",
    "nodule",
    "pneumonia",
}


@dataclass(slots=True)
class BaselineLesionProposal:
    lesion_mask_coarse_256: torch.Tensor
    confidence_map_256: torch.Tensor
    selected_classes: list[list[str]]
    bounding_boxes: list[tuple[int, int, int, int] | None]


@dataclass(slots=True)
class BaselineLesionProposerConfig:
    suspicious_class_threshold: float = 0.25
    fixed_binary_threshold: float | None = None
    min_region_px: int = 48
    opening_iters: int = 1
    closing_iters: int = 1
    fallback_blend: float = 0.35


class BaselineLesionProposer:
    """Baseline-only lesion source used when routing and experts are absent."""

    def __init__(
        self,
        config: BaselineLesionProposerConfig | None = None,
        *,
        class_names: tuple[str, ...] = TXV_CLASS_NAMES,
    ) -> None:
        self.config = config or BaselineLesionProposerConfig()
        self.class_names = class_names
        self.suspicious_indices = [
            index for index, name in enumerate(class_names) if name in SUSPICIOUS_CLASS_SET
        ]

    def _cam_map(
        self,
        features_7x7: torch.Tensor,
        pathology_logits: torch.Tensor,
        classifier_weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[list[str]]]:
        probs = torch.sigmoid(pathology_logits)
        batch = features_7x7.shape[0]
        cams: list[torch.Tensor] = []
        selected_names: list[list[str]] = []

        for index in range(batch):
            selected = [
                class_index
                for class_index in self.suspicious_indices
                if float(probs[index, class_index].item()) >= self.config.suspicious_class_threshold
            ]
            if not selected and self.suspicious_indices:
                best_idx = max(self.suspicious_indices, key=lambda class_idx: float(probs[index, class_idx].item()))
                selected = [best_idx]

            selected_names.append([self.class_names[class_index] for class_index in selected])

            if classifier_weight is not None:
                sample_maps = []
                for class_index in selected:
                    weight = classifier_weight[class_index].to(features_7x7.device).view(-1, 1, 1)
                    cam = torch.relu((features_7x7[index] * weight).sum(dim=0))
                    sample_maps.append(cam * probs[index, class_index])
                combined = torch.stack(sample_maps, dim=0).sum(dim=0) if sample_maps else torch.zeros((7, 7), device=features_7x7.device)
            else:
                combined = features_7x7[index].abs().mean(dim=0)
                if selected:
                    combined = combined * probs[index, selected].mean()

            cams.append(combined)

        return torch.stack(cams, dim=0).unsqueeze(1), selected_names

    def _image_fallback(self, x_224: torch.Tensor, lung_mask_256: torch.Tensor) -> torch.Tensor:
        image_01 = ((x_224 + 1024.0) / 2048.0).clamp(0.0, 1.0)
        image_256 = F.interpolate(image_01, size=(256, 256), mode="bilinear", align_corners=False)
        image_np = image_256.squeeze(1).detach().cpu().numpy()
        lung_np = (lung_mask_256.squeeze(1) > 0.5).detach().cpu().numpy()

        outputs = []
        for image, lung in zip(image_np, lung_np, strict=False):
            if lung.any():
                lung_values = image[lung]
                mean = float(lung_values.mean())
                std = float(lung_values.std()) + 1e-6
                suspicious = np.clip((image - mean) / std, 0.0, None)
            else:
                suspicious = np.zeros_like(image)

            max_value = float(suspicious.max())
            if max_value > 0.0:
                suspicious = suspicious / max_value
            suspicious *= lung.astype(np.float32)
            outputs.append(torch.from_numpy(suspicious).to(dtype=torch.float32))

        return torch.stack(outputs, dim=0).unsqueeze(1).to(x_224.device)

    def propose(
        self,
        *,
        x_224: torch.Tensor,
        features_7x7: torch.Tensor,
        pathology_logits: torch.Tensor,
        lung_mask_256: torch.Tensor,
        classifier_weight: torch.Tensor | None = None,
    ) -> BaselineLesionProposal:
        if x_224.ndim != 4 or tuple(x_224.shape[1:]) != (1, 224, 224):
            raise ValueError(f"Expected x_224 [B, 1, 224, 224], got {tuple(x_224.shape)}.")
        if features_7x7.ndim != 4 or tuple(features_7x7.shape[2:]) != (7, 7):
            raise ValueError(f"Expected features_7x7 [B, C, 7, 7], got {tuple(features_7x7.shape)}.")
        if pathology_logits.ndim != 2 or pathology_logits.shape[1] != len(self.class_names):
            raise ValueError(
                f"Expected pathology_logits [B, {len(self.class_names)}], got {tuple(pathology_logits.shape)}."
            )
        if lung_mask_256.ndim != 4 or tuple(lung_mask_256.shape[1:]) != (1, 256, 256):
            raise ValueError(f"Expected lung_mask_256 [B, 1, 256, 256], got {tuple(lung_mask_256.shape)}.")

        cam_map_7x7, selected_classes = self._cam_map(features_7x7, pathology_logits, classifier_weight)
        cam_map_256 = F.interpolate(cam_map_7x7, size=(256, 256), mode="bilinear", align_corners=False)
        cam_map_256 = cam_map_256.clamp_min(0.0)

        fallback = self._image_fallback(x_224, lung_mask_256)
        confidence_map = ((1.0 - self.config.fallback_blend) * cam_map_256) + (self.config.fallback_blend * fallback)

        masks: list[torch.Tensor] = []
        boxes: list[tuple[int, int, int, int] | None] = []

        for score_map, lung_mask in zip(confidence_map, lung_mask_256, strict=False):
            score_np = score_map.squeeze(0).detach().cpu().numpy()
            lung_np = (lung_mask.squeeze(0) > 0.5).detach().cpu().numpy()
            score_np = score_np * lung_np.astype(np.float32)

            max_value = float(score_np.max())
            if max_value <= 0.0:
                boxes.append(None)
                masks.append(torch.zeros_like(score_map.squeeze(0), dtype=torch.float32).cpu())
                continue
            if max_value > 0.0:
                score_np = score_np / max_value

            lung_values = score_np[lung_np]
            threshold = (
                self.config.fixed_binary_threshold
                if self.config.fixed_binary_threshold is not None
                else otsu_threshold(lung_values)
            )
            binary = score_np >= threshold
            cleaned = postprocess_binary_mask(
                binary,
                min_area=self.config.min_region_px,
                opening_iters=self.config.opening_iters,
                closing_iters=self.config.closing_iters,
            )
            cleaned &= lung_np

            ys, xs = np.where(cleaned)
            if len(xs) == 0:
                boxes.append(None)
            else:
                boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

            masks.append(torch.from_numpy(cleaned.astype(np.float32)))

        return BaselineLesionProposal(
            lesion_mask_coarse_256=torch.stack(masks, dim=0).unsqueeze(1).to(x_224.device),
            confidence_map_256=confidence_map.to(dtype=torch.float32),
            selected_classes=selected_classes,
            bounding_boxes=boxes,
        )
