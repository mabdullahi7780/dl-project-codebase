from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


TXV_CLASS_NAMES: tuple[str, ...] = (
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
)


def _has_xrv() -> bool:
    return importlib.util.find_spec("torchxrayvision") is not None


class MockTXVDenseNet(nn.Module):
    """Mock backbone matching the TorchXRayVision DenseNet output signature."""
    
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(1024, 18)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        # Expected output shape: [B, 1024, 7, 7]
        return torch.zeros((x.shape[0], 1024, 7, 7), device=x.device)


@dataclass(slots=True)
class TXVForwardOutput:
    features_7x7: torch.Tensor
    pooled_features: torch.Tensor
    pathology_logits: torch.Tensor
    domain_ctx: torch.Tensor
    classifier_weight: torch.Tensor | None
    class_names: tuple[str, ...]


class Component2SoftDomainContext(nn.Module):
    """
    Component 2: Soft Domain Context
    Frozen TorchXRayVision DenseNet121 + Trainable Domain Routing Head
    """

    def __init__(self, backend: str = "auto", weights: str = "densenet121-res224-all") -> None:
        super().__init__()
        self.active_backend = backend
        if backend == "auto":
            self.active_backend = "xrv" if _has_xrv() else "mock"
        elif backend == "xrv" and not _has_xrv():
            raise ImportError(
                "torchxrayvision is not installed. Install it or set backend to 'mock'."
            )

        if self.active_backend == "xrv":
            import torchxrayvision as xrv  # type: ignore
            self.txv_model = xrv.models.DenseNet(weights=weights)
        else:
            self.txv_model = MockTXVDenseNet()

        # Freeze the TXV backbone and pathology head
        for param in self.txv_model.parameters():
            param.requires_grad = False

        # Domain Routing Head
        self.domain_routing_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256)
        )

        # Fix 1: Binary TB classification head — trainable, separate from routing head.
        # Weight vector (shape [1024]) doubles as the CAM projection for the lesion proposer.
        self.tb_head = nn.Linear(1024, 1)

    def get_classifier_weight(self) -> torch.Tensor | None:
        classifier = getattr(self.txv_model, "classifier", None)
        if isinstance(classifier, nn.Linear):
            return classifier.weight.detach()
        return None

    def tb_head_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.tb_head.state_dict().items()}

    def load_tb_head_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.tb_head.load_state_dict(state_dict, strict=True)

    def routing_head_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.domain_routing_head.state_dict().items()
        }

    def load_routing_head_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.domain_routing_head.load_state_dict(state_dict, strict=True)

    def load_trained_routing_head(self, checkpoint_path: str | Path) -> Path:
        """Load a saved Component 2 routing-head checkpoint."""

        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Component 2 routing-head checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        state_dict: dict[str, torch.Tensor] | None = None

        if isinstance(payload, dict):
            nested = payload.get("domain_routing_head")
            if isinstance(nested, dict) and any(isinstance(v, torch.Tensor) for v in nested.values()):
                state_dict = {str(k): v for k, v in nested.items() if isinstance(v, torch.Tensor)}
            else:
                prefixed = {
                    key[len("domain_routing_head.") :]: value
                    for key, value in payload.items()
                    if key.startswith("domain_routing_head.") and isinstance(value, torch.Tensor)
                }
                if prefixed:
                    state_dict = prefixed
                elif any(isinstance(v, torch.Tensor) for v in payload.values()):
                    state_dict = {str(k): v for k, v in payload.items() if isinstance(v, torch.Tensor)}

        if not state_dict:
            raise ValueError(f"Unrecognised Component 2 routing-head payload at {path}.")

        self.load_routing_head_state_dict(state_dict)

        # Restore TB head if the checkpoint includes it.
        tb_sd = payload.get("tb_head") if isinstance(payload, dict) else None
        if isinstance(tb_sd, dict) and any(isinstance(v, torch.Tensor) for v in tb_sd.values()):
            self.load_tb_head_state_dict({str(k): v for k, v in tb_sd.items() if isinstance(v, torch.Tensor)})

        return path

    def forward_features(self, x_224: torch.Tensor) -> TXVForwardOutput:
        if x_224.ndim != 4 or x_224.shape[1] != 1 or tuple(x_224.shape[2:]) != (224, 224):
            raise ValueError(f"Expected input [B, 1, 224, 224], got {tuple(x_224.shape)}")

        features = self.txv_model.features(x_224)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        pathology_logits = self.txv_model.classifier(pooled)

        domain_ctx_raw = self.domain_routing_head(pooled)
        domain_ctx = F.normalize(domain_ctx_raw, p=2, dim=1)

        return TXVForwardOutput(
            features_7x7=features,
            pooled_features=pooled,
            pathology_logits=pathology_logits,
            domain_ctx=domain_ctx,
            classifier_weight=self.get_classifier_weight(),
            class_names=TXV_CLASS_NAMES,
        )

    def forward_tb_logit(self, x_224: torch.Tensor) -> torch.Tensor:
        """Returns [B, 1] raw TB logit. Backbone is frozen; only tb_head runs in grad mode."""
        with torch.no_grad():
            features = self.txv_model.features(x_224)
            pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        return self.tb_head(pooled)

    def forward(self, x_224: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_224: [B, 1, 224, 224] TXV-normalised tensor in range [-1024, 1024]
        Returns:
            domain_ctx: [B, 256] L2-normalized domain embedding
            pathology_logits: [B, 18] Raw pathology predictions for FP Auditor
        """
        output = self.forward_features(x_224)
        return output.domain_ctx, output.pathology_logits


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Supervised Contrastive Loss for pulling same-domain features together 
    and pushing cross-domain features apart.
    """
    device = features.device
    batch_size = features.shape[0]

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # Create mask for same-class pairs
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # Mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # Compute log prob
    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

    # Mean of log-likelihood over positive pairs
    mask_sum = mask.sum(1)
    mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

    # Exclude elements with no positive pairs
    valid_mask = mask.sum(1) > 0
    loss = -mean_log_prob_pos[valid_mask]

    if loss.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss.mean()
