import numpy as np
from scipy import ndimage
import torch
from typing import Mapping, Union, Any

def compute_timika_score(
    mask_refined: Union[np.ndarray, torch.Tensor],
    lung_mask: Union[np.ndarray, torch.Tensor],
    mask_e2: Union[np.ndarray, torch.Tensor, None] = None
) -> dict[str, Union[float, int, str]]:
    """
    Component 8: Compute Metrics - ALP + Cavity Flag + Timika Score
    100% deterministic NumPy. Zero GPU. Zero training.
    
    Args:
        mask_refined: Lesion mask array/tensor, shape [1, 1024, 1024] or [1024, 1024]
        lung_mask: Lung mask array/tensor, shape [1, 1024, 1024] or [1024, 1024]
        mask_e2: Expert 2 (Cavity) mask array/tensor, shape [1, 256, 256] or [256, 256]
        
    Returns:
        A dictionary containing the calculated metrics.
    """
    # Detach and move to CPU if tensor
    if isinstance(mask_refined, torch.Tensor):
        mask_refined = mask_refined.detach().cpu().numpy()
    if isinstance(lung_mask, torch.Tensor):
        lung_mask = lung_mask.detach().cpu().numpy()
    if isinstance(mask_e2, torch.Tensor):
        mask_e2 = mask_e2.detach().cpu().numpy()
        
    # Squeeze batch/channel dimension if present to match [0] indexing required by logic
    if mask_refined.ndim == 3 and mask_refined.shape[0] == 1:
        mask_refined = mask_refined[0]
    if lung_mask.ndim == 3 and lung_mask.shape[0] == 1:
        lung_mask = lung_mask[0]
    if mask_e2 is not None and mask_e2.ndim == 3 and mask_e2.shape[0] == 1:
        mask_e2 = mask_e2[0]

    # Convert to binary
    lesion_bin = (mask_refined > 0.5).astype(np.uint8)  # [1024, 1024]
    lung_bin = (lung_mask > 0.5).astype(np.uint8)       # [1024, 1024]
    # Cavity threshold raised to 0.85 — Expert 2 sigmoid outputs fire broadly
    # on lightly-trained weights; only genuinely high-confidence regions should
    # trigger a cavity diagnosis (prevents false-positive Timika inflation).
    cavity_bin = (
        np.zeros((256, 256), dtype=np.uint8)
        if mask_e2 is None
        else (mask_e2 > 0.85).astype(np.uint8)
    )

    # ALP (Affected Lung Percentage)
    lesion_in_lung = lesion_bin * lung_bin

    lung_area = lung_bin.sum()
    if lung_area == 0:
        ALP = 0.0
    else:
        ALP = (lesion_in_lung.sum() / lung_area) * 100.0  # float, 0-100

    # Cavitation flag — min size raised to 200px (≈28×28) to avoid small noise
    # blobs triggering a cavity flag and inflating the Timika score by +40.
    labeled, n = ndimage.label(cavity_bin)

    cavity_flag = 0
    if n > 0:
        sizes = ndimage.sum(cavity_bin, labeled, range(1, n + 1))
        if np.isscalar(sizes):
            sizes = [sizes]
        if any(s > 200 for s in sizes):
            cavity_flag = 1
            
    # Timika Score
    timika_score = ALP + 40 * cavity_flag  # scalar, range 0-140
    
    # Severity classification
    severity = 'severe' if ALP > 50 else 'moderate' if ALP > 25 else 'mild'

    return {
        "ALP": float(ALP),
        "cavity_flag": int(cavity_flag),
        "timika_score": float(timika_score),
        "severity": severity
    }
