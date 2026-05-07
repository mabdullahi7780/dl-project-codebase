from __future__ import annotations

import numpy as np


try:
    from scipy import ndimage as ndi
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "The baseline morphology helpers require `scipy`. Install it from requirements.txt."
    ) from exc


def otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.5
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return min_value

    hist, bin_edges = np.histogram(values, bins=64, range=(min_value, max_value))
    hist = hist.astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_edges[:-1])
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    idx = int(np.nanargmax(sigma_b))
    return float(bin_edges[idx])


def connected_component_stats(mask: np.ndarray) -> list[int]:
    labeled, count = ndi.label(mask.astype(bool))
    if count == 0:
        return []
    sizes = ndi.sum(mask.astype(np.uint8), labeled, index=np.arange(1, count + 1))
    return [int(size) for size in np.atleast_1d(sizes).tolist()]


def remove_small_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    labeled, count = ndi.label(mask.astype(bool))
    if count == 0:
        return mask.astype(bool)

    keep = np.zeros_like(mask, dtype=bool)
    sizes = np.atleast_1d(ndi.sum(mask.astype(np.uint8), labeled, index=np.arange(1, count + 1)))
    for component_id, size in enumerate(sizes, start=1):
        if int(size) >= int(min_area):
            keep |= labeled == component_id
    return keep


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask.astype(bool))


def binary_erode(mask: np.ndarray, *, iterations: int = 1) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    return ndi.binary_erosion(mask.astype(bool), iterations=iterations)


def postprocess_binary_mask(
    mask: np.ndarray,
    *,
    min_area: int,
    opening_iters: int = 1,
    closing_iters: int = 1,
) -> np.ndarray:
    cleaned = mask.astype(bool)
    if opening_iters > 0:
        cleaned = ndi.binary_opening(cleaned, iterations=opening_iters)
    if closing_iters > 0:
        cleaned = ndi.binary_closing(cleaned, iterations=closing_iters)
    cleaned = fill_binary_holes(cleaned)
    cleaned = remove_small_components(cleaned, min_area=min_area)
    return cleaned.astype(bool)


def adaptive_lesion_threshold(
    prob_map: np.ndarray,
    lung_mask: np.ndarray,
    *,
    floor: float = 0.5,
    ceil: float = 0.8,
    min_pixels: int = 64,
) -> float:
    """Per-image adaptive binarization threshold for the fused lesion probability.

    Otsu inside the lung region, clipped to [floor, ceil].  floor=0.5/ceil=0.8
    prevents Otsu from firing on more than 50 % of in-lung pixels on weak/diffuse
    CAMs (healthy images), which was causing NIH ALP = 34.8 % before this fix.
    When the fused map is near-empty (all probabilities below floor) the threshold
    is pushed to ceil so under-fired predictions don't survive.
    """
    p = np.asarray(prob_map, dtype=np.float32)
    p = p[..., 0] if p.ndim == 3 and p.shape[0] == 1 else p
    l = np.asarray(lung_mask) > 0.5
    l = l[..., 0] if l.ndim == 3 and l.shape[0] == 1 else l
    if not l.any():
        return 0.5
    in_lung = p[l]
    if in_lung.size < min_pixels or float(in_lung.max()) < floor:
        return ceil
    t = otsu_threshold(in_lung)
    return float(np.clip(t, floor, ceil))
