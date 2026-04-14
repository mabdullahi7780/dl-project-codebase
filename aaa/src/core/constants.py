from __future__ import annotations

DATASET_ID_ALIASES: dict[str, str] = {
    "montgomery": "montgomery",
    "mont": "montgomery",
    "mc": "montgomery",
    "shenzhen": "shenzhen",
    "sz": "shenzhen",
    "tbx11k": "tbx11k",
    "tbx": "tbx11k",
    "nih": "nih_cxr14",
    "nih14": "nih_cxr14",
    "nih_cxr14": "nih_cxr14",
    "chestxray14": "nih_cxr14",
    "chest_xray14": "nih_cxr14",
}

DATASET_BIT_DEPTH_DIVISORS: dict[str, float] = {
    "montgomery": 4095.0,
    "shenzhen": 255.0,
    "tbx11k": 255.0,
    "nih_cxr14": 255.0,
}

DEFAULT_CLAHE_BY_DATASET: dict[str, bool] = {
    "montgomery": True,
    "shenzhen": True,
    "tbx11k": False,
    "nih_cxr14": False,
}

MANDATORY_CLAHE_DATASETS: set[str] = {"montgomery"}

MIN_IMAGE_EDGE_PX = 512
ASPECT_RATIO_RANGE = (0.7, 1.4)

X1024_SIZE = 1024
X224_SIZE = 224

VALID_PA_VIEWS = {"PA", "POSTEROANTERIOR"}
