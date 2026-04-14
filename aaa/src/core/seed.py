from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 1337) -> None:
    """Seed python, numpy, and torch for reproducible runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
