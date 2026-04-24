"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
