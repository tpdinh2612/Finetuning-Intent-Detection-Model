# =========================================================
# Deterministic Seed Utility
# =========================================================
"""
Centralized seed-locking utility to guarantee reproducibility
across all stochastic components (Python, NumPy, PyTorch).
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Lock all random number generators to the given seed.

    This ensures deterministic behavior across:
    - Python's built-in random module
    - NumPy's random state
    - PyTorch CPU and all CUDA devices
    - cuDNN backend (deterministic mode)

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force cuDNN deterministic algorithms at the cost of slight performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure reproducible behavior for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)
