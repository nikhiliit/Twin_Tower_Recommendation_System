"""Global seed setter for reproducibility.

Provides a single function to seed all sources of randomness used
in the training pipeline: Python, NumPy, PyTorch (CPU + CUDA), and
CUDA deterministic backends.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Seeds Python's ``random`` module, NumPy, and PyTorch (CPU and
    CUDA). Also enables deterministic cuDNN algorithms.

    Args:
        seed: Integer seed value. Must be non-negative.

    Raises:
        ValueError: If *seed* is negative.

    Example:
        >>> set_global_seed(42)
    """
    if seed < 0:
        raise ValueError(
            f"Seed must be non-negative, got {seed}."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic algorithms where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 1.8+ deterministic flag.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Not available in older PyTorch.

    logger.info("Global seed set to %d.", seed)
