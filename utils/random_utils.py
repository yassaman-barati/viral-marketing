# influence_maximization_project/utils/random_utils.py
"""Utilities for managing randomness."""

import random
import numpy as np

def set_random_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)