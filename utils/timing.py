# influence_maximization_project/utils/timing.py
"""Utilities for timing code execution."""

import time
from typing import Callable, Any

def time_function(func: Callable, *args: Any, **kwargs: Any) -> tuple:
    """Time the execution of a function.

    Args:
        func: The function to time.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        A tuple of (result, runtime).
    """
    start = time.time()
    result = func(*args, **kwargs)
    runtime = time.time() - start
    return result, runtime