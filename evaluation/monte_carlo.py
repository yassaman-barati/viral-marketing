# influence_maximization_project/evaluation/monte_carlo.py
"""Monte Carlo estimation for influence spread."""

import time
import numpy as np
import networkx as nx
from typing import Callable, Tuple, List

def estimate_influence(
    model_function: Callable,
    g: nx.Graph,
    seeds: List[int],
    num_simulations: int,
    seed: int = None,
    **model_params
) -> Tuple[float, float, float]:
    """Estimate the expected influence spread using Monte Carlo simulations.

    Args:
        model_function: The diffusion model simulation function.
        g: The input graph.
        seeds: List of seed nodes.
        num_simulations: Number of simulations.
        seed: Random seed for reproducibility.
        **model_params: Additional parameters for the model.

    Returns:
        A tuple of (mean spread, standard deviation, runtime).
    """
    if seed is not None:
        np.random.seed(seed)
    
    spreads = []
    start_time = time.time()
    
    for _ in range(num_simulations):
        spread = model_function(g, seeds, **model_params)
        spreads.append(spread)
    
    runtime = time.time() - start_time
    mean = np.mean(spreads)
    std = np.std(spreads)
    
    return mean, std, runtime