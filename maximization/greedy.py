# influence_maximization_project/maximization/greedy.py
"""Implementation of the Greedy algorithm for Influence Maximization."""

from typing import List, Callable, Tuple
import networkx as nx
from evaluation.monte_carlo import estimate_influence

def greedy_im(
    g: nx.Graph,
    k: int,
    model_function: Callable,
    num_simulations: int,
    **model_params
) -> Tuple[List[int], List[float]]:
    """Run the greedy algorithm to select k seeds.

    Uses Monte Carlo simulations to estimate marginal gains.
    Based on submodularity of the influence function.

    Args:
        g: The input graph.
        k: Number of seeds to select.
        model_function: The diffusion model simulation function.
        num_simulations: Number of Monte Carlo simulations.
        **model_params: Additional parameters for the model.

    Returns:
        A tuple of (seed set, list of expected spreads after each addition).
    """
    seeds: List[int] = []
    spread_progression: List[float] = [0.0]
    current_spread = 0.0
    
    for _ in range(k):
        candidates = set(g.nodes()) - set(seeds)
        marginal_gains = {}
        
        for cand in candidates:
            temp_seeds = seeds + [cand]
            mean_spread, _, _ = estimate_influence(
                model_function, g, temp_seeds, num_simulations, **model_params
            )
            marginal_gains[cand] = mean_spread - current_spread
        
        if not marginal_gains:
            break
        
        best_cand = max(marginal_gains, key=marginal_gains.get)
        print(f"  Added seed {best_cand:4}  | current spread est: {current_spread:.1f}", flush=True)
        seeds.append(best_cand)
        current_spread += marginal_gains[best_cand]
        spread_progression.append(current_spread)
    
    return seeds, spread_progression