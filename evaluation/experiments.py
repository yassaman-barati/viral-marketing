# influence_maximization_project/evaluation/experiments.py
"""Experimental framework for comparing influence maximization methods."""

import time
from typing import Dict, List, Callable, Any
import networkx as nx
from maximization.greedy import greedy_im
from maximization.heuristics import high_degree, page_rank, random_selection
from .monte_carlo import estimate_influence

def run_experiments(
    g: nx.Graph,
    seed_sizes: List[int],
    num_simulations: int,
    random_seed: int,
    model_functions: Dict[str, Callable],
    model_params: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, List]]]:
    """Run experiments comparing different methods and models.

    Args:
        g: The input graph.
        seed_sizes: List of seed set sizes.
        num_simulations: Number of Monte Carlo simulations.
        random_seed: Random seed.
        model_functions: Dictionary of model names to simulation functions.
        model_params: Dictionary of model names to their parameters.

    Returns:
        A nested dictionary with results: model -> method -> 'spreads'/'runtimes' -> list of (k, value, std for spreads).
    """
    methods = {
        'greedy': lambda g, k, model_func, params: greedy_im(g, k, model_func, num_simulations, **params)[0],
        'degree': high_degree,
        'pagerank': page_rank,
        'random': lambda g, k: random_selection(g, k, random_seed)
    }
    
    results: Dict[str, Dict[str, Dict[str, List]]] = {}
    
    for model_name, model_func in model_functions.items():
        results[model_name] = {}
        params = model_params.get(model_name, {})
        
        for method_name, method_func in methods.items():
            results[model_name][method_name] = {'spreads': [], 'runtimes': []}
            
            for k in seed_sizes:
                print(f"[{model_name}] {method_name:8}  k={k:2} → selecting seeds...", flush=True)
                start_time = time.time()
                
                if method_name == 'greedy':
                    seeds = method_func(g, k, model_func, params)
                elif method_name == 'random':
                    seeds = method_func(g, k)
                else:
                    seeds = method_func(g, k)
                
                runtime_select = time.time() - start_time
                
                mean_spread, std, runtime_eval = estimate_influence(
                    model_func, g, seeds, num_simulations, seed=random_seed, **params
                )
                
                total_runtime = runtime_select + runtime_eval
                
                results[model_name][method_name]['spreads'].append((k, mean_spread, std))
                results[model_name][method_name]['runtimes'].append((k, total_runtime))
    
    return results