# influence_maximization_project/main.py
"""Main script for running the influence maximization experiments."""

import os
import matplotlib.pyplot as plt
import networkx as nx
from config import (
    GRAPH_TYPE, NUM_NODES, ERDOS_P, BARABASI_M, EDGELIST_PATH,
    SEED_SIZES, NUM_SIMULATIONS, RANDOM_SEED, IC_PROPAGATION_PROB, OUTPUT_DIR
)
from graph_loader import (
    generate_erdos_renyi, generate_barabasi_albert, load_from_edgelist, graph_statistics
)
from diffusion.independent_cascade import simulate_ic
from diffusion.linear_threshold import simulate_lt
from evaluation.experiments import run_experiments
from utils.random_utils import set_random_seed

def load_graph() -> nx.Graph:
    """Load or generate the graph based on config."""
    if GRAPH_TYPE == 'erdos_renyi':
        return generate_erdos_renyi(NUM_NODES, ERDOS_P)
    elif GRAPH_TYPE == 'barabasi_albert':
        return generate_barabasi_albert(NUM_NODES, BARABASI_M)
    elif GRAPH_TYPE == 'edgelist':
        return load_from_edgelist(EDGELIST_PATH)
    else:
        raise ValueError("Invalid graph type.")

def plot_results(results: dict, output_dir: str) -> None:
    """Generate plots for coverage and runtime."""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in results:
        # Coverage vs Seed Size
        plt.figure()
        for method_name in results[model_name]:
            data = results[model_name][method_name]['spreads']
            ks = [d[0] for d in data]
            spreads = [d[1] for d in data]
            plt.plot(ks, spreads, label=method_name)
        plt.xlabel('Seed Size')
        plt.ylabel('Expected Spread')
        plt.title(f'{model_name} - Coverage vs Seed Size')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{model_name}_coverage.png'))
        plt.close()
        
        # Runtime vs Seed Size
        plt.figure()
        for method_name in results[model_name]:
            data = results[model_name][method_name]['runtimes']
            ks = [d[0] for d in data]
            runtimes = [d[1] for d in data]
            plt.plot(ks, runtimes, label=method_name)
        plt.xlabel('Seed Size')
        plt.ylabel('Runtime (s)')
        plt.title(f'{model_name} - Runtime vs Seed Size')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{model_name}_runtime.png'))
        plt.close()

def print_summary_table(results: dict) -> None:
    """Print a summary table to console."""
    print("Summary Table:")
    for model_name in results:
        print(f"\nModel: {model_name}")
        print("Seed Size | Method | Spread (mean ± std) | Runtime (s)")
        for k in SEED_SIZES:
            for method_name in results[model_name]:
                spread_data = next(d for d in results[model_name][method_name]['spreads'] if d[0] == k)
                runtime_data = next(d for d in results[model_name][method_name]['runtimes'] if d[0] == k)
                print(f"{k:9} | {method_name:6} | {spread_data[1]:.2f} ± {spread_data[2]:.2f} | {runtime_data[1]:.2f}")

if __name__ == "__main__":
    set_random_seed(RANDOM_SEED)
    
    g = load_graph()
    stats = graph_statistics(g)
    print("Graph Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    model_functions = {
        'IC': simulate_ic,
        'LT': simulate_lt
    }
    
    model_params = {
        'IC': {'p': IC_PROPAGATION_PROB},
        'LT': {}
    }
    
    results = run_experiments(
        g, SEED_SIZES, NUM_SIMULATIONS, RANDOM_SEED, model_functions, model_params
    )
    
    plot_results(results, OUTPUT_DIR)
    print_summary_table(results)