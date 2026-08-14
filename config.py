"""Configuration file for the influence maximization project."""

import numpy as np

# Graph parameters
GRAPH_TYPE = 'barabasi_albert'  # Options: 'erdos_renyi', 'barabasi_albert', 'edgelist'
NUM_NODES = 1000
ERDOS_P = 0.01
BARABASI_M = 3
EDGELIST_PATH = 'path/to/edgelist.txt'  # Placeholder

# Diffusion model parameters
IC_PROPAGATION_PROB = 0.1
NUM_SIMULATIONS = 100

# Experiment parameters
SEED_SIZES = [1, 2, 5, 10, 20, 30]
RANDOM_SEED = 42

# Plotting
OUTPUT_DIR = 'outputs'