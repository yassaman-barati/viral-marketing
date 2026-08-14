# Influence Maximization & Viral Marketing in Complex Networks

A comprehensive Python framework for simulating stochastic diffusion models (**Independent Cascade** and **Linear Threshold**) and benchmarking **Influence Maximization (IM)** algorithms (Greedy optimization vs. Centrality-based heuristics).

---

## 📌 Project Overview

Influence Maximization (IM) is a fundamental problem in viral marketing and network science: *Given a social graph and an information diffusion model, find a seed set of $k$ nodes that maximizes the expected spread of influence.*

### Implemented Components
- **Diffusion Models:**
  - **Independent Cascade (IC)**: Edge-activation based stochastic spreading.
  - **Linear Threshold (LT)**: Node-threshold based collective influence spreading.
- **Seed Selection Strategies:**
  - **Greedy Strategy**: Uses Monte Carlo simulations for expected spread estimation with theoretical $(1 - 1/e)$-approximation guarantee.
  - **Degree Centrality Heuristic**: Selects high-degree hub nodes.
  - **PageRank Centrality Heuristic**: Selects structurally prestigious nodes.
  - **Random Baseline**: Selects seeds uniformly at random.
- **Experimental Evaluation:**
  - **Influence Coverage vs. Seed Size ($k$)**
  - **Computational Runtime Benchmarking**

---

## 🗂️ Project Structure

```text
.
├── diffusion/
│   ├── independent_cascade.py    # IC diffusion simulation
│   └── linear_threshold.py       # LT diffusion simulation
├── maximization/
│   ├── greedy.py                 # Greedy influence maximization algorithm
│   └── heuristics.py             # Degree, PageRank, and Random heuristics
├── evaluation/
│   ├── monte_carlo.py            # Monte Carlo spread evaluation
│   └── experiments.py           # Experiment runner for varying seed sizes
├── outputs/                      # Output visualization plots
│   ├── IC_coverage.png
│   ├── IC_runtime.png
│   ├── LT_coverage.png
│   └── LT_runtime.png
├── utils/
│   ├── random_utils.py           # Random seed and sampling helpers
│   └── timing.py                 # Benchmarking and timing decorators
├── config.py                     # Hyperparameters and simulation configurations
├── graph_loader.py               # Graph generation and network loading
├── main.py                       # Main pipeline execution entry point
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
