# influence_maximization_project/maximization/heuristics.py
"""Heuristic methods for seed selection in Influence Maximization."""

import random
from typing import List
import networkx as nx

def high_degree(g: nx.Graph, k: int) -> List[int]:
    """Select top k nodes by degree centrality.

    Args:
        g: The input graph.
        k: Number of seeds.

    Returns:
        List of selected nodes.
    """
    degrees = dict(g.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    return sorted_nodes[:k]

def page_rank(g: nx.Graph, k: int) -> List[int]:
    """Select top k nodes by PageRank.

    Args:
        g: The input graph.
        k: Number of seeds.

    Returns:
        List of selected nodes.
    """
    pr = nx.pagerank(g)
    sorted_nodes = sorted(pr, key=pr.get, reverse=True)
    return sorted_nodes[:k]

def random_selection(g: nx.Graph, k: int, seed: int = None) -> List[int]:
    """Select k random nodes.

    Args:
        g: The input graph.
        k: Number of seeds.
        seed: Random seed for reproducibility.

    Returns:
        List of selected nodes.
    """
    if seed is not None:
        random.seed(seed)
    return random.sample(list(g.nodes()), k)