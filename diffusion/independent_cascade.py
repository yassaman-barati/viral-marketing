# influence_maximization_project/diffusion/independent_cascade.py
"""Implementation of the Independent Cascade diffusion model."""

import random
from typing import List, Set
import networkx as nx

def simulate_ic(g: nx.Graph, seeds: List[int], p: float) -> int:
    """Simulate the Independent Cascade model on a graph.

    Each active node attempts to activate its inactive neighbors with probability p.
    Each attempt happens exactly once.

    Args:
        g: The input graph.
        seeds: List of seed nodes.
        p: Propagation probability.

    Returns:
        The number of activated nodes.
    """
    active = set(seeds)
    newly_active = set(seeds)
    
    while newly_active:
        next_newly_active = set()
        for node in newly_active:
            for neighbor in g.neighbors(node):
                if neighbor not in active:
                    if random.random() < p:
                        next_newly_active.add(neighbor)
        active.update(next_newly_active)
        newly_active = next_newly_active
    
    return len(active)