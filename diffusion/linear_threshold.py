# influence_maximization_project/diffusion/linear_threshold.py
"""Implementation of the Linear Threshold diffusion model."""

import random
from typing import List, Dict
import networkx as nx

def simulate_lt(g: nx.Graph, seeds: List[int]) -> int:
    """Simulate the Linear Threshold model on a graph.

    Each node has a random threshold uniform in [0,1].
    Edge weights are uniform 1/deg(incoming), but since undirected, use 1/deg.

    Args:
        g: The input graph.
        seeds: List of seed nodes.

    Returns:
        The number of activated nodes.
    """
    thresholds: Dict[int, float] = {node: random.uniform(0, 1) for node in g.nodes()}
    active = set(seeds)
    newly_active = set(seeds)
    
    while newly_active:
        next_newly_active = set()
        influences: Dict[int, float] = {node: 0.0 for node in g.nodes() if node not in active}
        
        for node in newly_active:
            for neighbor in g.neighbors(node):
                if neighbor not in active:
                    weight = 1.0 / g.degree(neighbor)
                    influences[neighbor] += weight
        
        for node, influence in influences.items():
            if influence >= thresholds[node]:
                next_newly_active.add(node)
        
        active.update(next_newly_active)
        newly_active = next_newly_active
    
    return len(active)