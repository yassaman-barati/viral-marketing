"""Module for loading and generating graphs using NetworkX."""

import networkx as nx
from typing import Dict

def generate_erdos_renyi(n: int, p: float) -> nx.Graph:
    """Generate an Erdos-Renyi random graph.

    Args:
        n: Number of nodes.
        p: Probability for edge creation.

    Returns:
        A NetworkX Graph object.
    """
    return nx.erdos_renyi_graph(n, p)

def generate_barabasi_albert(n: int, m: int) -> nx.Graph:
    """Generate a Barabási-Albert preferential attachment graph.

    Args:
        n: Number of nodes.
        m: Number of edges to attach from a new node to existing nodes.

    Returns:
        A NetworkX Graph object.
    """
    return nx.barabasi_albert_graph(n, m)

def load_from_edgelist(file_path: str) -> nx.Graph:
    """Load a graph from an edge list file.

    Args:
        file_path: Path to the edge list file.

    Returns:
        A NetworkX Graph object.
    """
    return nx.read_edgelist(file_path, create_using=nx.Graph)

def graph_statistics(g: nx.Graph) -> Dict[str, float]:
    """Compute basic statistics for a graph.

    Args:
        g: The input graph.

    Returns:
        A dictionary with statistics: num_nodes, num_edges, avg_degree, density.
    """
    stats = {
        'num_nodes': g.number_of_nodes(),
        'num_edges': g.number_of_edges(),
        'avg_degree': sum(d for n, d in g.degree()) / g.number_of_nodes(),
        'density': nx.density(g)
    }
    return stats