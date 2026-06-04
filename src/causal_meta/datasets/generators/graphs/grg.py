from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import torch


class GeometricRandomGenerator:
    """Geometric Random Graph DAG sampler.

    Generates an undirected random geometric graph and then orients edges via
    a random node ordering to produce a DAG.

    The model places ``n`` nodes uniformly at random in a ``dim``-dimensional
    unit cube and connects every pair of nodes whose Euclidean distance is at
    most ``radius``.  The resulting graphs exhibit strong spatial locality:
    nodes that are geometrically close tend to be connected, producing
    clustered, lattice-like structures that differ markedly from the global
    connectivity patterns of Erdos-Renyi or scale-free graphs.

    Args:
        radius: Distance threshold for edge creation.  Must be positive.
        dim: Dimensionality of the embedding space.  Defaults to 2.
    """

    def __init__(self, radius: float = 0.3, dim: int = 2) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        if dim < 1:
            raise ValueError("dim must be at least 1.")
        self.radius = radius
        self.dim = dim

    def __call__(
        self,
        n_nodes: int,
        *,
        seed: Optional[int] = None,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        graph = nx.random_geometric_graph(
            n=n_nodes, radius=self.radius, dim=self.dim, seed=seed
        )

        # Orient edges via a random node ordering to produce a DAG.
        order_rng = rng if rng is not None else np.random.default_rng(seed)
        order = order_rng.permutation(n_nodes)
        order_position = {node: idx for idx, node in enumerate(order)}

        adjacency = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
        for u, v in graph.edges():
            if order_position[u] < order_position[v]:
                adjacency[u, v] = 1.0
            else:
                adjacency[v, u] = 1.0
        return adjacency
