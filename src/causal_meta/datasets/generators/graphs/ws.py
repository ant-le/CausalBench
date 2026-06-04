from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import torch


class WattsStrogatzGenerator:
    """Watts-Strogatz small-world DAG sampler.

    Generates an undirected small-world graph using the Watts-Strogatz model
    and then orients edges via a random node ordering to produce a DAG.

    The Watts-Strogatz model starts from a regular ring lattice where each
    node is connected to its ``k`` nearest neighbours, then rewires each edge
    with probability ``p``.  Low ``p`` yields high clustering with long path
    lengths (lattice-like), while high ``p`` yields low clustering with short
    path lengths (random-graph-like).  The "small-world" regime lies at
    intermediate ``p`` where clustering remains high but path lengths are short.

    Args:
        k: Each node is joined with its ``k`` nearest neighbours in the ring
            topology.  Must be even and >= 2.
        p: Probability of rewiring each edge.  Must be in [0, 1].
    """

    def __init__(self, k: int = 4, p: float = 0.3) -> None:
        if k < 2 or k % 2 != 0:
            raise ValueError("k must be an even integer >= 2.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")
        self.k = k
        self.p = p

    def __call__(
        self,
        n_nodes: int,
        *,
        seed: Optional[int] = None,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        if n_nodes <= self.k:
            raise ValueError(
                f"n_nodes ({n_nodes}) must be greater than k ({self.k}) "
                "for Watts-Strogatz graphs."
            )

        graph = nx.watts_strogatz_graph(n=n_nodes, k=self.k, p=self.p, seed=seed)

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
