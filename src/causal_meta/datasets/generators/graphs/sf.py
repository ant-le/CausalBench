from __future__ import annotations
from typing import Optional
import networkx as nx
import numpy as np
import torch


class ScaleFreeGenerator:
    """Barabási-Albert based DAG sampler."""

    def __init__(self, m: int = 2) -> None:
        if m < 1:
            raise ValueError("m must be at least 1.")
        self.m = m

    def __call__(
        self,
        n_nodes: int,
        *,
        seed: Optional[int] = None,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        if n_nodes <= self.m:
            raise ValueError(
                "n_nodes must be greater than m for Barabási-Albert graphs.")

        graph = nx.barabasi_albert_graph(n=n_nodes, m=self.m, seed=seed)
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
