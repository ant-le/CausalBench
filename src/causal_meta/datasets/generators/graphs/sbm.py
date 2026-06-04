from __future__ import annotations
from typing import Optional
import networkx as nx
import numpy as np
import torch


class SBMGenerator:
    """Stochastic Block Model DAG sampler."""

    def __init__(self, n_blocks: int, p_intra: float, p_inter: float) -> None:
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive.")
        if not (0 <= p_intra <= 1 and 0 <= p_inter <= 1):
            raise ValueError("Probabilities must be in [0, 1].")
        self.n_blocks = n_blocks
        self.p_intra = p_intra
        self.p_inter = p_inter

    def __call__(
        self,
        n_nodes: int,
        *,
        seed: Optional[int] = None,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        if n_nodes < self.n_blocks:
            raise ValueError("n_nodes must be at least as large as n_blocks.")

        base_size, remainder = divmod(n_nodes, self.n_blocks)
        block_sizes = [base_size + (1 if i < remainder else 0)
                       for i in range(self.n_blocks)]
        probabilities = [
            [self.p_intra if i ==
                j else self.p_inter for j in range(self.n_blocks)]
            for i in range(self.n_blocks)
        ]

        graph = nx.stochastic_block_model(
            block_sizes, probabilities, seed=seed)
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
