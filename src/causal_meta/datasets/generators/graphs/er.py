from __future__ import annotations
from typing import Optional
import numpy as np
import torch


class ErdosRenyiGenerator:
    """Vectorized Erdős-Rényi DAG sampler using torch.bernoulli."""

    def __init__(
        self, edge_prob: Optional[float] = None, *, sparsity: Optional[float] = None
    ) -> None:
        if edge_prob is None and sparsity is None:
            raise TypeError(
                "Either 'edge_prob' or its alias 'sparsity' must be provided."
            )
        if edge_prob is not None and sparsity is not None and edge_prob != sparsity:
            raise ValueError(
                "'edge_prob' and 'sparsity' must match if both are provided."
            )

        edge_prob = sparsity if edge_prob is None else edge_prob
        if edge_prob is None:  # pragma: no cover - defensive
            raise TypeError("Missing required edge probability.")
        if edge_prob < 0 or edge_prob > 1:
            raise ValueError("Probability must be in [0, 1].")
        self.edge_prob = float(edge_prob)

    def __call__(
        self,
        n_nodes: int,
        *,
        seed: Optional[int] = None,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        torch_generator = torch_generator or torch.Generator()
        if seed is not None:
            torch_generator = torch_generator.manual_seed(seed)

        mask = torch.triu(torch.ones((n_nodes, n_nodes)), diagonal=1)
        prob_matrix = mask * self.edge_prob
        adjacency = torch.bernoulli(prob_matrix, generator=torch_generator)

        # Randomize node order while preserving acyclicity (P^T A P).
        # This mirrors the BCNP/AVICI synthetic workflow where DAG variables
        # are permuted after generation to avoid fixed topological indexing.
        if rng is not None:
            perm = torch.as_tensor(rng.permutation(n_nodes), dtype=torch.long)
        else:
            perm = torch.randperm(n_nodes, generator=torch_generator)

        adjacency = adjacency.index_select(0, perm).index_select(1, perm)
        return adjacency.float()
