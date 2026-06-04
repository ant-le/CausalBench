from __future__ import annotations

from typing import List, Optional, Protocol

import numpy as np
import torch
from torch import nn


class MechanismFactory(Protocol):
    """Protocol for mechanism factories."""

    def make_mechanism(
        self, input_dim: int, torch_generator: torch.Generator
    ) -> nn.Module: ...

    def __call__(
        self,
        adjacency_matrix: torch.Tensor,
        *,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[nn.Module]: ...


def build_mechanisms_from_adjacency(
    factory: MechanismFactory,
    adjacency_matrix: torch.Tensor,
    *,
    torch_generator: Optional[torch.Generator] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[nn.Module]:
    """Shared implementation of the factory ``__call__`` boilerplate.

    Iterates over nodes, counts parents from the adjacency matrix, and
    delegates to ``factory.make_mechanism``.

    Args:
        factory: A mechanism factory implementing ``make_mechanism``.
        adjacency_matrix: Binary adjacency with shape ``(N, N)``.
        torch_generator: Optional generator for reproducibility.
        rng: Unused — accepted for protocol compatibility.

    Returns:
        List of ``nn.Module`` mechanisms, one per node.
    """
    _ = rng
    torch_generator = torch_generator or torch.Generator()

    mechanisms: List[nn.Module] = []
    n_nodes = adjacency_matrix.shape[0]
    for node in range(n_nodes):
        parents = torch.nonzero(adjacency_matrix[:, node], as_tuple=False).flatten()
        mechanisms.append(factory.make_mechanism(int(parents.numel()), torch_generator))
    return mechanisms
