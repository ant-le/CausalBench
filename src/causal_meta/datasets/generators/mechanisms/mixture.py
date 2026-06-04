from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
import torch
from torch import nn
from causal_meta.datasets.generators.mechanisms.base import (
    MechanismFactory,
    build_mechanisms_from_adjacency,
)


class MixtureMechanismFactory:
    """Samples a mechanism factory per node from a weighted mixture."""

    def __init__(
        self, factories: Sequence[MechanismFactory], weights: Sequence[float]
    ) -> None:
        if len(factories) != len(weights):
            raise ValueError("Factories and weights must have the same length.")
        if not factories:
            raise ValueError("At least one factory is required.")

        self.factories = factories
        total = sum(weights)
        if total <= 0:
            raise ValueError("Sum of weights must be positive.")
        self.weights = [w / total for w in weights]

    def make_mechanism(
        self, input_dim: int, torch_generator: torch.Generator
    ) -> nn.Module:
        # Pick a factory at random using the torch_generator
        idx = torch.multinomial(
            torch.tensor(self.weights), 1, generator=torch_generator
        ).item()
        return self.factories[idx].make_mechanism(input_dim, torch_generator)

    def __call__(
        self,
        adjacency_matrix: torch.Tensor,
        *,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[nn.Module]:
        return build_mechanisms_from_adjacency(
            self, adjacency_matrix, torch_generator=torch_generator, rng=rng
        )
