from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import torch
from causal_meta.datasets.generators.graphs.base import GraphGenerator


class MixtureGraphGenerator:
    """Samples a graph generator from a weighted mixture for each call."""

    def __init__(self, generators: Sequence[GraphGenerator], weights: Sequence[float]) -> None:
        if len(generators) != len(weights):
            raise ValueError(
                "Generators and weights must have the same length.")
        if not generators:
            raise ValueError("At least one generator is required.")

        self.generators = generators
        total = sum(weights)
        if total <= 0:
            raise ValueError("Sum of weights must be positive.")
        self.weights = [w / total for w in weights]

    def __call__(
        self,
        n_nodes: int,
        *,
        seed: Optional[int] = None,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        # We need a source of randomness to pick the generator.
        # Prioritize rng (numpy) if provided, otherwise make one from seed.
        local_rng = rng if rng is not None else np.random.default_rng(seed)

        idx = local_rng.choice(len(self.generators), p=self.weights)
        selected_gen = self.generators[idx]

        return selected_gen(
            n_nodes,
            seed=seed,
            torch_generator=torch_generator,
            rng=rng
        )
