from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
from torch import nn

from .base import build_mechanisms_from_adjacency


class LinearMechanism(nn.Module):
    """Additive linear mechanism with Gaussian noise."""

    def __init__(self, weights: torch.Tensor, noise_scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("weights", weights.view(-1))
        self.register_buffer(
            "noise_scale", torch.tensor(float(noise_scale), dtype=torch.float32)
        )

    def forward(self, parents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        linear_term = (
            parents @ self.weights
            if parents.numel() > 0
            else torch.zeros(
                parents.shape[0], device=parents.device, dtype=parents.dtype
            )
        )
        return linear_term + noise.view(-1) * self.noise_scale


class LinearMechanismFactory:
    """Factory producing linear mechanisms aligned with the adjacency."""

    def __init__(
        self,
        weight_scale: float = 1.0,
        noise_concentration: float = 2.0,
        noise_rate: float = 2.0,
    ) -> None:
        self.weight_scale = weight_scale
        self.noise_concentration = noise_concentration
        self.noise_rate = noise_rate

    def make_mechanism(
        self, input_dim: int, torch_generator: torch.Generator
    ) -> nn.Module:
        # We must ensure torch_generator state is advanced, but torch.randn requires a global RNG
        # or a generator kwarg. Gamma doesn't accept a generator directly in older torch versions easily
        # or requires standard sampling. We use the generator directly where possible.
        if input_dim > 0:
            weights = (
                torch.randn(input_dim, generator=torch_generator) * self.weight_scale
            )
        else:
            weights = torch.zeros(0)

        # Gamma distribution sampling isn't trivially generator-aware in all versions,
        # so we sample uniform and transform or fork. Forking is safer.
        # Actually, torch.distributions...sample() doesn't take a generator.
        # We'll use a workaround: sample uniform from generator and inverse CDF (ppf) or just fork.
        # Forking with set_state is robust.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                int(
                    torch.randint(
                        0, torch.iinfo(torch.int64).max, (1,), generator=torch_generator
                    ).item()
                )
            )
            noise_scale = torch.distributions.Gamma(
                self.noise_concentration, self.noise_rate
            ).sample()

        return LinearMechanism(weights, noise_scale=noise_scale.item())

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
