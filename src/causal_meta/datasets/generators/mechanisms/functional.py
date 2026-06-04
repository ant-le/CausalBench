from __future__ import annotations
from typing import List, Optional, Callable
import torch
from torch import nn
import numpy as np

from .base import build_mechanisms_from_adjacency


class FunctionalMechanism(nn.Module):
    """Base class for functional mechanisms."""

    def __init__(
        self,
        weights: torch.Tensor,
        func: Callable[[torch.Tensor], torch.Tensor],
        noise_scale: float = 0.1,
        additive_noise: bool = True,
        pre_function_noise: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("weights", weights.view(-1))
        self.register_buffer("noise_scale", torch.tensor(float(noise_scale)))
        self.func = func
        self.additive_noise = additive_noise
        self.pre_function_noise = pre_function_noise

    def forward(self, parents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Linear combination of parents
        if parents.numel() > 0:
            linear = parents @ self.weights
        else:
            # If no parents, bias term (or 0)
            linear = torch.zeros(parents.shape[0], device=parents.device)

        if self.pre_function_noise:
            linear = linear + noise.view(-1) * self.noise_scale

        # Apply function
        output = self.func(linear)

        # Add noise
        if self.additive_noise:
            output = output + noise.view(-1) * self.noise_scale

        return output


class SquareMechanismFactory:
    def __init__(self, weight_scale: float = 1.0, noise_scale: float = 0.1) -> None:
        self.weight_scale = weight_scale
        self.noise_scale = noise_scale

    def make_mechanism(
        self, input_dim: int, torch_generator: torch.Generator
    ) -> nn.Module:
        if input_dim > 0:
            weights = (
                torch.randn(input_dim, generator=torch_generator) * self.weight_scale
            )
        else:
            weights = torch.zeros(0)

        return FunctionalMechanism(
            weights, func=lambda x: x**2, noise_scale=self.noise_scale
        )

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


class PeriodicMechanismFactory:
    def __init__(self, weight_scale: float = 1.0, noise_scale: float = 0.1) -> None:
        self.weight_scale = weight_scale
        self.noise_scale = noise_scale

    def make_mechanism(
        self, input_dim: int, torch_generator: torch.Generator
    ) -> nn.Module:
        if input_dim > 0:
            weights = (
                torch.randn(input_dim, generator=torch_generator) * self.weight_scale
            )
        else:
            weights = torch.zeros(0)

        # sin(4 * pi * x)
        return FunctionalMechanism(
            weights,
            func=lambda x: torch.sin(4 * torch.pi * x),
            noise_scale=self.noise_scale,
        )

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


class LogisticMapMechanismFactory:
    def __init__(self, weight_scale: float = 1.0, noise_scale: float = 1.0) -> None:
        self.weight_scale = weight_scale
        self.noise_scale = noise_scale

    def make_mechanism(
        self, input_dim: int, torch_generator: torch.Generator
    ) -> nn.Module:
        if input_dim > 0:
            weights = (
                torch.randn(input_dim, generator=torch_generator) * self.weight_scale
            )
        else:
            weights = torch.zeros(0)

        # 4 * sigmoid(x) * (1 - sigmoid(x)); noise enters inside the bounded
        # transform so root nodes remain stochastic without additive noise.
        def logistic_map(x):
            s = torch.sigmoid(x)
            return 4 * s * (1 - s)

        return FunctionalMechanism(
            weights,
            func=logistic_map,
            noise_scale=self.noise_scale,
            additive_noise=False,
            pre_function_noise=True,
        )

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
