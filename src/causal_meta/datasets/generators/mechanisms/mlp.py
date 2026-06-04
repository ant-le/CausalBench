from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
from torch import nn

from .base import build_mechanisms_from_adjacency


class MLPMechanism(nn.Module):
    """Three-layer MLP mechanism with noise concatenated to inputs."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, parents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([parents, noise], dim=1)
        return self.net(inputs).squeeze(-1)


class MLPMechanismFactory:
    """Factory producing two-layer MLP mechanisms."""

    def __init__(self, hidden_dim: int = 32) -> None:
        self.hidden_dim = hidden_dim

    def make_mechanism(
        self, input_dim: int, torch_generator: torch.Generator
    ) -> nn.Module:
        # Initialize module on CPU (default). We want random weights to be deterministic
        # based on torch_generator. PyTorch init uses global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                int(
                    torch.randint(
                        0, torch.iinfo(torch.int64).max, (1,), generator=torch_generator
                    ).item()
                )
            )
            mech = MLPMechanism(input_dim=input_dim, hidden_dim=self.hidden_dim)
        return mech

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
