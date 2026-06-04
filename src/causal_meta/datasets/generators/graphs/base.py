from __future__ import annotations
from typing import Optional, Protocol
import numpy as np
import torch

class GraphGenerator(Protocol):
    """Protocol for graph generation strategies."""

    def __call__(
        self,
        n_nodes: int,
        *,
        seed: Optional[int] = None,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        ...
