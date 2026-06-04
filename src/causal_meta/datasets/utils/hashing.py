from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import torch
from torch import nn


def compute_graph_hash(
    adj: torch.Tensor,
    mechanisms: Sequence[nn.Module] | None = None,
    *,
    include_mechanisms: bool = False,
) -> str:
    """Return a deterministic hash for an adjacency matrix and optionally its mechanisms.

    Args:
        adj: Adjacency matrix tensor of shape (N, N).
        mechanisms: Optional sequence of mechanism modules (one per node).
        include_mechanisms: If True and mechanisms are provided, include mechanism
            parameters in the hash. This enables disjointness checks that respect
            functional generalization on identical DAGs.

    Returns:
        A SHA-256 hex digest string.
    """
    hasher = hashlib.sha256()

    # Hash adjacency structure
    adj_bytes = adj.detach().cpu().numpy().astype(np.int8).tobytes()
    hasher.update(adj_bytes)

    # Optionally hash mechanism parameters
    if include_mechanisms and mechanisms is not None:
        for i, mech in enumerate(mechanisms):
            # Add node index to disambiguate mechanisms at different positions
            hasher.update(f"node_{i}:".encode())
            hasher.update(type(mech).__name__.encode())

            # Hash all registered buffers and parameters
            state_dict = mech.state_dict()
            for key in sorted(state_dict.keys()):
                tensor = state_dict[key]
                hasher.update(f"{key}:".encode())
                # Use float32 for consistent hashing across dtypes
                tensor_bytes = (
                    tensor.detach().cpu().to(dtype=torch.float32).numpy().tobytes()
                )
                hasher.update(tensor_bytes)

    return hasher.hexdigest()
