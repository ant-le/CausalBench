"""Real-world causal-discovery benchmark datasets.

Each loader returns ``(data, adjacency)`` where *data* is a float32 tensor of
shape ``(N, d)`` and *adjacency* is a float32 tensor of shape ``(d, d)``.
"""

from __future__ import annotations

from causal_meta.datasets.real_world.registry import load_real_world_dataset

__all__ = ["load_real_world_dataset"]
