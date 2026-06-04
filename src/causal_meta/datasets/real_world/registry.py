"""Central dispatch for real-world dataset loaders."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import torch

# Lazy-imported loaders are registered here.  Each loader function must accept
# optional ``**kwargs`` and return ``(data, adjacency)`` as float32 tensors.
_LOADER_REGISTRY: Dict[str, Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = {}


def _register(name: str, fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]]) -> None:
    _LOADER_REGISTRY[name] = fn


def load_real_world_dataset(
    loader_name: str,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a real-world dataset by name.

    Args:
        loader_name: Registered loader identifier (e.g. ``"sachs"``).
        **kwargs: Forwarded to the loader function.

    Returns:
        ``(data, adjacency)`` — both float32 tensors.

    Raises:
        KeyError: If *loader_name* is not registered.
    """
    # Ensure built-in loaders are registered.
    _ensure_builtins()

    fn = _LOADER_REGISTRY.get(loader_name)
    if fn is None:
        available = sorted(_LOADER_REGISTRY)
        raise KeyError(
            f"Unknown real-world dataset loader '{loader_name}'. Available: {available}"
        )
    return fn(**kwargs)


_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Import submodules to trigger registration.
    from causal_meta.datasets.real_world import sachs as _sachs  # noqa: F401
    from causal_meta.datasets.real_world import syntren as _syntren  # noqa: F401

    _BUILTINS_LOADED = True
