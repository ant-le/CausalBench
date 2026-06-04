"""Configurable noise distribution samplers for SCM data generation.

The training data uses standard Gaussian noise exclusively. This module
provides alternative noise distributions (Laplace, Uniform) for OOD
evaluation of noise-distribution robustness. All samplers produce
zero-mean, unit-variance noise so the only distributional difference
is the shape (tail weight, kurtosis), not the scale.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import torch

# Type alias: (shape, device, dtype) -> Tensor
NoiseSampler = Callable[[Tuple[int, ...], torch.device, torch.dtype], torch.Tensor]

# sqrt(3) for scaling uniform to unit variance
_SQRT3 = math.sqrt(3.0)


def _gaussian_sampler(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Standard Gaussian noise, N(0, 1)."""
    return torch.randn(shape, device=device, dtype=dtype)


def _laplace_sampler(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Laplace(0, 1/sqrt(2)) noise, scaled so Var = 1.

    Laplace(loc, scale) has variance 2*scale^2.
    Setting scale = 1/sqrt(2) gives Var = 1.
    """
    scale = 1.0 / math.sqrt(2.0)
    return (
        torch.distributions.Laplace(0.0, scale)
        .sample(shape)
        .to(device=device, dtype=dtype)
    )


def _uniform_sampler(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Uniform(-sqrt(3), sqrt(3)) noise, scaled so Var = 1.

    U(a, b) has variance (b - a)^2 / 12.
    Setting a = -sqrt(3), b = sqrt(3) gives Var = 12 / 12 = 1.
    """
    return torch.rand(shape, device=device, dtype=dtype) * 2.0 * _SQRT3 - _SQRT3


_NOISE_REGISTRY: dict[str, NoiseSampler] = {
    "gaussian": _gaussian_sampler,
    "laplace": _laplace_sampler,
    "uniform": _uniform_sampler,
}

SUPPORTED_NOISE_TYPES: tuple[str, ...] = tuple(_NOISE_REGISTRY.keys())


def create_noise_sampler(noise_type: str = "gaussian") -> NoiseSampler:
    """Return a noise sampler callable for the given distribution type.

    All returned samplers produce zero-mean, unit-variance noise so
    that the only OOD factor is the distributional shape (tail weight,
    kurtosis), not the amplitude.

    Args:
        noise_type: One of ``"gaussian"``, ``"laplace"``, ``"uniform"``.

    Returns:
        A callable ``(shape, device, dtype) -> Tensor``.

    Raises:
        ValueError: If *noise_type* is not recognised.
    """
    sampler = _NOISE_REGISTRY.get(noise_type.lower())
    if sampler is None:
        raise ValueError(
            f"Unknown noise type '{noise_type}'. "
            f"Supported types: {SUPPORTED_NOISE_TYPES}"
        )
    return sampler
