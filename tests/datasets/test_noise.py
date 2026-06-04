"""Tests for the configurable noise distribution samplers."""

from __future__ import annotations

import math

import pytest
import torch

from causal_meta.datasets.noise import (
    SUPPORTED_NOISE_TYPES,
    NoiseSampler,
    create_noise_sampler,
)


# ── Factory basics ─────────────────────────────────────────────────────


def test_create_noise_sampler_returns_callable() -> None:
    for noise_type in SUPPORTED_NOISE_TYPES:
        sampler = create_noise_sampler(noise_type)
        assert callable(sampler)


def test_unknown_noise_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown noise type"):
        create_noise_sampler("beta")


def test_default_is_gaussian() -> None:
    sampler = create_noise_sampler()
    out = sampler((5, 1), torch.device("cpu"), torch.float32)
    assert out.shape == (5, 1)


# ── Output shape and dtype ─────────────────────────────────────────────


@pytest.mark.parametrize("noise_type", SUPPORTED_NOISE_TYPES)
def test_output_shape(noise_type: str) -> None:
    sampler = create_noise_sampler(noise_type)
    shape = (100, 1)
    out = sampler(shape, torch.device("cpu"), torch.float32)
    assert out.shape == shape
    assert out.dtype == torch.float32


@pytest.mark.parametrize("noise_type", SUPPORTED_NOISE_TYPES)
def test_output_dtype_float64(noise_type: str) -> None:
    sampler = create_noise_sampler(noise_type)
    out = sampler((50,), torch.device("cpu"), torch.float64)
    # Laplace sampler uses .to() so dtype should match.
    assert out.dtype == torch.float64


# ── Statistical properties (zero mean, unit variance) ──────────────────


@pytest.mark.parametrize("noise_type", SUPPORTED_NOISE_TYPES)
def test_zero_mean(noise_type: str) -> None:
    sampler = create_noise_sampler(noise_type)
    samples = sampler((50_000,), torch.device("cpu"), torch.float32)
    mean = samples.mean().item()
    assert abs(mean) < 0.05, f"{noise_type}: mean={mean}"


@pytest.mark.parametrize("noise_type", SUPPORTED_NOISE_TYPES)
def test_unit_variance(noise_type: str) -> None:
    sampler = create_noise_sampler(noise_type)
    samples = sampler((50_000,), torch.device("cpu"), torch.float32)
    var = samples.var().item()
    assert abs(var - 1.0) < 0.1, f"{noise_type}: var={var}"


# ── Uniform bounds ─────────────────────────────────────────────────────


def test_uniform_bounds() -> None:
    sampler = create_noise_sampler("uniform")
    samples = sampler((100_000,), torch.device("cpu"), torch.float32)
    sqrt3 = math.sqrt(3.0)
    assert samples.min().item() >= -sqrt3 - 1e-6
    assert samples.max().item() <= sqrt3 + 1e-6


# ── Laplace heavier tails than Gaussian ────────────────────────────────


def test_laplace_heavier_tails() -> None:
    """Laplace should have excess kurtosis > 0 (kurtosis = 6 for Laplace)."""
    sampler = create_noise_sampler("laplace")
    samples = sampler((100_000,), torch.device("cpu"), torch.float64)
    mean = samples.mean()
    m4 = ((samples - mean) ** 4).mean()
    m2 = ((samples - mean) ** 2).mean()
    kurtosis = (m4 / m2**2).item()
    # Laplace kurtosis = 6, Gaussian = 3.  Check it's clearly above 3.
    assert kurtosis > 4.0, f"kurtosis={kurtosis}"
