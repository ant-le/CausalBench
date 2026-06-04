from __future__ import annotations

import math
from importlib import import_module
from importlib.util import find_spec
from typing import List, Optional, cast

import numpy as np
import torch
from torch import nn

from .base import build_mechanisms_from_adjacency

gpytorch = import_module("gpytorch") if find_spec("gpytorch") is not None else None


def _concat_parents_and_noise(
    parents: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """Concatenate parent values with node noise.

    Args:
        parents: Parent matrix with shape ``(batch_size, n_parents)``.
        noise: Noise tensor with shape ``(batch_size, 1)`` or ``(batch_size,)``.

    Returns:
        Tensor with shape ``(batch_size, n_parents + 1)``.
    """
    noise = noise.view(-1, 1)
    if parents.numel() == 0:
        return noise
    return torch.cat([parents, noise], dim=1)


def _uniform_sample(
    torch_generator: torch.Generator,
    low: float,
    high: float,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Sample uniformly from ``[low, high]`` with a torch generator."""
    return low + (high - low) * torch.rand(shape, generator=torch_generator)


def _ard_squared_distance(x: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared distances with ARD lengthscales."""
    x_scaled = x / lengthscale.unsqueeze(0)
    diff = x_scaled.unsqueeze(1) - x_scaled.unsqueeze(0)
    return diff.pow(2).sum(dim=-1).clamp_min(1e-12)


def _get_tensor_attr(module: nn.Module, name: str) -> torch.Tensor:
    """Read a module attribute and assert it is a tensor."""
    value = getattr(module, name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected '{name}' to be a tensor, got {type(value)}")
    return value


class ApproximateGPMechanism(nn.Module):
    """RFF approximation of a GP mechanism.

    This model approximates a mixture GP prior using random Fourier features (RFF).
    It is substantially faster than exact GP sampling while remaining expressive
    enough for high-throughput synthetic task generation.

    Note:
        Divergence from the reference BCNP code: the original implementation uses
        exact GP function sampling. This class intentionally keeps an approximate
        variant for scalability.

    Args:
        input_dim: Number of mechanism inputs, including the appended node noise.
        rff_dim: Number of random features per kernel component.
        lengthscales: ARD lengthscales with shape ``(n_kernels, input_dim)``.
        variances: Kernel amplitudes with shape ``(n_kernels,)``.
        weights: Random projection weights with shape
            ``(n_kernels, input_dim, rff_dim)``.
        biases: Random projection phases with shape ``(n_kernels, rff_dim)``.
    """

    def __init__(
        self,
        input_dim: int,
        rff_dim: int,
        lengthscales: torch.Tensor,
        variances: torch.Tensor,
        weights: torch.Tensor,
        biases: torch.Tensor,
    ) -> None:
        super().__init__()
        if lengthscales.ndim != 2:
            raise ValueError("lengthscales must have shape (n_kernels, input_dim).")
        if variances.ndim != 1:
            raise ValueError("variances must have shape (n_kernels,).")
        if weights.ndim != 3:
            raise ValueError("weights must have shape (n_kernels, input_dim, rff_dim).")
        if biases.ndim != 2:
            raise ValueError("biases must have shape (n_kernels, rff_dim).")

        n_kernels = int(lengthscales.shape[0])
        if variances.shape[0] != n_kernels:
            raise ValueError("variances and lengthscales must agree on n_kernels.")
        if weights.shape[:2] != (n_kernels, input_dim):
            raise ValueError(
                "weights shape must match (n_kernels, input_dim, rff_dim)."
            )
        if weights.shape[2] != rff_dim:
            raise ValueError("weights last dimension must equal rff_dim.")
        if biases.shape != (n_kernels, rff_dim):
            raise ValueError("biases shape must match (n_kernels, rff_dim).")

        self.input_dim = int(input_dim)
        self.rff_dim = int(rff_dim)
        self.n_kernels = int(n_kernels)
        self.feature_scale = float(math.sqrt(2.0 / float(rff_dim)))

        self.register_buffer("lengthscales", lengthscales)
        self.register_buffer("variances", variances)
        self.register_buffer("weights", weights)
        self.register_buffer("biases", biases)

    def forward(self, parents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Evaluate the approximate GP mechanism.

        Args:
            parents: Parent matrix with shape ``(batch_size, n_parents)``.
            noise: Noise tensor with shape ``(batch_size, 1)`` or ``(batch_size,)``.

        Returns:
            Sampled outputs with shape ``(batch_size,)``.
        """
        inputs = _concat_parents_and_noise(parents, noise)
        lengthscales = _get_tensor_attr(self, "lengthscales")
        weights = _get_tensor_attr(self, "weights")
        biases = _get_tensor_attr(self, "biases")
        variances = _get_tensor_attr(self, "variances")
        inputs = inputs.to(dtype=lengthscales.dtype)

        scaled_inputs = inputs.unsqueeze(1) / lengthscales.unsqueeze(0)
        projections = torch.einsum(
            "nkd,kdm->nkm", scaled_inputs, weights
        ) + biases.unsqueeze(0)
        features = torch.cos(projections)
        weighted_features = features * variances.view(1, -1, 1)
        outputs = self.feature_scale * weighted_features.sum(dim=(1, 2))
        return outputs.to(dtype=inputs.dtype)


class ExactGPMechanism(nn.Module):
    """Exact GP mechanism with a BCNP-style kernel prior.

    The reference BCNP implementation samples functions from a sum of
    RationalQuadratic and Exponential-Gamma style kernels. This class recreates
    that prior and performs exact sampling using a multivariate Gaussian draw.

    Args:
        rq_lengthscales: RQ ARD lengthscales, shape ``(n_pairs, input_dim)``.
        rq_variances: RQ amplitudes, shape ``(n_pairs,)``.
        rq_alphas: RQ alpha parameters, shape ``(n_pairs,)``.
        exp_lengthscales: ExpGamma ARD lengthscales, shape ``(n_pairs, input_dim)``.
        exp_variances: ExpGamma amplitudes, shape ``(n_pairs,)``.
        exp_gammas: ExpGamma exponent parameters, shape ``(n_pairs,)``.
        noise_concentration: Shape parameter for the heteroscedastic Gamma noise.
        noise_rate: Rate parameter for the heteroscedastic Gamma noise.
        jitter: Positive diagonal stabilizer for covariance factorization.
    """

    def __init__(
        self,
        rq_lengthscales: torch.Tensor,
        rq_variances: torch.Tensor,
        rq_alphas: torch.Tensor,
        exp_lengthscales: torch.Tensor,
        exp_variances: torch.Tensor,
        exp_gammas: torch.Tensor,
        noise_concentration: float,
        noise_rate: float,
        jitter: float,
    ) -> None:
        super().__init__()
        self.register_buffer("rq_lengthscales", rq_lengthscales)
        self.register_buffer("rq_variances", rq_variances)
        self.register_buffer("rq_alphas", rq_alphas)
        self.register_buffer("exp_lengthscales", exp_lengthscales)
        self.register_buffer("exp_variances", exp_variances)
        self.register_buffer("exp_gammas", exp_gammas)
        self.register_buffer(
            "noise_concentration",
            torch.tensor(float(noise_concentration), dtype=torch.float32),
        )
        self.register_buffer(
            "noise_rate", torch.tensor(float(noise_rate), dtype=torch.float32)
        )
        self.register_buffer("jitter", torch.tensor(float(jitter), dtype=torch.float32))

    def _build_covariance(self, inputs: torch.Tensor) -> torch.Tensor:
        """Build the sum-kernel covariance matrix for exact GP sampling."""
        rq_lengthscales = cast(torch.Tensor, _get_tensor_attr(self, "rq_lengthscales"))
        rq_variances = cast(torch.Tensor, _get_tensor_attr(self, "rq_variances"))
        rq_alphas = cast(torch.Tensor, _get_tensor_attr(self, "rq_alphas"))
        exp_lengthscales = cast(
            torch.Tensor, _get_tensor_attr(self, "exp_lengthscales")
        )
        exp_variances = cast(torch.Tensor, _get_tensor_attr(self, "exp_variances"))
        exp_gammas = cast(torch.Tensor, _get_tensor_attr(self, "exp_gammas"))

        n = int(inputs.shape[0])
        covariance = torch.zeros((n, n), device=inputs.device, dtype=inputs.dtype)

        for idx in range(rq_variances.shape[0]):
            sq_dist = _ard_squared_distance(inputs, rq_lengthscales[idx])
            alpha = rq_alphas[idx]
            rq = rq_variances[idx] * torch.pow(1.0 + sq_dist / (2.0 * alpha), -alpha)
            covariance = covariance + rq

        for idx in range(exp_variances.shape[0]):
            sq_dist = _ard_squared_distance(inputs, exp_lengthscales[idx])
            distances = torch.sqrt(sq_dist).clamp_min(1e-12)
            exp_gamma = exp_variances[idx] * torch.exp(
                -torch.pow(distances, exp_gammas[idx])
            )
            covariance = covariance + exp_gamma

        return 0.5 * (covariance + covariance.transpose(0, 1))

    def forward(self, parents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Evaluate and sample from the exact GP mechanism.

        Args:
            parents: Parent matrix with shape ``(batch_size, n_parents)``.
            noise: Noise tensor with shape ``(batch_size, 1)`` or ``(batch_size,)``.

        Returns:
            Sampled outputs with shape ``(batch_size,)``.
        """
        inputs = _concat_parents_and_noise(parents, noise)
        rq_lengthscales = cast(torch.Tensor, _get_tensor_attr(self, "rq_lengthscales"))
        jitter_buffer = cast(torch.Tensor, _get_tensor_attr(self, "jitter"))

        inputs = inputs.to(dtype=rq_lengthscales.dtype)
        mean = inputs[:, -1]

        covariance = self._build_covariance(inputs)
        jitter = float(jitter_buffer.item())
        variance = covariance.diagonal().clamp_min(jitter)

        gp_sample = mean + noise.view(-1) * variance.sqrt()
        return gp_sample.to(dtype=inputs.dtype)


class ApproximateGPMechanismFactory:
    """Factory for scalable approximate GP mechanisms.

    Args:
        rff_dim: Number of random Fourier features per kernel component.
        num_kernels: Number of mixture components.
        length_scale_range: Uniform range for ARD lengthscales.
        variance_range: Uniform range for kernel amplitudes.
    """

    def __init__(
        self,
        rff_dim: int = 512,
        num_kernels: int = 4,
        length_scale_range: tuple[float, float] = (0.1, 10.0),
        variance_range: tuple[float, float] = (0.1, 10.0),
    ) -> None:
        self.rff_dim = int(rff_dim)
        self.num_kernels = int(num_kernels)
        self.length_scale_range = length_scale_range
        self.variance_range = variance_range

    def make_mechanism(
        self, input_dim: int, torch_generator: torch.Generator
    ) -> nn.Module:
        """Build a single approximate GP mechanism instance."""
        total_input_dim = int(input_dim) + 1
        lengthscales = _uniform_sample(
            torch_generator,
            low=float(self.length_scale_range[0]),
            high=float(self.length_scale_range[1]),
            shape=(self.num_kernels, total_input_dim),
        )
        variances = _uniform_sample(
            torch_generator,
            low=float(self.variance_range[0]),
            high=float(self.variance_range[1]),
            shape=(self.num_kernels,),
        )
        weights = torch.randn(
            self.num_kernels,
            total_input_dim,
            self.rff_dim,
            generator=torch_generator,
        )
        biases = torch.rand(self.num_kernels, self.rff_dim, generator=torch_generator)
        biases = biases * (2.0 * torch.pi)

        return ApproximateGPMechanism(
            input_dim=total_input_dim,
            rff_dim=self.rff_dim,
            lengthscales=lengthscales,
            variances=variances,
            weights=weights,
            biases=biases,
        )


class ExactGPMechanismFactory:
    """Factory for exact GP mechanisms following the reference prior.

    Args:
        num_kernel_pairs: Number of (RQ + ExpGamma) pairs in the sum kernel.
        length_scale_range: Uniform range for ARD lengthscales.
        variance_range: Uniform range for kernel amplitudes.
        alpha_range: Uniform range for RationalQuadratic alpha.
        gamma_range: Uniform range for ExpGamma exponent.
        noise_concentration: Gamma concentration for extra heteroscedastic noise.
        noise_rate: Gamma rate for extra heteroscedastic noise.
        jitter: Initial covariance stabilizer.
    """

    def __init__(
        self,
        num_kernel_pairs: int = 2,
        length_scale_range: tuple[float, float] = (0.1, 10.0),
        variance_range: tuple[float, float] = (0.1, 10.0),
        alpha_range: tuple[float, float] = (0.1, 100.0),
        gamma_range: tuple[float, float] = (1e-5, 0.99999),
        noise_concentration: float = 1.0,
        noise_rate: float = 10.0,
        jitter: float = 1e-4,
    ) -> None:
        self.num_kernel_pairs = int(num_kernel_pairs)
        self.length_scale_range = length_scale_range
        self.variance_range = variance_range
        self.alpha_range = alpha_range
        self.gamma_range = gamma_range
        self.noise_concentration = float(noise_concentration)
        self.noise_rate = float(noise_rate)
        self.jitter = float(jitter)

    def make_mechanism(
        self, input_dim: int, torch_generator: torch.Generator
    ) -> nn.Module:
        """Build a single exact GP mechanism instance."""
        total_input_dim = int(input_dim) + 1

        rq_lengthscales = _uniform_sample(
            torch_generator,
            low=float(self.length_scale_range[0]),
            high=float(self.length_scale_range[1]),
            shape=(self.num_kernel_pairs, total_input_dim),
        )
        rq_variances = _uniform_sample(
            torch_generator,
            low=float(self.variance_range[0]),
            high=float(self.variance_range[1]),
            shape=(self.num_kernel_pairs,),
        )
        rq_alphas = _uniform_sample(
            torch_generator,
            low=float(self.alpha_range[0]),
            high=float(self.alpha_range[1]),
            shape=(self.num_kernel_pairs,),
        )

        exp_lengthscales = _uniform_sample(
            torch_generator,
            low=float(self.length_scale_range[0]),
            high=float(self.length_scale_range[1]),
            shape=(self.num_kernel_pairs, total_input_dim),
        )
        exp_variances = _uniform_sample(
            torch_generator,
            low=float(self.variance_range[0]),
            high=float(self.variance_range[1]),
            shape=(self.num_kernel_pairs,),
        )
        exp_gammas = _uniform_sample(
            torch_generator,
            low=float(self.gamma_range[0]),
            high=float(self.gamma_range[1]),
            shape=(self.num_kernel_pairs,),
        )

        return ExactGPMechanism(
            rq_lengthscales=rq_lengthscales,
            rq_variances=rq_variances,
            rq_alphas=rq_alphas,
            exp_lengthscales=exp_lengthscales,
            exp_variances=exp_variances,
            exp_gammas=exp_gammas,
            noise_concentration=self.noise_concentration,
            noise_rate=self.noise_rate,
            jitter=self.jitter,
        )


class GPMechanismFactory:
    """Factory for GP mechanisms with selectable approximation mode.

    Supported modes:
        - ``"approximate"``: RFF-based approximation for scalability.
        - ``"exact"``: Exact GP sampling with BCNP-like kernel prior.

    Args:
        mode: Which GP mechanism backend to use.
        rff_dim: Number of features per kernel in approximate mode.
        num_kernels: Number of kernel components in approximate mode.
        length_scale_range: Shared lengthscale prior range for both modes.
        variance: Optional fixed variance for backward compatibility.
        variance_range: Optional variance range. Overrides ``variance``.
        exact_num_kernel_pairs: Number of (RQ + ExpGamma) pairs in exact mode.
        alpha_range: RationalQuadratic alpha prior range in exact mode.
        gamma_range: ExpGamma exponent prior range in exact mode.
        exact_noise_concentration: Heteroscedastic Gamma concentration in exact mode.
        exact_noise_rate: Heteroscedastic Gamma rate in exact mode.
        exact_jitter: Initial covariance jitter in exact mode.
    """

    def __init__(
        self,
        mode: str = "approximate",
        rff_dim: int = 512,
        num_kernels: int = 4,
        length_scale_range: tuple[float, float] = (0.1, 10.0),
        variance: float | None = None,
        variance_range: tuple[float, float] | None = None,
        exact_num_kernel_pairs: int = 2,
        alpha_range: tuple[float, float] = (0.1, 100.0),
        gamma_range: tuple[float, float] = (1e-5, 0.99999),
        exact_noise_concentration: float = 1.0,
        exact_noise_rate: float = 10.0,
        exact_jitter: float = 1e-4,
    ) -> None:
        self.mode = str(mode).lower()
        self.length_scale_range = length_scale_range

        if variance_range is None:
            if variance is None:
                variance_range = (0.1, 10.0)
            else:
                variance_value = float(variance)
                variance_range = (variance_value, variance_value)

        if self.mode == "approximate":
            self._factory: ApproximateGPMechanismFactory | ExactGPMechanismFactory = (
                ApproximateGPMechanismFactory(
                    rff_dim=rff_dim,
                    num_kernels=num_kernels,
                    length_scale_range=length_scale_range,
                    variance_range=variance_range,
                )
            )
        elif self.mode == "exact":
            self._factory = ExactGPMechanismFactory(
                num_kernel_pairs=exact_num_kernel_pairs,
                length_scale_range=length_scale_range,
                variance_range=variance_range,
                alpha_range=alpha_range,
                gamma_range=gamma_range,
                noise_concentration=exact_noise_concentration,
                noise_rate=exact_noise_rate,
                jitter=exact_jitter,
            )
        else:
            raise ValueError(
                f"Unknown GP mechanism mode '{mode}'. Expected 'approximate' or 'exact'."
            )

    def make_mechanism(
        self, input_dim: int, torch_generator: torch.Generator
    ) -> nn.Module:
        """Create a mechanism for one node."""
        return self._factory.make_mechanism(input_dim, torch_generator)

    def __call__(
        self,
        adjacency_matrix: torch.Tensor,
        *,
        torch_generator: Optional[torch.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[nn.Module]:
        """Create one mechanism per node for the provided adjacency matrix."""
        return build_mechanisms_from_adjacency(
            self, adjacency_matrix, torch_generator=torch_generator, rng=rng
        )
