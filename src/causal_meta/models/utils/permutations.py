from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def sample_gumbel(
    shape: Tuple[int, ...],
    *,
    device: torch.device,
    eps: float = 1e-20,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample standard Gumbel noise."""
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + eps) + eps)


def sinkhorn(
    log_alpha: torch.Tensor, n_iters: int = 20, tol: float = 1e-6
) -> torch.Tensor:
    """Incomplete Sinkhorn normalization in log-space.

    Args:
        log_alpha: Tensor of shape (B, N, N) or (N, N).
        n_iters: Number of normalization iterations.
        tol: Early-stop tolerance on row/col sums.

    Returns:
        Doubly-stochastic matrices as a tensor of shape (B, N, N).
    """
    if log_alpha.ndim == 2:
        log_alpha = log_alpha.unsqueeze(0)
    if log_alpha.ndim != 3 or log_alpha.shape[-1] != log_alpha.shape[-2]:
        raise ValueError("log_alpha must have shape (N, N) or (B, N, N).")

    n = int(log_alpha.size(1))
    log_alpha = log_alpha.reshape(-1, n, n)

    for _ in range(int(n_iters)):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

        exp_alpha = torch.exp(log_alpha)
        if (
            torch.abs(1.0 - exp_alpha.sum(-1)).max() < tol
            and torch.abs(1.0 - exp_alpha.sum(-2)).max() < tol
        ):
            break

    return torch.exp(log_alpha)


def _hungarian_matching(matrix_batch: torch.Tensor) -> torch.Tensor:
    """Solve max_P sum_ij M_ij P_ij via Hungarian algorithm.

    Args:
        matrix_batch: Tensor of shape (B, N, N) or (N, N).

    Returns:
        listperms: Long tensor of shape (B, N) with column indices per row.
    """
    if matrix_batch.ndim == 2:
        matrix_batch = matrix_batch.unsqueeze(0)
    if matrix_batch.ndim != 3 or matrix_batch.shape[-1] != matrix_batch.shape[-2]:
        raise ValueError("matrix_batch must have shape (N, N) or (B, N, N).")

    x = matrix_batch.detach().float().cpu().numpy()
    batch_size, n, _ = x.shape
    sol = np.zeros((batch_size, n), dtype=np.int64)
    for i in range(batch_size):
        # linear_sum_assignment minimizes cost; use negative for maximization
        sol[i, :] = linear_sum_assignment(-x[i])[1].astype(np.int64)
    return torch.from_numpy(sol)


def _listperm2matperm(listperm: torch.Tensor) -> torch.Tensor:
    """Convert list permutation (B, N) to permutation matrices (B, N, N)."""
    if listperm.ndim != 2:
        raise ValueError("listperm must have shape (B, N).")
    bsz, n = listperm.shape
    eye = torch.eye(n, dtype=torch.float32)
    # eye[listperm] -> (B, N, N), where row i picks column listperm[b, i]
    return eye[listperm]


def sample_permutation(
    *,
    log_alpha: torch.Tensor,
    temp: float = 1.0,
    n_samples: int = 1,
    noise_factor: float = 1.0,
    n_iters: int = 20,
    squeeze: bool = True,
    hard: bool = False,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample permutation matrices via Gumbel-Sinkhorn.

    This mirrors the reference implementation used by the BCNP paper code.

    Args:
        log_alpha: Tensor of shape (B, N, N).
        temp: Temperature.
        n_samples: Number of permutation samples.
        noise_factor: Gumbel noise scaling.
        n_iters: Sinkhorn iterations.
        squeeze: If True and n_samples == 1, return a 3D tensor (B, N, N).
        hard: If True, apply Hungarian matching with straight-through gradients.
        device: Device for sampling gumbel noise.

    Returns:
        sink: Tensor of shape (B, n_samples, N, N) (or (B, N, N) if squeezed).
        log_alpha_w_noise: Noisy logits divided by temperature.
    """
    if log_alpha.ndim != 3 or log_alpha.shape[-1] != log_alpha.shape[-2]:
        raise ValueError("log_alpha must have shape (B, N, N).")

    n = int(log_alpha.size(1))
    log_alpha = log_alpha.reshape(-1, n, n)
    batch_size = int(log_alpha.size(0))
    dtype = log_alpha.dtype

    if device is None:
        device_obj = log_alpha.device
    elif isinstance(device, torch.device):
        device_obj = device
    else:
        device_obj = torch.device(device)

    # Repeat across samples: (n_samples * B, N, N)
    log_alpha_w_noise = log_alpha.repeat(int(n_samples), 1, 1)

    if float(noise_factor) == 0.0:
        noise = 0.0
    else:
        noise = sample_gumbel(
            (int(n_samples) * batch_size, n, n), device=device_obj, dtype=dtype
        ) * float(noise_factor)

    log_alpha_w_noise = (log_alpha_w_noise + noise) / float(temp)
    sink = sinkhorn(log_alpha_w_noise.clone(), n_iters=int(n_iters))

    if int(n_samples) > 1 or squeeze is False:
        sink = sink.reshape(int(n_samples), batch_size, n, n).transpose(1, 0)
        log_alpha_w_noise = log_alpha_w_noise.reshape(
            int(n_samples), batch_size, n, n
        ).transpose(1, 0)

    if hard:
        # Straight-through Hungarian.
        # Flatten across batch/samples for matching.
        if log_alpha_w_noise.ndim == 4:
            flat = log_alpha_w_noise.transpose(0, 1).reshape(-1, n, n)
            hard_list = _hungarian_matching(flat)
            hard_mat = _listperm2matperm(hard_list).to(device_obj).to(dtype)
            hard_mat = hard_mat.reshape(int(n_samples), batch_size, n, n).transpose(
                1, 0
            )
            sink = hard_mat - sink.detach() + sink
        else:
            hard_list = _hungarian_matching(log_alpha_w_noise)
            hard_mat = _listperm2matperm(hard_list).to(device_obj).to(dtype)
            sink = hard_mat - sink.detach() + sink

    if squeeze and int(n_samples) == 1 and sink.ndim == 4:
        sink = sink[:, 0]
        log_alpha_w_noise = log_alpha_w_noise[:, 0]

    return sink, log_alpha_w_noise
