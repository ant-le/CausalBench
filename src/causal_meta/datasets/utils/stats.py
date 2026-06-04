from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List

import numpy as np
import torch
from scipy import stats

if TYPE_CHECKING:
    from causal_meta.datasets.scm import SCMFamily


MECHANISM_DISTANCE_OBS_SAMPLES = 256
"""Number of observational samples per task for mechanism distance."""


def _linear_r2(y: np.ndarray, x: np.ndarray) -> float:
    """Fit a linear model and return a clipped R^2 score in [0, 1].

    Returns 0.0 when the data contains non-finite values or the SVD
    inside ``lstsq`` fails to converge (e.g. chaotic mechanisms).
    """
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if y.shape[0] != x.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if y.shape[0] < 2:
        return 1.0

    x_design = np.concatenate(
        [np.ones((x.shape[0], 1), dtype=np.float64), x.astype(np.float64)],
        axis=1,
    )
    y64 = y.astype(np.float64)

    # Non-finite values (NaN/Inf from chaotic mechanisms) make SVD diverge.
    if not (np.isfinite(x_design).all() and np.isfinite(y64).all()):
        return 0.0

    try:
        beta, *_ = np.linalg.lstsq(x_design, y64, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    y_hat = x_design @ beta

    ss_tot = float(np.sum((y64 - y64.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 1.0

    ss_res = float(np.sum((y64 - y_hat) ** 2))
    r2 = 1.0 - (ss_res / ss_tot)
    return float(np.clip(r2, 0.0, 1.0))


def _task_nonlinearity_score(
    family: "SCMFamily",
    *,
    seed: int,
    obs_samples: int,
) -> float:
    """Estimate nonlinearity for one sampled task via linear-fit residual gap.

    The score is the mean ``1 - R^2`` across nodes with at least one parent.
    Higher values indicate stronger deviation from linear parent-child relations.
    """
    instance = family.sample_task(seed)
    adjacency = instance.adjacency_matrix.detach().cpu()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 13_579)
        samples = instance.sample(obs_samples).detach().cpu().numpy()

    node_scores: List[float] = []
    n_nodes = int(adjacency.shape[0])
    for node in range(n_nodes):
        parents = torch.nonzero(adjacency[:, node] > 0.5, as_tuple=False).flatten()
        if parents.numel() == 0:
            continue
        x = samples[:, parents.tolist()]
        y = samples[:, node]
        r2 = _linear_r2(y, x)
        node_scores.append(float(1.0 - r2))

    if not node_scores:
        return 0.0
    return float(np.mean(node_scores))


def degree_sequence(adjacency: torch.Tensor) -> torch.Tensor:
    """Return total degree (in + out) for each node."""
    adj = adjacency.float()
    return adj.sum(dim=0) + adj.sum(dim=1)


def spectral_radius(adjacency: torch.Tensor) -> float:
    """Compute spectral radius using a symmetrized adjacency."""
    sym_adj = (adjacency + adjacency.T) / 2.0
    eigenvalues = torch.linalg.eigvalsh(sym_adj)
    return float(eigenvalues.abs().max().item())


def degree_histogram(degrees: Iterable[float]) -> np.ndarray:
    values = list(degrees)
    if not values:
        return np.array([1.0], dtype=float)
    max_degree = max(1, int(max(values)))
    hist, _ = np.histogram(values, bins=np.arange(0, max_degree + 2), density=True)
    hist = np.clip(hist, 1e-12, None)
    return hist / hist.sum()


def get_family_stats(family: SCMFamily, n_samples: int = 100) -> Dict[str, float]:
    """Compute average graph statistics for a family."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    avg_degrees: List[float] = []
    sparsities: List[float] = []
    spectral_radii: List[float] = []

    for seed in range(n_samples):
        instance = family.sample_task(seed)
        adjacency = instance.adjacency_matrix.detach().cpu()
        degrees = degree_sequence(adjacency)
        avg_degrees.append(float(degrees.mean().item()))

        n_nodes = adjacency.shape[0]
        possible_edges = n_nodes * (n_nodes - 1) / 2
        sparsity = (
            float(adjacency.sum().item() / possible_edges)
            if possible_edges > 0
            else 0.0
        )
        sparsities.append(sparsity)

        spectral_radii.append(spectral_radius(adjacency))

    return {
        "avg_degree": float(np.mean(avg_degrees)) if avg_degrees else 0.0,
        "sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
        "spectral_radius": float(np.mean(spectral_radii)) if spectral_radii else 0.0,
    }


def compute_family_distance(
    family_a: SCMFamily,
    family_b: SCMFamily,
    metric: str = "spectral",
    n_samples: int = 50,
) -> float:
    """Compute a distance between two SCM families.

    Supported metrics:
    - ``"spectral"``: L1 gap in average graph spectral radius.
    - ``"kl"``: KL divergence between degree distributions.
    - ``"mechanism"``: Absolute gap in average nonlinearity score
      (mean ``1-R^2`` of linear regressions over parent-child relations).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    metric = metric.lower()
    if metric == "spectral":
        stats_a = get_family_stats(family_a, n_samples=n_samples)
        stats_b = get_family_stats(family_b, n_samples=n_samples)
        return abs(stats_a["spectral_radius"] - stats_b["spectral_radius"])
    if metric == "kl":
        degrees_a: List[float] = []
        degrees_b: List[float] = []
        for seed in range(n_samples):
            adj_a = family_a.sample_task(seed).adjacency_matrix.detach().cpu()
            adj_b = family_b.sample_task(seed).adjacency_matrix.detach().cpu()
            degrees_a.extend(degree_sequence(adj_a).tolist())
            degrees_b.extend(degree_sequence(adj_b).tolist())

        p = degree_histogram(degrees_a)
        q = degree_histogram(degrees_b)
        # Align histogram lengths
        if p.shape[0] < q.shape[0]:
            p = np.pad(p, (0, q.shape[0] - p.shape[0]), constant_values=1e-12)
        elif q.shape[0] < p.shape[0]:
            q = np.pad(q, (0, p.shape[0] - q.shape[0]), constant_values=1e-12)
        return float(stats.entropy(p, q))
    if metric == "mechanism":
        scores_a: List[float] = []
        scores_b: List[float] = []
        for seed in range(n_samples):
            scores_a.append(
                _task_nonlinearity_score(
                    family_a,
                    seed=seed,
                    obs_samples=MECHANISM_DISTANCE_OBS_SAMPLES,
                )
            )
            scores_b.append(
                _task_nonlinearity_score(
                    family_b,
                    seed=seed,
                    obs_samples=MECHANISM_DISTANCE_OBS_SAMPLES,
                )
            )
        return abs(float(np.mean(scores_a)) - float(np.mean(scores_b)))

    raise ValueError(f"Unsupported metric: {metric}")
