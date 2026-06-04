from __future__ import annotations

import torch

from causal_meta.models.random.model import RandomModel


def _is_acyclic(adjacency: torch.Tensor) -> bool:
    """Check DAG acyclicity via nilpotency of adjacency powers."""
    n_nodes = int(adjacency.shape[0])
    mat = adjacency.to(dtype=torch.float32)
    power = mat.clone()
    trace_sum = torch.trace(power)
    for _ in range(1, n_nodes):
        power = power @ mat
        trace_sum = trace_sum + torch.trace(power)
    return bool(torch.isclose(trace_sum, torch.tensor(0.0), atol=1e-5).item())


def test_random_model_sample_shape_and_acyclicity() -> None:
    model = RandomModel(num_nodes=6, p_edge=0.25)
    x = torch.zeros(3, 8, 6)

    samples = model.sample(x, num_samples=7)

    assert samples.shape == (3, 7, 6, 6)
    assert torch.allclose(
        torch.diagonal(samples, dim1=-2, dim2=-1), torch.zeros(3, 7, 6)
    )
    for batch_idx in range(samples.shape[0]):
        for sample_idx in range(samples.shape[1]):
            assert _is_acyclic(samples[batch_idx, sample_idx])


def test_random_model_matches_edge_probability_in_expectation() -> None:
    p_edge = 0.2
    n_nodes = 10
    model = RandomModel(num_nodes=n_nodes, p_edge=p_edge)
    x = torch.zeros(12, 5, n_nodes)

    samples = model.sample(x, num_samples=64)
    max_edges = n_nodes * (n_nodes - 1) / 2
    edge_fraction = float(samples.sum().item()) / (
        samples.shape[0] * samples.shape[1] * max_edges
    )
    assert abs(edge_fraction - p_edge) < 0.05


def test_random_model_has_no_pretraining() -> None:
    model = RandomModel(num_nodes=4, p_edge=0.3)
    assert model.needs_pretraining is False
