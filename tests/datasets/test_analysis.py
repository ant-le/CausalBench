import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from causal_meta.datasets.utils import (
    compute_family_distance,
    get_family_stats,
    plot_degree_distribution,
    visualize_adjacency,
)
from causal_meta.datasets.generators.graphs import ErdosRenyiGenerator
from causal_meta.datasets.generators.mechanisms import (
    LinearMechanismFactory,
    MLPMechanismFactory,
)
from causal_meta.datasets.scm import SCMFamily


def _build_family(edge_prob: float = 0.3, n_nodes: int = 5) -> SCMFamily:
    generator = ErdosRenyiGenerator(edge_prob=edge_prob)
    mechanism = LinearMechanismFactory(weight_scale=0.1)
    return SCMFamily(
        name="test_family",
        n_nodes=n_nodes,
        graph_generator=generator,
        mechanism_factory=mechanism,
    )


def _build_mlp_family(edge_prob: float = 0.3, n_nodes: int = 5) -> SCMFamily:
    generator = ErdosRenyiGenerator(edge_prob=edge_prob)
    mechanism = MLPMechanismFactory(hidden_dim=16)
    return SCMFamily(
        name="test_mlp_family",
        n_nodes=n_nodes,
        graph_generator=generator,
        mechanism_factory=mechanism,
    )


def test_get_family_stats_returns_expected_keys() -> None:
    stats = get_family_stats(_build_family(), n_samples=5)
    assert set(["avg_degree", "sparsity", "spectral_radius"]).issubset(stats.keys())
    assert stats["avg_degree"] >= 0
    assert 0 <= stats["sparsity"] <= 1
    assert stats["spectral_radius"] >= 0


def test_plot_degree_distribution_returns_figure() -> None:
    fig = plot_degree_distribution(_build_family(), n_samples=4)
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    assert len(ax.patches) > 0
    plt.close(fig)


def test_visualize_adjacency_renders_matrix() -> None:
    adjacency = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    fig = visualize_adjacency(adjacency)
    ax = fig.axes[0]
    assert ax.images[0].get_array().shape == adjacency.shape
    plt.close(fig)


def test_compute_family_distance_spectral_differs_for_density() -> None:
    sparse_family = _build_family(edge_prob=0.1)
    dense_family = _build_family(edge_prob=0.8)
    distance = compute_family_distance(
        sparse_family, dense_family, metric="spectral", n_samples=6
    )
    assert distance >= 0
    assert distance > 0.01


def test_compute_family_distance_mechanism_detects_nonlinearity() -> None:
    linear_family = _build_family(edge_prob=0.3)
    mlp_family = _build_mlp_family(edge_prob=0.3)
    distance = compute_family_distance(
        linear_family,
        mlp_family,
        metric="mechanism",
        n_samples=6,
    )
    assert distance >= 0.0
