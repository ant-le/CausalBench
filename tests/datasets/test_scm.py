import networkx as nx
import pytest
import torch

from causal_meta.datasets.noise import create_noise_sampler
from causal_meta.datasets.scm import SCMFamily, SCMInstance
from causal_meta.datasets.generators.graphs import ErdosRenyiGenerator
from causal_meta.datasets.generators.mechanisms import LinearMechanismFactory


def test_scm_family_generates_dag() -> None:
    family = SCMFamily(
        name="test_dag",
        n_nodes=4,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.5),
        mechanism_factory=LinearMechanismFactory(),
    )

    instance = family.sample_task(seed=21)
    adjacency = instance.adjacency_matrix
    graph = nx.from_numpy_array(adjacency.numpy(), create_using=nx.DiGraph)

    assert adjacency.shape == (4, 4)
    assert nx.is_directed_acyclic_graph(graph)


def test_ancestral_sampling_returns_finite_values() -> None:
    family = SCMFamily(
        name="test_sampling",
        n_nodes=3,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.6),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.1),
    )
    instance = family.sample_task(seed=3)

    samples = instance.sample(num_samples=15)

    assert samples.shape == (15, 3)
    assert torch.isfinite(samples).all()


def test_scm_family_sample_graph_matches_sample_task_adjacency() -> None:
    family = SCMFamily(
        name="test_graph_match",
        n_nodes=4,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.5),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.1),
    )
    seed = 123
    adj_only = family.sample_graph(seed)
    instance = family.sample_task(seed)
    assert torch.equal(adj_only, instance.adjacency_matrix)


# ── Noise distribution integration ────────────────────────────────────


def test_scm_instance_custom_noise_sampler() -> None:
    """SCMInstance accepts a custom noise sampler and produces valid output."""
    family = SCMFamily(
        name="test_custom_noise",
        n_nodes=3,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.6),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.1),
    )
    task = family.sample_task(seed=7)
    # Build a new instance with laplace noise sharing the same graph/mechanisms.
    laplace_instance = SCMInstance(
        adjacency_matrix=task.adjacency_matrix,
        mechanisms=task.mechanisms,
        noise_sampler=create_noise_sampler("laplace"),
    )
    samples = laplace_instance.sample(num_samples=20)
    assert samples.shape == (20, 3)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("noise_type", ["gaussian", "laplace", "uniform"])
def test_scm_family_noise_type(noise_type: str) -> None:
    """SCMFamily with explicit noise_type produces finite samples."""
    family = SCMFamily(
        name=f"test_{noise_type}",
        n_nodes=3,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.6),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.1),
        noise_type=noise_type,
    )
    instance = family.sample_task(seed=42)
    samples = instance.sample(num_samples=50)
    assert samples.shape == (50, 3)
    assert torch.isfinite(samples).all()


def test_scm_family_bad_noise_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown noise type"):
        SCMFamily(
            name="bad_noise",
            n_nodes=3,
            graph_generator=ErdosRenyiGenerator(edge_prob=0.5),
            mechanism_factory=LinearMechanismFactory(),
            noise_type="cauchy",
        )


def test_do_preserves_noise_sampler() -> None:
    """The do() operator should propagate the noise sampler."""
    family = SCMFamily(
        name="test_do_noise",
        n_nodes=4,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.5),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.1),
        noise_type="laplace",
    )
    instance = family.sample_task(seed=10)
    intervened = instance.do({0: 1.0})
    samples = intervened.sample(num_samples=10)
    assert samples.shape == (10, 4)
    assert torch.isfinite(samples).all()
    # Check intervention took effect.
    assert (samples[:, 0] == 1.0).all()
