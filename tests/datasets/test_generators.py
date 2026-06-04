import networkx as nx
import torch

from causal_meta.datasets.generators.graphs import (
    ErdosRenyiGenerator,
    GeometricRandomGenerator,
    SBMGenerator,
    ScaleFreeGenerator,
    WattsStrogatzGenerator,
)
from causal_meta.datasets.generators.mechanisms import (
    GPMechanismFactory,
    LinearMechanismFactory,
    LogisticMapMechanismFactory,
    MLPMechanismFactory,
)
from causal_meta.datasets.scm import SCMFamily


def test_scm_instance_sampling_shape() -> None:
    generator = SBMGenerator(n_blocks=2, p_intra=0.8, p_inter=0.1)
    factory = MLPMechanismFactory(hidden_dim=8)
    family = SCMFamily(
        name="test_sbm", n_nodes=6, graph_generator=generator, mechanism_factory=factory
    )

    instance = family.sample_task(seed=7)
    samples = instance.sample(num_samples=5)

    assert samples.shape == (5, 6)
    assert torch.isfinite(samples).all()


def test_sample_task_seed_reproducibility() -> None:
    generator = ErdosRenyiGenerator(edge_prob=0.3)
    factory = LinearMechanismFactory(weight_scale=0.5)
    family = SCMFamily(
        name="test_er", n_nodes=5, graph_generator=generator, mechanism_factory=factory
    )

    instance_a = family.sample_task(seed=123)
    instance_b = family.sample_task(seed=123)

    assert torch.equal(instance_a.adjacency_matrix, instance_b.adjacency_matrix)

    for mech_a, mech_b in zip(instance_a.mechanisms, instance_b.mechanisms):
        for key, tensor_a in mech_a.state_dict().items():
            tensor_b = mech_b.state_dict()[key]
            assert torch.allclose(tensor_a, tensor_b)


def test_gp_mechanism_task_seed_reproducibility() -> None:
    generator = ErdosRenyiGenerator(edge_prob=0.3)
    factory = GPMechanismFactory(
        mode="approximate",
        rff_dim=32,
        num_kernels=2,
        length_scale_range=(0.5, 1.0),
        variance=1.0,
    )
    family = SCMFamily(
        name="test_gp", n_nodes=5, graph_generator=generator, mechanism_factory=factory
    )

    instance_a = family.sample_task(seed=123)
    instance_b = family.sample_task(seed=123)

    assert torch.equal(instance_a.adjacency_matrix, instance_b.adjacency_matrix)

    for mech_a, mech_b in zip(instance_a.mechanisms, instance_b.mechanisms):
        for key, tensor_a in mech_a.state_dict().items():
            tensor_b = mech_b.state_dict()[key]
            assert torch.allclose(tensor_a, tensor_b)


def test_exact_gp_mechanism_task_seed_reproducibility() -> None:
    generator = ErdosRenyiGenerator(edge_prob=0.3)
    factory = GPMechanismFactory(
        mode="exact",
        exact_num_kernel_pairs=1,
        length_scale_range=(0.5, 1.0),
        variance_range=(0.5, 1.0),
        alpha_range=(0.2, 0.8),
        gamma_range=(0.2, 0.8),
        exact_noise_concentration=1.0,
        exact_noise_rate=10.0,
    )
    family = SCMFamily(
        name="test_exact_gp",
        n_nodes=5,
        graph_generator=generator,
        mechanism_factory=factory,
    )

    instance_a = family.sample_task(seed=42)
    instance_b = family.sample_task(seed=42)

    assert torch.equal(instance_a.adjacency_matrix, instance_b.adjacency_matrix)
    for mech_a, mech_b in zip(instance_a.mechanisms, instance_b.mechanisms):
        for key, tensor_a in mech_a.state_dict().items():
            tensor_b = mech_b.state_dict()[key]
            assert torch.allclose(tensor_a, tensor_b)


def test_exact_gp_mechanism_produces_finite_samples() -> None:
    generator = ErdosRenyiGenerator(edge_prob=0.4)
    factory = GPMechanismFactory(
        mode="exact",
        exact_num_kernel_pairs=1,
        length_scale_range=(0.5, 1.5),
        variance_range=(0.5, 1.5),
        alpha_range=(0.2, 0.9),
        gamma_range=(0.2, 0.9),
    )
    family = SCMFamily(
        name="test_exact_gp_finite",
        n_nodes=4,
        graph_generator=generator,
        mechanism_factory=factory,
    )
    instance = family.sample_task(seed=17)
    samples = instance.sample(num_samples=12)

    assert samples.shape == (12, 4)
    assert torch.isfinite(samples).all()


def test_logistic_map_root_nodes_are_stochastic_and_bounded() -> None:
    factory = LogisticMapMechanismFactory(noise_scale=1.0)
    mechanism = factory.make_mechanism(0, torch.Generator().manual_seed(11))
    parents = torch.zeros((512, 0), dtype=torch.float32)
    noise = torch.randn((512, 1), generator=torch.Generator().manual_seed(12))

    values = mechanism(parents, noise)

    assert values.min().item() >= 0.0
    assert values.max().item() <= 1.0
    assert values.std().item() > 0.01


def test_erdos_renyi_density() -> None:
    n_nodes = 80
    edge_prob = 0.2
    generator = ErdosRenyiGenerator(edge_prob=edge_prob)

    adjacency = generator(n_nodes, seed=0)
    possible_edges = n_nodes * (n_nodes - 1) / 2
    density = adjacency.sum().item() / possible_edges

    assert abs(density - edge_prob) < 0.03


def test_erdos_renyi_randomizes_node_order() -> None:
    generator = ErdosRenyiGenerator(edge_prob=1.0)
    n_nodes = 8

    non_upper_count = 0
    for seed in range(10):
        adjacency = generator(n_nodes=n_nodes, seed=seed)
        if not torch.equal(adjacency, torch.triu(adjacency, diagonal=1)):
            non_upper_count += 1

    assert non_upper_count > 0


def test_scale_free_connected() -> None:
    generator = ScaleFreeGenerator(m=2)
    adjacency = generator(n_nodes=20, seed=99)

    undirected = nx.from_numpy_array(adjacency.numpy(), create_using=nx.Graph)
    assert nx.is_connected(undirected)


# ── Watts-Strogatz tests ──


def test_watts_strogatz_produces_dag() -> None:
    generator = WattsStrogatzGenerator(k=4, p=0.3)
    adjacency = generator(n_nodes=20, seed=42)

    assert adjacency.shape == (20, 20)
    assert adjacency.dtype == torch.float32
    # No self-loops
    assert adjacency.diag().sum().item() == 0.0
    # DAG check: the oriented graph should be a DAG
    graph = nx.from_numpy_array(adjacency.numpy(), create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(graph)


def test_watts_strogatz_seed_reproducibility() -> None:
    generator = WattsStrogatzGenerator(k=4, p=0.5)
    adj_a = generator(n_nodes=15, seed=7)
    adj_b = generator(n_nodes=15, seed=7)
    assert torch.equal(adj_a, adj_b)


def test_watts_strogatz_different_seeds_differ() -> None:
    generator = WattsStrogatzGenerator(k=4, p=0.5)
    adj_a = generator(n_nodes=15, seed=0)
    adj_b = generator(n_nodes=15, seed=1)
    assert not torch.equal(adj_a, adj_b)


def test_watts_strogatz_has_edges() -> None:
    generator = WattsStrogatzGenerator(k=4, p=0.3)
    adjacency = generator(n_nodes=20, seed=0)
    assert adjacency.sum().item() > 0


# ── Geometric Random Graph tests ──


def test_geometric_random_produces_dag() -> None:
    generator = GeometricRandomGenerator(radius=0.5, dim=2)
    adjacency = generator(n_nodes=20, seed=42)

    assert adjacency.shape == (20, 20)
    assert adjacency.dtype == torch.float32
    # No self-loops
    assert adjacency.diag().sum().item() == 0.0
    # DAG check
    graph = nx.from_numpy_array(adjacency.numpy(), create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(graph)


def test_geometric_random_seed_reproducibility() -> None:
    generator = GeometricRandomGenerator(radius=0.4, dim=2)
    adj_a = generator(n_nodes=15, seed=7)
    adj_b = generator(n_nodes=15, seed=7)
    assert torch.equal(adj_a, adj_b)


def test_geometric_random_different_seeds_differ() -> None:
    generator = GeometricRandomGenerator(radius=0.4, dim=2)
    adj_a = generator(n_nodes=15, seed=0)
    adj_b = generator(n_nodes=15, seed=1)
    assert not torch.equal(adj_a, adj_b)


def test_geometric_random_has_edges() -> None:
    generator = GeometricRandomGenerator(radius=0.5, dim=2)
    adjacency = generator(n_nodes=20, seed=0)
    assert adjacency.sum().item() > 0


def test_geometric_random_higher_dim() -> None:
    generator = GeometricRandomGenerator(radius=0.6, dim=3)
    adjacency = generator(n_nodes=15, seed=5)

    assert adjacency.shape == (15, 15)
    graph = nx.from_numpy_array(adjacency.numpy(), create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(graph)
