import math

import networkx as nx
import torch

from causal_meta.datasets.generators.graphs import ErdosRenyiGenerator
from causal_meta.datasets.generators.mechanisms import LinearMechanismFactory
from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.torch_datasets import MetaFixedDataset
from causal_meta.datasets.utils import collate_fn_scm
from causal_meta.models.factory import ModelFactory

# Ensure models are registered
import causal_meta.models  # noqa: F401


def _make_dataset_batch(n_nodes: int = 5, samples_per_task: int = 32, seed: int = 0):
    family = SCMFamily(
        name="test_integration",
        n_nodes=n_nodes,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.4),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.1),
    )
    dataset = MetaFixedDataset(family, seeds=[seed], samples_per_task=samples_per_task)
    item = dataset[0]
    out = collate_fn_scm([item])
    return out["data"], out["adjacency"]


def _is_dag(adjacency: torch.Tensor) -> bool:
    graph = nx.from_numpy_array(
        adjacency.detach().cpu().numpy(), create_using=nx.DiGraph
    )
    return nx.is_directed_acyclic_graph(graph)


def test_avici_model_runs_on_dataset_batch() -> None:
    x, adj = _make_dataset_batch(n_nodes=6, samples_per_task=16, seed=123)
    model = ModelFactory.create(
        {"type": "avici", "num_nodes": 6, "d_model": 16, "nhead": 2, "num_layers": 2}
    )

    logits = model(x)
    assert logits.shape == (1, 6, 6)
    assert torch.isfinite(logits).all()

    loss = torch.nn.BCEWithLogitsLoss()(logits.view(1, -1), adj.view(1, -1))
    assert torch.isfinite(loss).all()

    samples = model.sample(x, num_samples=4)
    assert samples.shape == (1, 4, 6, 6)
    diag = torch.diagonal(samples, dim1=-2, dim2=-1)
    assert torch.all(diag == 0)
    assert set(samples.unique().tolist()).issubset({0.0, 1.0})


def test_bcnp_model_runs_on_dataset_batch_and_samples_dags() -> None:
    x, adj = _make_dataset_batch(n_nodes=6, samples_per_task=16, seed=123)
    model = ModelFactory.create(
        {
            "type": "bcnp",
            "num_nodes": 6,
            "d_model": 16,
            "nhead": 2,
            "num_layers": 2,
            "n_perm_samples": 4,
            "sinkhorn_iter": 10,
        }
    )

    all_probs = model(x)
    assert all_probs.shape == (4, 1, 6, 6)
    assert torch.isfinite(all_probs).all()
    assert all_probs.min().item() >= 0.0
    assert all_probs.max().item() <= 1.0
    diag_probs = torch.diagonal(all_probs, dim1=-2, dim2=-1)
    assert torch.all(diag_probs == 0)

    probs_flat = all_probs.reshape(all_probs.size(0), all_probs.size(1), -1).clamp(
        1e-6, 1 - 1e-6
    )
    target_flat = adj.reshape(adj.size(0), -1)
    dist = torch.distributions.Bernoulli(probs=probs_flat)
    log_prob = dist.log_prob(target_flat.unsqueeze(0))
    log_prob_sum = torch.logsumexp(log_prob, dim=0) - math.log(log_prob.size(0))
    loss = (-log_prob_sum).mean()
    assert torch.isfinite(loss).all()

    samples = model.sample(x, num_samples=5)
    assert samples.shape == (1, 5, 6, 6)
    diag = torch.diagonal(samples, dim1=-2, dim2=-1)
    assert torch.all(diag == 0)
    assert set(samples.unique().tolist()).issubset({0.0, 1.0})

    for sample in samples[0]:
        assert _is_dag(sample)


def test_avici_and_bcnp_accept_two_channel_inputs() -> None:
    x, _ = _make_dataset_batch(n_nodes=5, samples_per_task=12, seed=7)
    intervention_mask = torch.zeros_like(x)
    intervention_mask[:, -4:, 2] = 1.0
    x_two_channel = torch.stack([x, intervention_mask], dim=-1)

    avici = ModelFactory.create(
        {"type": "avici", "num_nodes": 5, "d_model": 16, "nhead": 2, "num_layers": 2}
    )
    bcnp = ModelFactory.create(
        {
            "type": "bcnp",
            "num_nodes": 5,
            "d_model": 16,
            "nhead": 2,
            "num_layers": 2,
            "n_perm_samples": 3,
            "sinkhorn_iter": 10,
        }
    )

    avici_logits = avici(x_two_channel)
    bcnp_probs = bcnp(x_two_channel)

    assert avici_logits.shape == (1, 5, 5)
    assert bcnp_probs.shape == (3, 1, 5, 5)
    assert torch.isfinite(avici_logits).all()
    assert torch.isfinite(bcnp_probs).all()
