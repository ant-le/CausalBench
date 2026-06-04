from __future__ import annotations

import torch
import pytest

from causal_meta.datasets.generators.graphs.er import ErdosRenyiGenerator
from causal_meta.datasets.generators.mechanisms.linear import LinearMechanismFactory
from causal_meta.datasets.scm import SCMFamily
from causal_meta.runners.metrics.scm import SCMMetrics
from causal_meta.runners.utils.scoring import LinearGaussianScorer


def test_linear_gaussian_scorer_fit_and_score_nll() -> None:
    torch.manual_seed(0)
    n = 64

    x0 = torch.randn(n)
    x1 = 2.0 * x0 + 0.1 * torch.randn(n)
    data = torch.stack([x0, x1], dim=1)

    adjacency = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    scorer = LinearGaussianScorer(adjacency=adjacency, obs_data=data)
    scorer.fit()

    nll = scorer.score_nll(data)
    nll_per_node = scorer.score_nll_per_node(data)
    assert torch.isfinite(torch.tensor(nll))
    assert torch.isfinite(torch.tensor(nll_per_node))
    assert nll >= 0.0
    assert nll_per_node >= 0.0


def test_linear_gaussian_scorer_score_nll_per_node_is_size_comparable() -> None:
    torch.manual_seed(0)
    n = 64

    x0 = torch.randn(n)
    x1 = 2.0 * x0 + 0.1 * torch.randn(n)
    data_small = torch.stack([x0, x1], dim=1)
    data_large = torch.stack([x0, x1, x0, x1], dim=1)

    adjacency_small = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    adjacency_large = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    scorer_small = LinearGaussianScorer(adjacency=adjacency_small, obs_data=data_small)
    scorer_large = LinearGaussianScorer(adjacency=adjacency_large, obs_data=data_large)
    scorer_small.fit()
    scorer_large.fit()

    nll_small = scorer_small.score_nll(data_small)
    nll_large = scorer_large.score_nll(data_large)
    nll_per_node_small = scorer_small.score_nll_per_node(data_small)
    nll_per_node_large = scorer_large.score_nll_per_node(data_large)

    assert nll_large == pytest.approx(2.0 * nll_small, rel=0.05)
    assert nll_per_node_large == pytest.approx(nll_per_node_small, rel=0.05)


def test_scm_metrics_inil_with_family_generated_interventions() -> None:
    family = SCMFamily(
        name="test_family",
        n_nodes=3,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.4),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.5),
    )
    seed = 13

    instance = family.sample_task(seed)
    obs_data = instance.sample(32)
    graph_samples = instance.adjacency_matrix.unsqueeze(0)

    metrics = SCMMetrics(metrics=["inil", "inil_per_node"])
    metrics.update(
        obs_data=obs_data,
        graph_samples=graph_samples,
        family=family,
        seeds=[seed],
    )

    summary = metrics.compute(summary_stats=False)
    assert "inil" in summary
    assert "inil_per_node" in summary
    assert torch.isfinite(torch.tensor(summary["inil"]))
    assert torch.isfinite(torch.tensor(summary["inil_per_node"]))


def test_scm_metrics_inil_with_precomputed_interventional_data() -> None:
    family = SCMFamily(
        name="test_family",
        n_nodes=3,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.3),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.4),
    )
    seed = 9
    instance = family.sample_task(seed)

    obs_data = instance.sample(24)
    graph_samples = instance.adjacency_matrix.unsqueeze(0)
    interventional_data = SCMMetrics._generate_interventional_data(
        family=family,
        seed=seed,
        n_samples=24,
    )

    metrics = SCMMetrics(metrics=["inil", "inil_per_node"])
    metrics.update(
        obs_data=obs_data,
        graph_samples=graph_samples,
        interventional_data=interventional_data,
    )

    raw = metrics.get_raw_results()
    assert "inil" in raw
    assert "inil_per_node" in raw
    assert len(raw["inil"]) == 1
    assert len(raw["inil_per_node"]) == 1
    assert torch.isfinite(torch.tensor(raw["inil"][0]))
    assert torch.isfinite(torch.tensor(raw["inil_per_node"][0]))
