import torch

from causal_meta.datasets.utils.collate import collate_fn_interventional


def _make_item(
    *,
    seed: int,
    obs_x: torch.Tensor,
    interventions_x: list[torch.Tensor],
    intervention_value: float = 0.0,
) -> dict:
    n_nodes = int(obs_x.shape[1])
    interventions = []
    for target in range(n_nodes):
        interventions.append(
            {
                "target": target,
                "value": intervention_value,
                "data": interventions_x[target],
                "adjacency": torch.zeros(n_nodes, n_nodes),
            }
        )

    return {
        "seed": seed,
        "observational": {"data": obs_x, "adjacency": torch.ones(n_nodes, n_nodes)},
        "interventions": interventions,
    }


def test_collate_fn_interventional_stacks_and_normalizes_per_task() -> None:
    # Two tasks, 3 samples, 2 nodes.
    obs_x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    obs_x2 = torch.tensor([[2.0, 1.0], [4.0, 2.0], [6.0, 3.0]])

    item1 = _make_item(
        seed=11,
        obs_x=obs_x1,
        interventions_x=[obs_x1 + 10.0, obs_x1 + 20.0],
        intervention_value=0.0,
    )
    item2 = _make_item(
        seed=22,
        obs_x=obs_x2,
        interventions_x=[obs_x2 + 1.0, obs_x2 + 2.0],
        intervention_value=1.0,
    )

    out = collate_fn_interventional([item1, item2])

    assert out["seed"].tolist() == [11, 22]

    obs = out["observational"]
    ints = out["interventions"]

    assert obs["data"].shape == (2, 3, 2)
    assert obs["adjacency"].shape == (2, 2, 2)
    assert ints["data"].shape == (2, 2, 3, 2)
    assert ints["adjacency"].shape == (2, 2, 2, 2)
    assert ints["target"].shape == (2, 2)
    assert ints["value"].shape == (2, 2)

    # Observational normalization is per-task.
    for i in range(2):
        x = obs["data"][i]
        assert torch.allclose(x.mean(dim=0), torch.zeros(2), atol=1e-6)
        assert torch.allclose(x.std(dim=0, unbiased=False), torch.ones(2), atol=1e-6)

    # Interventions use the *observational* stats for each task.
    mean1 = obs_x1.mean(dim=0, keepdim=True)
    std1 = obs_x1.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    expected_int0_task1 = ((obs_x1 + 10.0) - mean1) / std1
    assert torch.allclose(ints["data"][0, 0], expected_int0_task1, atol=1e-6)

    mean2 = obs_x2.mean(dim=0, keepdim=True)
    std2 = obs_x2.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    expected_int1_task2 = ((obs_x2 + 2.0) - mean2) / std2
    assert torch.allclose(ints["data"][1, 1], expected_int1_task2, atol=1e-6)

    assert ints["target"].tolist() == [[0, 1], [0, 1]]
    assert ints["value"].tolist() == [[0.0, 0.0], [1.0, 1.0]]
