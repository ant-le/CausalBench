from __future__ import annotations

from unittest.mock import patch

import torch

from causal_meta.runners.metrics.graph import Metrics


def test_metrics_handler_update_and_compute() -> None:
    """Stateful update + compute (one-shot mode removed)."""
    metrics = Metrics(metrics=["e-shd", "e-edgef1"])
    target = torch.tensor([[[0, 1], [0, 0]]]).float()
    samples = torch.tensor([[[[0, 1], [0, 0]]], [[[0, 0], [0, 0]]]]).float()

    metrics.update(target, samples)
    results = metrics.compute(summary_stats=False)

    assert "e-shd" in results
    assert "e-edgef1" in results
    assert results["e-shd"] == 0.5
    assert results["e-edgef1"] == 0.5


@patch("torch.distributed.is_available")
@patch("torch.distributed.is_initialized")
@patch("torch.distributed.get_world_size")
@patch("torch.distributed.all_gather_object")
def test_metrics_gather_distributed(
    mock_all_gather, mock_world_size, mock_init, mock_avail
) -> None:
    mock_avail.return_value = True
    mock_init.return_value = True
    mock_world_size.return_value = 2

    rank0_data = {"e-shd": [0.5, 1.0]}
    rank1_data = {"e-shd": [1.5, 2.0]}

    def side_effect(gathered_list, local_obj):
        _ = local_obj
        gathered_list[0] = rank0_data
        gathered_list[1] = rank1_data
        return None

    mock_all_gather.side_effect = side_effect

    metrics_handler = Metrics()
    # Manually set history so gather_raw_results has something
    metrics_handler.history["e-shd"] = [0.5, 1.0]
    gathered = metrics_handler.gather_raw_results()

    assert gathered["e-shd"] == [0.5, 1.0, 1.5, 2.0]
