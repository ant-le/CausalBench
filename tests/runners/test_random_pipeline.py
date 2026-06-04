from __future__ import annotations

import os

from omegaconf import OmegaConf

from causal_meta.main import run_pipeline


def test_random_pipeline_infers_training_sparsity_and_runs(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "name": "random_pipeline_smoke",
            "logger": {"wandb": {"enabled": False}},
            "data": {
                "train_family": {
                    "name": "train",
                    "n_nodes": 6,
                    "graph_cfg": {"type": "er", "sparsity": 0.25},
                    "mech_cfg": {"type": "linear"},
                },
                "test_families": {
                    "test": {
                        "name": "test",
                        "n_nodes": 6,
                        "graph_cfg": {"type": "er", "sparsity": 0.25},
                        "mech_cfg": {"type": "linear"},
                    }
                },
                "seeds_val": [11],
                "seeds_test": [21],
                "base_seed": 42,
                "samples_per_task": 12,
                "safety_checks": False,
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {
                "type": "random",
                "num_nodes": 6,
            },
            "inference": {
                "n_samples": 2,
                "inil_graph_samples": 1,
            },
        }
    )

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_pipeline(cfg)
    finally:
        os.chdir(original_cwd)

    assert abs(float(cfg.model.p_edge) - 0.25) < 1e-8
    assert (tmp_path / "metrics.json").exists()
