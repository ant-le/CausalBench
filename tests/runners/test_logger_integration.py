import logging

import pytest
from omegaconf import OmegaConf

from causal_meta.main import run_pipeline
from causal_meta.runners.logger.local import LocalLogger


class TestLocalLogger:
    def test_implements_protocol(self):
        logger = LocalLogger()
        assert isinstance(logger, LocalLogger)
        # Runtime check for protocol adherence (static check is done by mypy)
        assert hasattr(logger, "log_metrics")
        assert hasattr(logger, "log_hyperparams")
        assert hasattr(logger, "finish")

    def test_log_metrics(self):
        logger = LocalLogger()
        metrics = {"train/loss": 0.5, "val/acc": 0.9}
        logger.log_metrics(metrics, step=10)

        assert len(logger.history) == 1
        entry = logger.history[0]
        assert entry["step"] == 10
        assert entry["train/loss"] == 0.5
        assert entry["val/acc"] == 0.9

    def test_log_metrics_uses_step_prefix(self, caplog):
        logger = LocalLogger()

        with caplog.at_level(logging.INFO):
            logger.log_metrics({"train/loss": 0.5}, step=12)

        assert "Step 12: train/loss: 0.5000" in caplog.text

    def test_log_hyperparams(self, caplog):
        logger = LocalLogger()
        params = {"lr": 0.01, "batch_size": 32}

        with caplog.at_level(logging.INFO):
            logger.log_hyperparams(params)

        assert "Hyperparameters: {'lr': 0.01, 'batch_size': 32}" in caplog.text


def test_pipeline_integration_smoke(tmp_path):
    """
    Runs the pipeline with the default 'smoke_test' config (which has wandb disabled).
    Verifies that the LocalLogger is used (implicitly, by checking that no wandb error occurs)
    and that artifacts are produced.
    """
    # Create a minimal valid config
    cfg = OmegaConf.create(
        {
            "name": "integration_test",
            "logger": {"wandb": {"enabled": False}},
            "data": {
                "train_family": {
                    "name": "train",
                    "n_nodes": 5,
                    "graph_cfg": {"type": "er", "sparsity": 0.5},
                    "mech_cfg": {"type": "linear", "weight_scale": 1.0},
                },
                "test_families": {
                    "test_1": {
                        "name": "test_1",
                        "n_nodes": 5,
                        "graph_cfg": {"type": "er", "sparsity": 0.5},
                        "mech_cfg": {"type": "linear", "weight_scale": 1.0},
                    }
                },
                "seeds_val": [100],
                "seeds_test": [200],
                "base_seed": 42,
                "samples_per_task": 10,
                "safety_checks": False,
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {
                "type": "avici",
                "num_nodes": 5,
                "d_model": 8,
                "nhead": 2,
                "num_layers": 2,
                "dim_feedforward": 16,
                "dropout": 0.0,
            },
            "trainer": {
                "max_optimizer_steps": 2,  # Very short run
                "target_global_tasks_per_step": 1,
                "log_every_n_steps": 1,
                "val_check_interval_steps": 2,
                "lr": 0.001,
                "tf32": False,
            },
            "inference": {
                "n_samples": 2,
                "inil_graph_samples": 1,
            },
        }
    )

    # Let's run it.
    # We change cwd to tmp_path to avoid writing to project root
    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_pipeline(cfg)
    finally:
        os.chdir(original_cwd)

    # Check for artifacts
    # The default behavior writes to os.getcwd() if hydra config is missing
    # Structure: <cwd>/checkpoints/last.pt
    # Structure: <cwd>/metrics.json

    assert (tmp_path / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "metrics.json").exists()


def test_pipeline_falls_back_to_local_logger_when_wandb_init_fails(
    tmp_path, monkeypatch
):
    class _FailingWandbLogger:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("401 unauthorized")

    monkeypatch.setattr("causal_meta.main.WandbLogger", _FailingWandbLogger)

    cfg = OmegaConf.create(
        {
            "name": "integration_test_wandb_fallback",
            "logger": {
                "wandb": {
                    "enabled": True,
                    "mode": "online",
                    "project": "causal_meta",
                }
            },
            "data": {
                "train_family": {
                    "name": "train",
                    "n_nodes": 5,
                    "graph_cfg": {"type": "er", "sparsity": 0.5},
                    "mech_cfg": {"type": "linear", "weight_scale": 1.0},
                },
                "test_families": {
                    "test_1": {
                        "name": "test_1",
                        "n_nodes": 5,
                        "graph_cfg": {"type": "er", "sparsity": 0.5},
                        "mech_cfg": {"type": "linear", "weight_scale": 1.0},
                    }
                },
                "seeds_val": [100],
                "seeds_test": [200],
                "base_seed": 42,
                "samples_per_task": 10,
                "safety_checks": False,
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {
                "type": "avici",
                "num_nodes": 5,
                "d_model": 8,
                "nhead": 2,
                "num_layers": 2,
                "dim_feedforward": 16,
                "dropout": 0.0,
            },
            "trainer": {
                "max_optimizer_steps": 2,
                "target_global_tasks_per_step": 1,
                "log_every_n_steps": 1,
                "val_check_interval_steps": 2,
                "lr": 0.001,
                "tf32": False,
            },
            "inference": {
                "n_samples": 2,
                "inil_graph_samples": 1,
            },
        }
    )

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_pipeline(cfg)
    finally:
        os.chdir(original_cwd)

    assert (tmp_path / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "metrics.json").exists()


def test_pipeline_rejects_multiple_models(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "name": "multi_model_should_fail",
            "logger": {"wandb": {"enabled": False}},
            "data": {
                "train_family": {
                    "name": "train",
                    "n_nodes": 5,
                    "graph_cfg": {"type": "er", "sparsity": 0.5},
                    "mech_cfg": {"type": "linear", "weight_scale": 1.0},
                },
                "test_families": {
                    "test_1": {
                        "name": "test_1",
                        "n_nodes": 5,
                        "graph_cfg": {"type": "er", "sparsity": 0.5},
                        "mech_cfg": {"type": "linear", "weight_scale": 1.0},
                    }
                },
                "seeds_val": [100],
                "seeds_test": [200],
                "samples_per_task": 10,
                "safety_checks": False,
                "num_workers": 0,
                "pin_memory": False,
            },
            "models": {
                "avici": {
                    "type": "avici",
                    "num_nodes": 5,
                    "d_model": 8,
                    "nhead": 2,
                    "num_layers": 2,
                },
                "bcnp": {
                    "type": "bcnp",
                    "num_nodes": 5,
                    "d_model": 8,
                    "nhead": 2,
                    "num_layers": 2,
                },
            },
            "trainer": {
                "max_optimizer_steps": 1,
                "target_global_tasks_per_step": 1,
                "lr": 0.001,
            },
            "inference": {"n_samples": 1},
        }
    )

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError, match="Missing required config keys"):
            run_pipeline(cfg)
    finally:
        os.chdir(original_cwd)
