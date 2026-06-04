from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.main import _apply_model_specific_trainer_profile


def test_causal_meta_module_parses_target_based_config() -> None:
    cfg = OmegaConf.create(
        {
            "train_family": {
                "name": "train",
                "n_nodes": 4,
                "graph_cfg": {
                    "_target_": "causal_meta.datasets.generators.graphs.er.ErdosRenyiGenerator",
                    "sparsity": 0.3,
                },
                "mech_cfg": {
                    "_target_": "causal_meta.datasets.generators.mechanisms.linear.LinearMechanismFactory",
                    "weight_scale": 0.1,
                },
            },
            "test_families": {
                "test": {
                    "name": "test",
                    "n_nodes": 4,
                    "graph_cfg": {
                        "_target_": "causal_meta.datasets.generators.graphs.sf.ScaleFreeGenerator",
                        "m": 2,
                    },
                    "mech_cfg": {
                        "_target_": "causal_meta.datasets.generators.mechanisms.mlp.MLPMechanismFactory",
                        "hidden_dim": 8,
                    },
                }
            },
            "seeds_val": [0, 1, 2],
            "seeds_test": [10, 11, 12],
            "samples_per_task": 4,
        }
    )

    module = CausalMetaModule.from_config(cfg)
    module.setup()

    assert module.train_family is not None
    assert module.train_dataset is not None
    assert "test" in module.test_families
    assert "test" in module.test_datasets


def test_smoke_config_overrides_explicit_inference_groups() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "src" / "causal_meta" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="dg_2pretrain_smoke")

    assert cfg.inference.dibs.steps == 50
    assert cfg.inference.dibs.profile_overrides.linear_d20.n_particles == 8
    assert cfg.inference.bayesdag.max_epochs == 5
    assert cfg.inference.bayesdag.profile_overrides.linear.num_chains == 2


def test_default_config_selects_avici_trainer_profile() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "src" / "causal_meta" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="default")

    _apply_model_specific_trainer_profile(cfg)

    assert cfg.model.type == "avici"
    assert cfg.model.trainer_profile == "avici"
    assert cfg.trainer.scheduler == "cosine"
    assert cfg.trainer.validation_selection_mode == "max"
    assert cfg.trainer.max_optimizer_steps == 30
    assert cfg.trainer.target_global_tasks_per_step == 32


def test_benchmark_config_applies_full_avici_trainer_profile() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "src" / "causal_meta" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="dg_2pretrain_multimodel",
            overrides=["model=avici"],
        )

    _apply_model_specific_trainer_profile(cfg)

    assert cfg.trainer.scheduler == "cosine"
    assert cfg.trainer.validation_selection_mode == "max"
    assert cfg.trainer.max_optimizer_steps == 500000
    assert cfg.trainer.target_global_tasks_per_step == 32


def test_benchmark_config_applies_full_bcnp_trainer_profile() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "src" / "causal_meta" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="dg_2pretrain_multimodel",
            overrides=["model=bcnp"],
        )

    _apply_model_specific_trainer_profile(cfg)

    assert cfg.model.trainer_profile == "bcnp"
    assert cfg.trainer.scheduler == "cosine"
    assert cfg.trainer.validation_selection_mode == "min"
    assert cfg.trainer.max_optimizer_steps == 500000
    assert cfg.trainer.target_global_tasks_per_step == 32
