import pytest
from hydra import compose, initialize_config_module

from causal_meta.datasets.generators.configs import (ErdosRenyiConfig,
                                                     FamilyConfig, SBMConfig)
from causal_meta.datasets.generators.factory import (load_data_module_config,
                                                     load_graph_config)


def test_graph_loader_ignores_stale_hydra_merge_keys() -> None:
    graph_cfg = load_graph_config(
        {
            "type": "er",
            "sparsity": 0.1342,
            "n_blocks": 4,
            "p_intra": 0.6,
            "p_inter": 0.01,
        }
    )

    assert isinstance(graph_cfg, ErdosRenyiConfig)
    assert graph_cfg.sparsity == pytest.approx(0.1342)


def test_graph_loader_rejects_unknown_keys() -> None:
    with pytest.raises(TypeError, match="Unexpected graph config key"):
        load_graph_config({"type": "er", "sparcity": 0.1})


def test_graph_id_ablation_config_loads_after_hydra_merge() -> None:
    with initialize_config_module(
        config_module="causal_meta.configs",
        version_base=None,
    ):
        cfg = compose(
            config_name="dg_2pretrain_ablation_graph_id",
            overrides=["model=bcnp", "name=bcnp_ablation"],
        )

    data_cfg = load_data_module_config(cfg.data)
    val_family = data_cfg.val_families["ood_graph_sbm_linear_d20_n500"]
    test_family = data_cfg.test_families["ood_graph_sbm_linear_d20_n500"]

    assert isinstance(test_family, FamilyConfig)
    assert isinstance(val_family.graph_cfg, SBMConfig)
    assert isinstance(test_family.graph_cfg, SBMConfig)
