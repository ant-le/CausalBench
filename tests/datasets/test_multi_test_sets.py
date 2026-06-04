from causal_meta.datasets.data_module import CausalMetaModule
import torch


def test_multi_test_families() -> None:
    # Test initialization with multiple named test families

    cfg = {
        "train_family": {
            "name": "train",
            "n_nodes": 5,
            "graph_cfg": {"type": "er", "edge_prob": 0.3},
            "mech_cfg": {"type": "linear"},
        },
        "test_families": {
            "ood_graph": {
                "name": "ood_graph",
                "n_nodes": 5,
                "graph_cfg": {"type": "sf", "m": 2},
                "mech_cfg": {"type": "linear"},
            },
            "ood_mech": {
                "name": "ood_mech",
                "n_nodes": 5,
                "graph_cfg": {"type": "er", "edge_prob": 0.3},
                "mech_cfg": {"type": "mlp", "hidden_dim": 16},
            },
        },
        "seeds_val": [0],
        "seeds_test": [1],
        "samples_per_task": 10,
    }
    module = CausalMetaModule.from_config(cfg)
    module.setup()

    # Check if we have two test families
    assert len(module.test_families) == 2
    assert "ood_graph" in module.test_families
    assert "ood_mech" in module.test_families

    # Check if generators are correct
    assert "ScaleFreeGenerator" in str(
        type(module.test_families["ood_graph"].graph_generator)
    )
    assert "MLPMechanismFactory" in str(
        type(module.test_families["ood_mech"].mechanism_factory)
    )

    # Check DataLoaders
    loaders = module.test_dataloader()
    assert len(loaders) == 2
    assert isinstance(loaders["ood_graph"], torch.utils.data.DataLoader)
