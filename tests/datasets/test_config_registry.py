from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.datasets.generators.configs import (
    DataModuleConfig,
    ErdosRenyiConfig,
    FamilyConfig,
    LinearMechanismConfig,
)


def test_registry_based_initialization() -> None:
    # Test initialization using dictionaries via from_config
    cfg = {
        "train_family": {
            "name": "train",
            "n_nodes": 5,
            "graph_cfg": {"type": "er", "edge_prob": 0.3},
            "mech_cfg": {"type": "linear", "weight_scale": 1.0},
        },
        "test_families": {
            "test": {
                "name": "test",
                "n_nodes": 5,
                "graph_cfg": {"type": "scale_free", "m": 2},
                "mech_cfg": {"type": "mlp", "hidden_dim": 16},
            }
        },
        "seeds_val": [0],
        "seeds_test": [1],
        "samples_per_task": 10,
    }

    module = CausalMetaModule.from_config(cfg)
    module.setup()

    assert module.train_family is not None
    assert "test" in module.test_families
    assert module.test_families["test"] is not None

    assert "ErdosRenyiGenerator" in str(type(module.train_family.graph_generator))
    assert "LinearMechanismFactory" in str(type(module.train_family.mechanism_factory))


def test_mixture_config_parsing() -> None:
    # Test nested mixture parsing via from_config
    cfg = {
        "train_family": {
            "name": "mixture_test",
            "n_nodes": 10,
            "graph_cfg": {
                "type": "mixture",
                "weights": [0.5, 0.5],
                "generators": [
                    {"type": "er", "edge_prob": 0.1},
                    {"type": "sf", "m": 1},
                ],
            },
            "mech_cfg": {
                "type": "mixture",
                "weights": [0.8, 0.2],
                "factories": [{"type": "linear"}, {"type": "mlp", "hidden_dim": 32}],
            },
        },
        "test_families": {
            "test": {
                "name": "test",
                "n_nodes": 10,
                "graph_cfg": {"type": "er", "edge_prob": 0.1},
                "mech_cfg": {"type": "linear"},
            }
        },
        "seeds_val": [0],
        "seeds_test": [1],
    }

    module = CausalMetaModule.from_config(cfg)
    module.setup()

    assert module.train_family is not None

    assert "MixtureGraphGenerator" in str(type(module.train_family.graph_generator))
    assert "MixtureMechanismFactory" in str(type(module.train_family.mechanism_factory))

    gens = getattr(module.train_family.graph_generator, "generators")
    assert len(gens) == 2
    assert "ErdosRenyiGenerator" in str(type(gens[0]))
    assert "ScaleFreeGenerator" in str(type(gens[1]))


def test_explicit_config_initialization() -> None:
    # Test the new "Best Way" initialization
    config = DataModuleConfig(
        train_family=FamilyConfig(
            name="train",
            n_nodes=5,
            graph_cfg=ErdosRenyiConfig(edge_prob=0.5),
            mech_cfg=LinearMechanismConfig(),
        ),
        test_families={
            "test": FamilyConfig(
                name="test",
                n_nodes=5,
                graph_cfg=ErdosRenyiConfig(edge_prob=0.5),
                mech_cfg=LinearMechanismConfig(),
            )
        },
        seeds_val=[0],
        seeds_test=[1],
    )

    module = CausalMetaModule(config)
    module.setup()

    assert module.train_family is not None
    assert "test" in module.test_families


def test_gp_mechanism_config_supports_exact_mode() -> None:
    cfg = {
        "train_family": {
            "name": "train",
            "n_nodes": 4,
            "graph_cfg": {"type": "er", "edge_prob": 0.2},
            "mech_cfg": {
                "type": "gp",
                "mode": "exact",
                "exact_num_kernel_pairs": 1,
                "length_scale_range": [0.5, 1.0],
                "variance_range": [0.5, 1.0],
            },
        },
        "test_families": {
            "test": {
                "name": "test",
                "n_nodes": 4,
                "graph_cfg": {"type": "er", "edge_prob": 0.2},
                "mech_cfg": {"type": "linear"},
            }
        },
        "seeds_val": [0],
        "seeds_test": [1],
    }

    module = CausalMetaModule.from_config(cfg)
    module.setup()

    assert module.train_family is not None
    mechanism_factory = module.train_family.mechanism_factory
    assert "GPMechanismFactory" in str(type(mechanism_factory))
    assert getattr(mechanism_factory, "mode") == "exact"
