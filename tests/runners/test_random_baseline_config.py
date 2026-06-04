from __future__ import annotations

from omegaconf import OmegaConf

from causal_meta.main import infer_random_baseline_edge_probability


def test_infer_random_baseline_edge_probability_for_er() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "train_family": {
                    "name": "train",
                    "n_nodes": 8,
                    "graph_cfg": {"type": "er", "sparsity": 0.3},
                    "mech_cfg": {"type": "linear"},
                },
                "test_families": {
                    "test": {
                        "name": "test",
                        "n_nodes": 8,
                        "graph_cfg": {"type": "er", "sparsity": 0.3},
                        "mech_cfg": {"type": "linear"},
                    }
                },
                "seeds_val": [0],
                "seeds_test": [1],
            }
        }
    )

    p_edge = infer_random_baseline_edge_probability(cfg)
    assert abs(p_edge - 0.3) < 1e-8


def test_infer_random_baseline_edge_probability_for_mixture() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "train_family": {
                    "name": "train",
                    "n_nodes": 10,
                    "graph_cfg": {
                        "type": "mixture",
                        "weights": [0.7, 0.3],
                        "generators": [
                            {"type": "er", "sparsity": 0.1},
                            {"type": "er", "sparsity": 0.4},
                        ],
                    },
                    "mech_cfg": {"type": "linear"},
                },
                "test_families": {
                    "test": {
                        "name": "test",
                        "n_nodes": 10,
                        "graph_cfg": {"type": "er", "sparsity": 0.2},
                        "mech_cfg": {"type": "linear"},
                    }
                },
                "seeds_val": [0],
                "seeds_test": [1],
            }
        }
    )

    p_edge = infer_random_baseline_edge_probability(cfg)
    expected = 0.7 * 0.1 + 0.3 * 0.4
    assert abs(p_edge - expected) < 1e-8
