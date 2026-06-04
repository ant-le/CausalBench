import json

from omegaconf import OmegaConf

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.models.avici.model import AviciModel
from causal_meta.runners.tasks.evaluation import run as evaluation_run


def test_evaluation_writes_results_to_output_dir(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "name": "eval_test",
            "inference": {
                "n_samples": 2,
                "inil_graph_samples": 2,
            },
        }
    )

    data_cfg = {
        "train_family": {
            "name": "train",
            "n_nodes": 3,
            "graph_cfg": {"type": "er", "sparsity": 0.3},
            "mech_cfg": {"type": "linear"},
        },
        "test_families": {
            "test": {
                "name": "test",
                "n_nodes": 3,
                "graph_cfg": {"type": "er", "sparsity": 0.3},
                "mech_cfg": {"type": "linear"},
            }
        },
        "seeds_val": [0],
        "seeds_test": [1],
        "samples_per_task": 8,
        "num_workers": 0,
        "pin_memory": False,
        "safety_checks": True,
    }

    data_module = CausalMetaModule.from_config(data_cfg)
    data_module.setup()

    model = AviciModel(num_nodes=3, d_model=8, nhead=2, num_layers=2)

    evaluation_run(cfg, model, data_module, output_dir=tmp_path)

    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()

    payload = json.loads(metrics_path.read_text())
    assert "summary" in payload
    assert "raw" in payload
    assert "test" in payload["summary"]
    assert "graph_nll_per_edge_mean" in payload["summary"]["test"]
    assert "sampled-graph_nll_per_edge_mean" in payload["summary"]["test"]
    assert "sampled-ne-sid_mean" in payload["summary"]["test"]
    assert "sampled-valid_dag_pct_mean" in payload["summary"]["test"]
    assert "threshold_valid_dag_pct_mean" not in payload["summary"]["test"]
