from __future__ import annotations

import torch
from omegaconf import OmegaConf

from causal_meta.runners.tasks.evaluation import run as evaluation_run
from causal_meta.runners.utils.artifacts import (
    atomic_torch_save,
    torch_load,
)


class _DummyFixedDataset:
    def __init__(self, *, n_nodes: int = 3) -> None:
        self.seeds = [123]
        self.num_samples = 5
        self.n_nodes = n_nodes

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int):
        x = torch.zeros(self.num_samples, self.n_nodes)
        adj = torch.zeros(self.n_nodes, self.n_nodes)
        return {"seed": int(self.seeds[idx]), "data": x, "adjacency": adj}


class _DummyLoader:
    def __init__(self, dataset: _DummyFixedDataset) -> None:
        self.dataset = dataset


class _DummyDataModule:
    def __init__(self, dataset: _DummyFixedDataset) -> None:
        self._dataset = dataset
        self.test_families = {
            "dummy": type("Family", (), {"n_nodes": dataset.n_nodes})()
        }

    def test_dataloader(self):
        return {"dummy": _DummyLoader(self._dataset)}

    def test_interventional_dataloader(self):
        return {}


class _TrackingExplicitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sample_calls = 0

    @property
    def needs_pretraining(self) -> bool:
        return False

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        self.sample_calls += 1
        b, _, n = x.shape
        return torch.ones(b, num_samples, n, n)


class _ResizableExplicitModel(torch.nn.Module):
    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_node_updates: list[int] = []

    @property
    def needs_pretraining(self) -> bool:
        return False

    def set_num_nodes(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self.num_node_updates.append(num_nodes)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        b, _, n = x.shape
        if n != self.num_nodes:
            raise ValueError(
                "Input data node count does not match configured num_nodes."
            )
        return torch.zeros(b, num_samples, n, n)


def test_evaluation_explicit_model_always_samples_and_writes_artifact(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "name": "eval_explicit",
            "inference": {
                "n_samples": 5,
                "inil_graph_samples": 2,
                "cache_compress": True,
                "cache_dtype": "uint8",
            },
        }
    )

    dataset = _DummyFixedDataset()
    data_module = _DummyDataModule(dataset)
    model = _TrackingExplicitModel()

    existing_path = tmp_path / "inference" / "dummy" / "seed_123.pt.gz"
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {"seed": 123, "idx": 0, "graph_samples": torch.zeros(1, 5, 3, 3)},
        existing_path,
    )

    evaluation_run(cfg, model, data_module, output_dir=tmp_path)

    assert model.sample_calls == 1
    assert (tmp_path / "metrics.json").exists()

    artifact = torch_load(existing_path)
    assert torch.equal(
        artifact["graph_samples"],
        torch.ones(1, 5, dataset.n_nodes, dataset.n_nodes, dtype=torch.uint8),
    )


def test_evaluation_updates_explicit_model_num_nodes_per_dataset(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "name": "eval_explicit_nodes",
            "inference": {
                "n_samples": 2,
                "inil_graph_samples": 1,
            },
        }
    )

    dataset = _DummyFixedDataset(n_nodes=3)
    data_module = _DummyDataModule(dataset)
    model = _ResizableExplicitModel(num_nodes=60)

    evaluation_run(cfg, model, data_module, output_dir=tmp_path)

    assert model.num_nodes == 3
    assert model.num_node_updates == [3]
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "inference" / "dummy" / "seed_123.pt").exists() or (
        tmp_path / "inference" / "dummy" / "seed_123.pt.gz"
    ).exists()
