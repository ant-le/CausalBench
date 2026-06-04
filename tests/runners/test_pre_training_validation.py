import torch
from torch.utils.data import DataLoader

from causal_meta.datasets.utils import collate_fn_scm
from causal_meta.runners.tasks.pre_training import (
    _validation_metrics_from_config,
    validate,
)


class _DummyDataModule:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self.val_called = False

    def val_dataloader(self):
        self.val_called = True
        return {"id": self._loader}

    def test_dataloader(self):
        raise AssertionError("test_dataloader must not be called for validation.")


class _DummyModel(torch.nn.Module):
    def __init__(self, n_nodes: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        del mask
        batch_size = x.shape[0]
        return torch.zeros(batch_size, self.n_nodes, self.n_nodes)

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask
        batch_size = x.shape[0]
        return torch.zeros(batch_size, num_samples, self.n_nodes, self.n_nodes)


class _DummyAviciModel(_DummyModel):
    pass


def test_pre_training_validate_uses_validation_loader() -> None:
    n_nodes = 3
    x = torch.zeros(5, n_nodes)
    adj = torch.zeros(n_nodes, n_nodes)
    loader = DataLoader(
        [{"seed": 0, "data": x, "adjacency": adj}],
        batch_size=1,
        collate_fn=collate_fn_scm,
    )

    data_module = _DummyDataModule(loader)
    model = _DummyModel(n_nodes=n_nodes)

    metrics = validate(model, data_module, torch.device("cpu"))
    assert data_module.val_called is True
    assert metrics["id/e-edgef1"] == 1.0  # empty prediction matches empty truth
    assert metrics["id/ne-sid"] == 0.0
    assert metrics["id/ne-shd"] == 0.0
    assert metrics["id/valid_dag_pct"] == 100.0
    assert metrics["ne-sid"] == 0.0
    assert metrics["ne-shd"] == 0.0
    assert metrics["valid_dag_pct"] == 100.0
    assert "mean_e-edgef1" in metrics


def test_pre_training_validate_allows_normalized_metric_override() -> None:
    n_nodes = 3
    x = torch.zeros(5, n_nodes)
    adj = torch.zeros(n_nodes, n_nodes)
    loader = DataLoader(
        [{"seed": 0, "data": x, "adjacency": adj}],
        batch_size=1,
        collate_fn=collate_fn_scm,
    )

    data_module = _DummyDataModule(loader)
    model = _DummyModel(n_nodes=n_nodes)

    metrics = validate(
        model,
        data_module,
        torch.device("cpu"),
        metrics=["ne-sid"],
    )

    assert metrics["id/ne-sid"] == 0.0
    assert metrics["ne-sid"] == 0.0
    assert "id/e-edgef1" not in metrics
    assert "id/valid_dag_pct" not in metrics
    assert "mean_e-edgef1" not in metrics


def test_validation_metrics_from_config_adds_sampled_metrics_for_avici() -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {"type": "avici"},
            "trainer": {"validation_metrics": ["ne-sid", "valid_dag_pct"]},
        }
    )

    metrics = _validation_metrics_from_config(cfg)
    assert "ne-sid" in metrics
    assert "valid_dag_pct" in metrics
    assert "sampled-ne-sid" in metrics
    assert "sampled-valid_dag_pct" in metrics
