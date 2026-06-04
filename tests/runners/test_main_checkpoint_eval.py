from __future__ import annotations

import os

import pytest
import torch
from omegaconf import OmegaConf

from causal_meta.main import run_pipeline
from causal_meta.models.base import BaseModel


class _DummyAmortizedModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def needs_pretraining(self) -> bool:
        return True

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        _ = mask
        batch = x.shape[0]
        n_nodes = x.shape[-1]
        return self.weight * torch.ones(batch, n_nodes, n_nodes, device=x.device)

    def sample(self, x: torch.Tensor, num_samples: int = 1, mask=None) -> torch.Tensor:
        _ = mask
        batch = x.shape[0]
        n_nodes = x.shape[-1]
        return torch.zeros(batch, num_samples, n_nodes, n_nodes, device=x.device)

    def calculate_loss(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        _ = kwargs
        return ((output - target) ** 2).view(output.shape[0], -1).mean(dim=1)


class _DummyExplicitModel(_DummyAmortizedModel):
    @property
    def needs_pretraining(self) -> bool:
        return False


def _make_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "name": "checkpoint_eval_test",
            "logger": {"wandb": {"enabled": False}},
            "data": {
                "train_family": {
                    "name": "train",
                    "n_nodes": 3,
                    "graph_cfg": {"type": "er", "sparsity": 0.5},
                    "mech_cfg": {"type": "linear", "weight_scale": 1.0},
                },
                "test_families": {
                    "test": {
                        "name": "test",
                        "n_nodes": 3,
                        "graph_cfg": {"type": "er", "sparsity": 0.5},
                        "mech_cfg": {"type": "linear", "weight_scale": 1.0},
                    }
                },
                "seeds_val": [1],
                "seeds_test": [2],
                "base_seed": 0,
                "samples_per_task": 4,
                "safety_checks": False,
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {
                "type": "avici",
                "num_nodes": 3,
            },
            "trainer": {
                "max_optimizer_steps": 1,
                "target_global_tasks_per_step": 1,
                "log_every_n_steps": 1,
                "val_check_interval_steps": 1,
                "lr": 0.001,
                "tf32": False,
            },
            "inference": {
                "n_samples": 2,
                "use_best_checkpoint_for_eval": True,
            },
        }
    )


def test_run_pipeline_loads_best_checkpoint_and_skips_pretraining(
    tmp_path, monkeypatch
) -> None:
    cfg = _make_cfg()
    model = _DummyAmortizedModel()

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    with torch.no_grad():
        model.weight.fill_(7.0)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    with torch.no_grad():
        model.weight.zero_()

    pretrain_called = {"value": False}
    eval_observed = {"weight": None}

    monkeypatch.setattr(
        "causal_meta.main.CausalMetaModule.from_config", lambda _cfg: object()
    )
    monkeypatch.setattr("causal_meta.main.ModelFactory.create", lambda _params: model)

    def _fail_pretrain(*args, **kwargs):
        _ = args, kwargs
        pretrain_called["value"] = True
        raise AssertionError(
            "pre_training.run should be skipped when best checkpoint is loaded"
        )

    def _record_eval(_cfg, eval_model, _data_module, logger=None, output_dir=None):
        _ = logger, output_dir
        unwrapped = eval_model.module if hasattr(eval_model, "module") else eval_model
        eval_observed["weight"] = float(unwrapped.weight.detach().cpu().item())

    monkeypatch.setattr("causal_meta.main.pre_training.run", _fail_pretrain)
    monkeypatch.setattr("causal_meta.main.evaluation.run", _record_eval)

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_pipeline(cfg)
    finally:
        os.chdir(original_cwd)

    assert pretrain_called["value"] is False
    assert eval_observed["weight"] == pytest.approx(7.0)


def test_run_pipeline_errors_when_best_checkpoint_missing(
    tmp_path, monkeypatch
) -> None:
    cfg = _make_cfg()
    model = _DummyAmortizedModel()

    monkeypatch.setattr(
        "causal_meta.main.CausalMetaModule.from_config", lambda _cfg: object()
    )
    monkeypatch.setattr("causal_meta.main.ModelFactory.create", lambda _params: model)
    monkeypatch.setattr("causal_meta.main.evaluation.run", lambda *args, **kwargs: None)

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(FileNotFoundError, match="best checkpoint"):
            run_pipeline(cfg)
    finally:
        os.chdir(original_cwd)


def test_run_pipeline_errors_for_explicit_model_checkpoint_flag(
    tmp_path, monkeypatch
) -> None:
    cfg = _make_cfg()
    cfg.model.type = "random"
    model = _DummyExplicitModel()

    monkeypatch.setattr(
        "causal_meta.main.CausalMetaModule.from_config", lambda _cfg: object()
    )
    monkeypatch.setattr("causal_meta.main.ModelFactory.create", lambda _params: model)
    monkeypatch.setattr("causal_meta.main.evaluation.run", lambda *args, **kwargs: None)

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError, match="only supported for amortized models"):
            run_pipeline(cfg)
    finally:
        os.chdir(original_cwd)
