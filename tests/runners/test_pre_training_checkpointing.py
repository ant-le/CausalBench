from __future__ import annotations

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import pytest
import torch
from omegaconf import OmegaConf

from causal_meta.models.base import BaseModel
from causal_meta.runners.logger.local import LocalLogger
from causal_meta.runners.tasks.pre_training import _augment_validation_group_metrics
from causal_meta.runners.tasks.pre_training import _build_scheduler
from causal_meta.runners.tasks.pre_training import _is_metric_improved
from causal_meta.runners.tasks.pre_training import _resolve_validation_selection_metric
from causal_meta.runners.tasks.pre_training import _should_maximize_selection_metric
from causal_meta.runners.tasks.pre_training import (
    _validation_selection_mode_from_config,
)
from causal_meta.runners.tasks.pre_training import run as pre_training_run
from causal_meta.runners.tasks.pre_training import save_checkpoint


class _DummyPretrainModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def needs_pretraining(self) -> bool:
        return True

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        batch = x.shape[0]
        n_nodes = x.shape[-1]
        return self.weight * torch.ones(batch, n_nodes, n_nodes, device=x.device)

    def sample(self, x: torch.Tensor, num_samples: int = 1, mask=None) -> torch.Tensor:
        batch = x.shape[0]
        n_nodes = x.shape[-1]
        return torch.zeros(batch, num_samples, n_nodes, n_nodes, device=x.device)

    def calculate_loss(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        _ = kwargs
        diff = (output - target).pow(2)
        return diff.view(diff.shape[0], -1).mean(dim=1)


class _DummyTrainDataset:
    def __init__(self, base_seed: int) -> None:
        self.base_seed = int(base_seed)


class _DummyDataModule:
    def __init__(self, base_seed: int = 10) -> None:
        n_nodes = 3
        n_samples = 5
        self.train_dataset = _DummyTrainDataset(base_seed=base_seed)
        self._train_batches = [
            {
                "seed": torch.tensor([1]),
                "data": torch.zeros(1, n_samples, n_nodes),
                "adjacency": torch.zeros(1, n_nodes, n_nodes),
            }
        ]
        self._val_batches = [
            {
                "seed": torch.tensor([2]),
                "data": torch.zeros(1, n_samples, n_nodes),
                "adjacency": torch.zeros(1, n_nodes, n_nodes),
            }
        ]

    def train_dataloader(self, *, batch_size_override=None):
        _ = batch_size_override
        return self._train_batches

    def val_dataloader(self):
        return {"val": self._val_batches}


def _make_cfg(max_optimizer_steps: int = 2) -> OmegaConf:
    return OmegaConf.create(
        {
            "name": "pretrain_test",
            "seed": 777,
            "data": {
                "batch_size_train": 1,
                "base_seed": 10,
                "num_workers": 0,
            },
            "trainer": {
                "lr": 1e-3,
                "max_optimizer_steps": max_optimizer_steps,
                "target_global_tasks_per_step": 1,
                "log_every_n_steps": 100,
                "val_check_interval_steps": 100,
                "checkpoint_every_n_steps": 100,
                "scheduler": "none",
                "amp": False,
                "amp_dtype": "bf16",
            },
            "inference": {"n_samples": 2},
        }
    )


def test_save_checkpoint_contains_stream_resume_metadata(tmp_path) -> None:
    cfg = _make_cfg(max_optimizer_steps=24)
    model = _DummyPretrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    out = tmp_path / "checkpoint.pt"
    save_checkpoint(
        cfg,
        model,
        optimizer,
        scheduler,
        scaler,
        step=3,
        filepath=out,
        tasks_seen=24,
        train_stream_initial_base_seed=10,
        global_tasks_per_step=8,
        world_size=2,
        train_batch_size=4,
        accumulate_grad_batches=2,
    )

    state = torch.load(out, map_location="cpu")
    assert state["step"] == 3
    assert state["experiment_seed"] == 777
    assert state["tasks_seen"] == 24
    assert state["train_stream_initial_base_seed"] == 10
    assert state["train_stream_global_tasks_per_step"] == 8
    assert state["train_stream_world_size"] == 2
    assert state["train_stream_batch_size_train"] == 4
    assert state["train_stream_accumulate_grad_batches"] == 2
    assert state["train_stream_next_base_seed_if_num_workers_0"] == 34
    assert "scheduler_state_dict" in state


def test_pre_training_run_resumes_model_and_stream_seed(tmp_path) -> None:
    cfg = _make_cfg(max_optimizer_steps=2)
    model = _DummyPretrainModel()
    data_module = _DummyDataModule(base_seed=10)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "last.pt"

    with torch.no_grad():
        model.weight.fill_(5.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_checkpoint(
        cfg,
        model,
        optimizer,
        scheduler=None,
        scaler=scaler,
        step=2,
        filepath=resume_path,
        tasks_seen=2,
        train_stream_initial_base_seed=10,
        global_tasks_per_step=1,
        world_size=1,
        train_batch_size=1,
        accumulate_grad_batches=1,
    )

    with torch.no_grad():
        model.weight.zero_()

    pre_training_run(cfg, model, data_module, output_dir=tmp_path)

    assert torch.isclose(model.weight.detach().cpu(), torch.tensor(5.0))
    assert data_module.train_dataset.base_seed == 12

    last_state = torch.load(resume_path, map_location="cpu")
    assert last_state["step"] == 2


def test_pre_training_logs_progress_on_steps(tmp_path) -> None:
    cfg = _make_cfg(max_optimizer_steps=2)
    cfg.trainer.log_every_n_steps = 1
    model = _DummyPretrainModel()
    data_module = _DummyDataModule(base_seed=10)
    logger = LocalLogger()

    pre_training_run(cfg, model, data_module, logger=logger, output_dir=tmp_path)

    logged_steps = [entry["step"] for entry in logger.history if "train/loss" in entry]
    assert logged_steps == [1, 2]
    assert logger.history[0]["train/tasks_seen"] == 1.0


def test_build_scheduler_none_and_invalid() -> None:
    param = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([param], lr=1e-3)

    cfg_none = OmegaConf.create(
        {"trainer": {"scheduler": "none", "max_optimizer_steps": 10}}
    )
    assert _build_scheduler(optimizer, cfg_none) is None

    cfg_invalid = OmegaConf.create(
        {"trainer": {"scheduler": "linear", "max_optimizer_steps": 10}}
    )
    try:
        _build_scheduler(optimizer, cfg_invalid)
    except ValueError as exc:
        assert "trainer.scheduler" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported scheduler type")


def test_build_scheduler_with_linear_warmup() -> None:
    param = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    cfg = OmegaConf.create(
        {
            "trainer": {
                "scheduler": "cosine",
                "max_optimizer_steps": 10,
                "scheduler_t_max_steps": 10,
                "scheduler_warmup_ratio": 0.2,
                "scheduler_warmup_start_factor": 0.1,
            }
        }
    )

    scheduler = _build_scheduler(optimizer, cfg)
    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

    lrs = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        lrs.append(float(optimizer.param_groups[0]["lr"]))

    assert lrs[0] < lrs[1]
    assert lrs[-1] < lrs[1]


def test_build_scheduler_multistep_drops_learning_rate() -> None:
    param = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    cfg = OmegaConf.create(
        {
            "trainer": {
                "scheduler": "multistep",
                "max_optimizer_steps": 10,
                "scheduler_milestones_steps": [5],
                "scheduler_gamma": 0.1,
                "scheduler_warmup_ratio": 0.0,
            }
        }
    )

    scheduler = _build_scheduler(optimizer, cfg)
    assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)

    lrs = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(float(optimizer.param_groups[0]["lr"]))

    assert lrs[3] == pytest.approx(1e-3)
    assert lrs[4] == pytest.approx(1e-4)


def test_validation_selection_mode_helpers_support_sid_minimization() -> None:
    cfg = OmegaConf.create({"trainer": {"validation_selection_mode": "min"}})

    assert _validation_selection_mode_from_config(cfg) == "min"
    assert _should_maximize_selection_metric("mean_ood_ne-sid", "auto") is False
    assert _should_maximize_selection_metric("mean_id_e-edgef1", "auto") is True
    assert _is_metric_improved(0.2, 0.3, maximize=False) is True
    assert _is_metric_improved(0.3, 0.2, maximize=False) is False


def test_resolve_validation_selection_metric_uses_fallback_alias() -> None:
    metric_name, metric_value = _resolve_validation_selection_metric(
        {"mean_e-edgef1": 0.4},
        "mean_ood_ne-sid",
    )

    assert metric_name == "mean_e-edgef1"
    assert metric_value == pytest.approx(0.4)


def test_augment_validation_group_metrics_adds_id_and_ood_means() -> None:
    metrics = {
        "id_val_linear_er20/e-edgef1": 0.20,
        "id_val_neuralnet_er40/e-edgef1": 0.40,
        "ood_val_mechanism_periodic_er40/e-edgef1": 0.10,
        "id_val_linear_er20/ne-sid": 0.12,
        "id_val_neuralnet_er40/ne-sid": 0.18,
        "ood_val_mechanism_periodic_er40/ne-sid": 0.25,
        "id_val_linear_er20/auc": 0.70,
        "id_val_neuralnet_er40/auc": 0.80,
        "ood_val_mechanism_periodic_er40/auc": 0.50,
    }

    _augment_validation_group_metrics(
        metrics,
        group_prefixes={"id": ["id_"], "ood": ["ood_"]},
    )

    assert metrics["mean_id_e-edgef1"] == pytest.approx(0.30)
    assert metrics["mean_ood_e-edgef1"] == pytest.approx(0.10)
    assert metrics["mean_id_ne-sid"] == pytest.approx(0.15)
    assert metrics["mean_ood_ne-sid"] == pytest.approx(0.25)
    assert metrics["mean_id_auc"] == pytest.approx(0.75)
    assert metrics["mean_ood_auc"] == pytest.approx(0.50)
