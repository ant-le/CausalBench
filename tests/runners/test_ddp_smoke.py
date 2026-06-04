from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from causal_meta.runners.tasks.evaluation import run as evaluation_run


class _DummyFixedDataset:
    def __init__(self) -> None:
        self.seeds = [10, 11]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int):
        n_samples = 6
        n_nodes = 3
        input_data = torch.zeros(n_samples, n_nodes)
        adjacency_matrix = torch.zeros(n_nodes, n_nodes)
        return {
            "seed": int(self.seeds[idx]),
            "data": input_data,
            "adjacency": adjacency_matrix,
        }


class _DummyLoader:
    def __init__(self, dataset: _DummyFixedDataset) -> None:
        self.dataset = dataset


class _DummyDataModule:
    def __init__(self) -> None:
        self._dataset = _DummyFixedDataset()
        self.test_families = {}

    def test_dataloader(self):
        return {"dummy": _DummyLoader(self._dataset)}

    def test_interventional_dataloader(self):
        return {}


class _DummyExplicitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # DDP requires parameters to wrap.
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))

    @property
    def needs_pretraining(self) -> bool:
        return False

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        b, _, n = x.shape
        return torch.zeros(b, num_samples, n, n)


def _ddp_worker(*, rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    try:
        cfg = OmegaConf.create(
            {
                "name": "ddp_smoke",
                "inference": {
                    "n_samples": 2,
                    "inil_graph_samples": 1,
                },
            }
        )
        model = DDP(_DummyExplicitModel())
        data_module = _DummyDataModule()
        evaluation_run(cfg, model, data_module, output_dir=Path(output_dir))
    finally:
        dist.destroy_process_group()


def _can_bind_loopback() -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False
    finally:
        s.close()


def test_ddp_evaluation_does_not_hang_and_writes_results(tmp_path) -> None:
    if not _can_bind_loopback():
        pytest.skip(
            "Socket bind is not permitted in this environment; skipping DDP smoke test."
        )

    ctx = mp.get_context("spawn")
    init_file = tmp_path / "ddp_init"
    init_file.touch()

    world_size = 2
    procs = [
        ctx.Process(
            target=_ddp_worker,
            kwargs={
                "rank": rank,
                "world_size": world_size,
                "init_file": str(init_file),
                "output_dir": str(tmp_path),
            },
        )
        for rank in range(world_size)
    ]

    for p in procs:
        p.start()

    timeout_s = 60
    for p in procs:
        p.join(timeout=timeout_s)

    alive = [p for p in procs if p.is_alive()]
    if alive:
        for p in alive:
            p.terminate()
        for p in alive:
            p.join(timeout=5)
        raise AssertionError("DDP smoke test timed out (possible hang).")

    exit_codes = [p.exitcode for p in procs]
    assert exit_codes == [0, 0]

    assert (tmp_path / "metrics.json").exists()
