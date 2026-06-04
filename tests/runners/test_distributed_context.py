import os

import pytest

from causal_meta.runners.utils.distributed import DistributedContext


def test_distributed_context_current_local_rank_env(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "2")
    ctx = DistributedContext.current()
    assert ctx.local_rank == 2


def test_distributed_context_setup_requires_torchrun_env(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    for key in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError):
        _ = DistributedContext.setup()


def test_distributed_context_setup_ignores_slurm_only_env(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="MASTER_ADDR"):
        _ = DistributedContext.setup()

    assert "RANK" in os.environ
    assert "WORLD_SIZE" in os.environ
    assert "LOCAL_RANK" in os.environ
