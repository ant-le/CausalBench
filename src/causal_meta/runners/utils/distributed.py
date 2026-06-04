from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


def _infer_local_rank() -> int:
    """Infer local rank from torchrun/SLURM environment."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])

    return 0


def _safe_cuda_local_rank(local_rank: int) -> int:
    """Validate local rank against visible CUDA devices for this process."""
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("CUDA is available but no visible devices were found.")
    if 0 <= local_rank < device_count:
        return local_rank

    raise RuntimeError(
        "Invalid LOCAL_RANK for visible CUDA devices: "
        f"LOCAL_RANK={local_rank}, cuda_device_count={device_count}, "
        f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}'."
    )


def select_device(*, local_rank: int, is_distributed: bool) -> torch.device:
    """
    Select the preferred device for this process.

    CUDA is preferred when available (for DDP compatibility). If running
    single-process and CUDA is unavailable, MPS is selected when available;
    otherwise CPU is used.

    Args:
        local_rank: Local rank provided by torchrun/Slurm environment.
        is_distributed: Whether the process is in a distributed context.

    Returns:
        torch.device: The selected device for this process.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if not is_distributed and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class DistributedContext:
    """
    Lightweight helper for torch.distributed / DDP execution.

    `setup()` initializes the default process group when running under
    torchrun or SLURM+srun.
    `current()` is side-effect free and just reads the current distributed state.
    """

    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.is_distributed and dist.is_available() and dist.is_initialized():
            dist.barrier(
                device_ids=[self.local_rank] if self.device.type == "cuda" else None
            )

    @classmethod
    def current(cls) -> DistributedContext:
        local_rank = _infer_local_rank()
        is_distributed = dist.is_available() and dist.is_initialized()
        device = select_device(local_rank=local_rank, is_distributed=is_distributed)

        if is_distributed:
            return cls(
                is_distributed=True,
                rank=int(dist.get_rank()),
                world_size=int(dist.get_world_size()),
                local_rank=local_rank,
                device=device,
            )

        return cls(
            is_distributed=False,
            rank=0,
            world_size=1,
            local_rank=local_rank,
            device=device,
        )

    @classmethod
    def setup(cls) -> DistributedContext:
        """
        Initialize torch.distributed if requested by environment variables.

        This uses the default `env://` initialization and supports both torchrun
        (RANK/WORLD_SIZE/LOCAL_RANK) and SLURM+srun (SLURM_PROCID/
        SLURM_NTASKS/SLURM_LOCALID).
        """
        if dist.is_available() and dist.is_initialized():
            return cls.current()

        wants_distributed = (
            "LOCAL_RANK" in os.environ
            or "RANK" in os.environ
            or "SLURM_PROCID" in os.environ
        )
        if not wants_distributed:
            return cls.current()

        if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
        if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
            os.environ["LOCAL_RANK"] = str(_infer_local_rank())

        rank_env_present = "LOCAL_RANK" in os.environ or "RANK" in os.environ
        missing = [
            k
            for k in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
            if k not in os.environ
        ]
        if rank_env_present and missing:
            raise ValueError(
                "Distributed run detected (LOCAL_RANK/RANK set) but missing required "
                f"environment variables for env:// init: {missing}."
            )

        if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
            return cls.current()

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        local_rank = _infer_local_rank()
        if torch.cuda.is_available():
            local_rank = _safe_cuda_local_rank(local_rank)
            os.environ["LOCAL_RANK"] = str(local_rank)
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend=backend,
                device_id=torch.device(f"cuda:{local_rank}"),
            )
        else:
            dist.init_process_group(backend=backend)
        dist_ctx = cls.current()
        device = select_device(local_rank=local_rank, is_distributed=True)
        return cls(
            is_distributed=dist_ctx.is_distributed,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
            local_rank=dist_ctx.local_rank,
            device=device,
        )

    @staticmethod
    def cleanup() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
