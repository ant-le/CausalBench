"""Shared helpers for task modules (pre-training, evaluation, inference).

Centralises small utilities that were previously duplicated across multiple
task files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from causal_meta.models.base import BaseModel
from causal_meta.runners.utils.distributed import DistributedContext

log = logging.getLogger(__name__)


# ── Model helpers ──────────────────────────────────────────────────────


def unwrap_model(model: nn.Module) -> BaseModel:
    """Unwrap a (possibly DDP-wrapped) model and cast to ``BaseModel``.

    Uses structural (duck) typing — the caller is responsible for ensuring
    the underlying model satisfies the ``BaseModel`` interface.
    """
    raw = model.module if isinstance(model, DDP) else model
    return cast(BaseModel, raw)


def infer_device(model: nn.Module, dist_ctx: DistributedContext) -> torch.device:
    """Best-effort device inference from model parameters.

    Falls back to ``dist_ctx.device`` when the model has no parameters
    (e.g. explicit / non-amortised baselines).
    """
    params = list(model.parameters())
    return params[0].device if params else dist_ctx.device


def sampling_mode(model: BaseModel) -> str:
    """Return ``"external"`` if the model uses an external Python process."""
    external_process = getattr(model, "external_process", False)
    return "external" if external_process else "in_process"


# ── Data / sharding helpers ───────────────────────────────────────────


def shard_indices(n: int, rank: int, world_size: int) -> range:
    """Deterministic, no-padding sharding: rank *i* handles indices ``i, i+ws, …``."""
    return range(rank, n, world_size)


# ── Inference / cache helpers ─────────────────────────────────────────


def resolve_inference_root(
    cfg: DictConfig,
    output_dir: Path,
) -> Path:
    """Determine the root directory for inference artifacts.

    Precedence:
    1. ``cfg.inference.cache_dir`` (persistent shared cache).
    2. ``output_dir / "inference"`` (run-specific).
    """
    cache_dir = cfg.get("inference", {}).get("cache_dir", None)
    if cache_dir:
        return Path(str(cache_dir))
    return output_dir / "inference"
