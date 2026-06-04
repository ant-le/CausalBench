from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """
    Seed Python/NumPy/PyTorch RNGs for reproducible experiments.

    Notes:
    - This seeds process-level RNGs. It does not make `IterableDataset` resume deterministic.
    - For strict determinism you may need to disable non-deterministic ops and set
      environment variables depending on your CUDA/cuDNN setup.
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def get_experiment_seed(cfg: object, *, fallback: int = 0) -> int:
    """
    Best-effort extraction of an experiment seed from Hydra/OmegaConf configs.
    Prefers top-level `seed`, otherwise falls back to `data.base_seed`.
    """
    seed: Optional[int] = None

    try:
        seed_val = getattr(cfg, "seed")
        if seed_val is not None:
            seed = int(seed_val)
    except Exception:
        seed = None

    if seed is None:
        try:
            data = getattr(cfg, "data")
            base_seed = getattr(data, "base_seed", None)
            if base_seed is not None:
                seed = int(base_seed)
        except Exception:
            seed = None

    return int(fallback if seed is None else seed)

