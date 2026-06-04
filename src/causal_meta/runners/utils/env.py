from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys

import torch

from causal_meta import __version__

log = logging.getLogger(__name__)


def _get_git_revision() -> str:
    """Return short git commit hash + dirty flag, or 'unknown'."""
    try:
        rev = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return f"{rev}-dirty" if dirty else rev
    except Exception:
        return "unknown"


def log_environment_info() -> None:
    """Logs lightweight environment information for debugging/reproducibility."""
    info = [
        f"causal_meta: {__version__} (git: {_get_git_revision()})",
        f"Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})",
        f"PyTorch: {torch.__version__}",
        f"CUDA Available: {torch.cuda.is_available()}",
        f"MPS Available: {torch.backends.mps.is_available()}",
    ]

    if torch.cuda.is_available():
        info.append(f"CUDA Device Count: {torch.cuda.device_count()}")
        current = torch.cuda.current_device()
        info.append(f"Current Device: {torch.cuda.get_device_name(current)}")

    # Optional: Check for Slurm
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        info.append(f"SLURM_JOB_ID: {slurm_job_id}")

    log.info("Environment Report:\n" + "\n".join([f"  - {line}" for line in info]))
