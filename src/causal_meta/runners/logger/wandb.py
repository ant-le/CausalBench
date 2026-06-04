from __future__ import annotations

import logging
import os
from types import ModuleType
from typing import Any, Dict, Mapping, Optional

from .base import BaseLogger

log = logging.getLogger(__name__)


def _get(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)  # type: ignore[no-any-return]
        except Exception:
            pass
    return getattr(obj, key, default)


def wandb_enabled(cfg: Any) -> bool:
    logger_cfg = _get(cfg, "logger", {})
    wandb_cfg = _get(logger_cfg, "wandb", {})
    return bool(_get(wandb_cfg, "enabled", False))


def import_wandb_if_enabled(enabled: bool) -> ModuleType | None:
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "W&B logging is enabled but 'wandb' is not installed. "
            "Install it (e.g., `pip install wandb`) or disable it via `logger.wandb.enabled=false`."
        ) from exc
    return wandb


class WandbLogger(BaseLogger):
    """
    WandB implementation of the BaseLogger.

    Supports run resumption: if `resume_run_id` is provided and the run exists,
    the logger will resume that run instead of creating a new one.
    """

    def __init__(
        self,
        cfg: Any,
        output_dir: Optional[str] = None,
        resume_run_id: Optional[str] = None,
    ):
        """
        Initialize the WandB logger.

        Args:
            cfg: Experiment configuration (DictConfig or dict-like).
            output_dir: Directory for W&B local files.
            resume_run_id: If provided, attempt to resume this W&B run ID.
        """
        # Keep as Any so type checkers accept wandb's dynamic API.
        wandb_module = import_wandb_if_enabled(True)
        if wandb_module is None:
            raise RuntimeError("WandbLogger initialized but wandb import failed.")
        self.wandb_module: Any = wandb_module

        from omegaconf import OmegaConf

        wandb_cfg = _get(_get(cfg, "logger", {}), "wandb", {})

        # Initialize run
        default_name = _get(cfg, "name", "experiment")
        model_cfg = _get(cfg, "model", {})
        model_type = _get(model_cfg, "type", None)
        if model_type:
            default_name = f"{default_name}_{model_type}"

        run_name: str
        try:
            run_name_cfg = _get(wandb_cfg, "name", None)
        except Exception:
            # If OmegaConf interpolation fails, fall back to an auto-generated name.
            run_name_cfg = None

        if isinstance(run_name_cfg, str) and run_name_cfg.strip():
            run_name = run_name_cfg
        else:
            run_name = str(default_name)
            # Best-effort uniqueness for Hydra multiruns.
            job_num = os.environ.get("HYDRA_JOB_NUM")
            if job_num not in {None, "None", ""}:
                run_name = f"{run_name}_job{job_num}"
            else:
                run_name = f"{run_name}_pid{os.getpid()}"

        # Handle run resumption
        resume_mode: Optional[str] = None
        run_id: Optional[str] = None
        if resume_run_id:
            resume_mode = "allow"  # Resume if exists, otherwise create new
            run_id = resume_run_id
            log.info(f"Attempting to resume W&B run: {resume_run_id}")

        self.run = self.wandb_module.init(
            project=_get(wandb_cfg, "project", "causal_meta"),
            entity=_get(wandb_cfg, "entity", None),
            name=run_name,
            id=run_id,
            resume=resume_mode,
            config=(
                OmegaConf.to_container(cfg, resolve=True)
                if OmegaConf.is_config(cfg)
                else cfg
            ),  # handle raw dict or DictConfig
            tags=_get(wandb_cfg, "tags", []),
            mode=_get(wandb_cfg, "mode", "offline"),
            dir=output_dir,
        )

        self.allowed_metrics: list[str] | None = _get(
            wandb_cfg, "allowed_metrics", None
        )

        # Store run ID for checkpoint persistence
        self._run_id: Optional[str] = self.run.id if self.run else None

    @property
    def run_id(self) -> Optional[str]:
        """Return the current W&B run ID for checkpoint persistence."""
        return self._run_id

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        if self.run:
            filtered_metrics = metrics
            if getattr(self, "allowed_metrics", None) is not None:
                filtered_metrics = {}
                for k, v in metrics.items():
                    base_name = k.split("/")[-1]
                    if base_name.endswith("_sem") or base_name.endswith("_std"):
                        continue

                    for allowed in self.allowed_metrics:
                        if allowed in base_name:
                            if allowed == "valid_dag_pct" and "threshold" in base_name:
                                continue
                            filtered_metrics[k] = v
                            break

            if filtered_metrics:
                if step is not None:
                    self.wandb_module.log(filtered_metrics, step=step)
                else:
                    self.wandb_module.log(filtered_metrics)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        if self.run:
            self.wandb_module.config.update(params, allow_val_change=True)

    def finish(self) -> None:
        if self.run:
            self.wandb_module.finish()
