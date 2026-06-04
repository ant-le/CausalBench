from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Mapping, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.models.base import BaseModel
from causal_meta.models.factory import ModelFactory
from causal_meta.models.random.edge_prior import (
    infer_edge_probability as _infer_edge_probability,
    maybe_fill_edge_prior,
)
from causal_meta.runners.logger.base import BaseLogger
from causal_meta.runners.logger.local import LocalLogger
from causal_meta.runners.logger.wandb import WandbLogger
from causal_meta.runners.tasks import analysis, evaluation, pre_training
from causal_meta.runners.utils.artifacts import resolve_output_dir
from causal_meta.runners.utils.distributed import DistributedContext
from causal_meta.runners.utils.env import log_environment_info
from causal_meta.runners.utils.seeding import get_experiment_seed, seed_everything

log = logging.getLogger(__name__)

_EXPLICIT_INFERENCE_PARAM_KEYS: dict[str, frozenset[str]] = {
    "dibs": frozenset(
        {
            "mode",
            "steps",
            "seed",
            "use_marginal",
            "xla_preallocate",
            "external_process",
            "external_python",
            "external_timeout_s",
            "alpha",
            "gamma_z",
            "gamma_theta",
            "n_particles",
            "profile_overrides",
        }
    ),
    "bayesdag": frozenset(
        {
            "variant",
            "lambda_sparse",
            "num_chains",
            "sinkhorn_n_iter",
            "scale_noise",
            "scale_noise_p",
            "batch_size",
            "max_epochs",
            "standardize_data_mean",
            "standardize_data_std",
            "save_dir",
            "sparse_init",
            "input_perm",
            "vi_norm",
            "norm_layers",
            "res_connection",
            "external_python",
            "external_timeout_s",
            "device",
            "skip_evaluation",
            "profile_overrides",
        }
    ),
}


# ── Backward-compatible re-export ──────────────────────────────────────
def infer_random_baseline_edge_probability(cfg: DictConfig) -> float:
    """Thin wrapper kept for backward compatibility.

    The implementation now lives in
    ``causal_meta.models.random.edge_prior.infer_edge_probability``.
    """
    return _infer_edge_probability(cfg)


def _load_wandb_run_id_from_checkpoint(checkpoint_dir: Path) -> str | None:
    """Load wandb_run_id from last.pt checkpoint if it exists."""
    checkpoint_path = checkpoint_dir / "checkpoints" / "last.pt"
    if not checkpoint_path.exists():
        return None
    try:
        # Load only the necessary keys to avoid loading full model state
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint.get("wandb_run_id")
    except Exception as exc:
        log.warning(f"Failed to load wandb_run_id from checkpoint: {exc}")
        return None


def _attach_file_logger(output_dir: Path, *, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / filename
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename) == log_path:
                    return
            except Exception:
                continue

    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def _maybe_load_best_checkpoint_for_eval(
    cfg: DictConfig,
    model: BaseModel,
    *,
    output_dir: Path,
) -> bool:
    """Load ``checkpoints/best.pt`` for evaluation when explicitly requested.

    Returns ``True`` when a checkpoint was loaded and pre-training should be
    skipped for amortized models.
    """
    use_best = bool(cfg.get("inference", {}).get("use_best_checkpoint_for_eval", False))
    if not use_best:
        return False

    if not model.needs_pretraining:
        raise ValueError(
            "inference.use_best_checkpoint_for_eval=true is only supported for "
            "amortized models with checkpoints."
        )

    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Requested best checkpoint for evaluation, but no checkpoint was found at {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain model_state_dict"
        )

    model.load_state_dict(model_state_dict)
    log.info("Loaded best checkpoint for evaluation from %s", checkpoint_path)
    return True


def _resolve_explicit_model_inference_params(
    cfg: DictConfig,
    model_type: str,
) -> dict[str, object]:
    """Resolve explicit-model constructor params from `cfg.inference`.

    Uses model-specific allowlists so scientific/runtime settings for DiBS and
    BayesDAG live in `inference.<model>` rather than being merged generically
    into arbitrary model constructors.
    """

    allowed_keys = _EXPLICIT_INFERENCE_PARAM_KEYS.get(model_type)
    if allowed_keys is None:
        return {}

    inference_cfg = cfg.get("inference", {})
    if not isinstance(inference_cfg, Mapping):
        return {}

    selected = inference_cfg.get(model_type)
    if not isinstance(selected, Mapping):
        legacy_explicit = inference_cfg.get("explicit", {})
        if isinstance(legacy_explicit, Mapping):
            nested = legacy_explicit.get(model_type)
            if isinstance(nested, Mapping):
                selected = nested

    if not isinstance(selected, Mapping):
        return {}

    return {
        str(key): (
            OmegaConf.to_container(value, resolve=False)
            if isinstance(value, DictConfig)
            else value
        )
        for key, value in selected.items()
        if str(key) in allowed_keys
    }


def _apply_model_specific_trainer_profile(cfg_obj: DictConfig) -> None:
    """Overlay a model-specific trainer profile while preserving explicit overrides.

    The selected profile is taken from ``model.trainer_profile``. Values in the
    current ``cfg.trainer`` that still match ``trainer/default.yaml`` are
    replaced by the profile values, while values already overridden by the root
    config (for example smoke-task reductions) are preserved.
    """

    model_cfg = getattr(cfg_obj, "model", None)
    trainer_cfg = getattr(cfg_obj, "trainer", None)
    if model_cfg is None or trainer_cfg is None:
        return

    profile_name = getattr(model_cfg, "trainer_profile", None)
    if profile_name in {None, "", "default"}:
        return

    trainer_dir = Path(__file__).resolve().parent / "configs" / "trainer"
    default_path = trainer_dir / "default.yaml"
    profile_path = trainer_dir / f"{profile_name}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Unknown trainer profile '{profile_name}' at {profile_path}"
        )

    baseline = OmegaConf.to_container(OmegaConf.load(default_path), resolve=False)
    profile = OmegaConf.to_container(OmegaConf.load(profile_path), resolve=False)
    current = OmegaConf.to_container(trainer_cfg, resolve=False)

    def _merge_over_baseline(base: object, prof: object, cur: object) -> object:
        if not isinstance(prof, dict):
            if base == cur:
                return prof
            return cur

        base_dict = base if isinstance(base, dict) else {}
        cur_dict = cur if isinstance(cur, dict) else {}
        merged: dict[str, object] = dict(cur_dict)
        for key, prof_value in prof.items():
            base_value = base_dict.get(key)
            current_has_key = key in cur_dict
            current_value = cur_dict.get(key)
            if isinstance(prof_value, dict):
                merged[key] = _merge_over_baseline(
                    base_value,
                    prof_value,
                    current_value,
                )
                continue
            if (not current_has_key) or current_value == base_value:
                merged[key] = prof_value
        return merged

    merged_trainer = _merge_over_baseline(baseline, profile, current)
    with open_dict(cfg_obj):
        cfg_obj.trainer = OmegaConf.create(merged_trainer)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)


def run_pipeline(cfg: DictConfig) -> None:
    # Check for analysis task (e.g. python main.py +task=analysis analysis.target_dir=...)
    if cfg.get("task") == "analysis":
        base_output_dir = resolve_output_dir(cfg)
        analysis.run(cfg, base_output_dir)
        return

    def _validate_experiment_config(cfg_obj: DictConfig) -> None:
        missing = []
        for key in ["name", "data", "inference"]:
            if not hasattr(cfg_obj, key):
                missing.append(key)
        if not hasattr(cfg_obj, "model"):
            missing.append("model")
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        model_cfg = cfg_obj.model
        if not hasattr(model_cfg, "type"):
            raise ValueError("Missing required model key: model.type")

        # Only require optimizer/training settings for amortized models.
        # Explicit inference models (e.g., DiBS/BayesDAG) do not use trainer.*.
        amortized_model_types = {"avici", "bcnp"}
        model_type = str(getattr(model_cfg, "type", "")).lower()
        if model_type in amortized_model_types:
            if not hasattr(cfg_obj, "trainer"):
                raise ValueError(
                    "Missing required config key for amortized models: trainer"
                )
            trainer = cfg_obj.trainer
            legacy_task_keys = [
                "max_steps",
                "max_tasks_seen",
                "log_every_n_tasks",
                "val_check_interval",
                "val_check_interval_tasks",
                "checkpoint_every_n_tasks",
                "scheduler_t_max",
                "scheduler_t_max_tasks",
                "scheduler_warmup_tasks",
                "scheduler_milestones_tasks",
                "lr_scale_with_global_batch_sqrt",
                "lr_reference_global_tasks_per_step",
                "accumulate_grad_batches",
                "regulariser_update_interval",
            ]
            present_legacy_keys = [
                key for key in legacy_task_keys if hasattr(trainer, key)
            ]
            if present_legacy_keys:
                raise ValueError(
                    "Legacy task-based amortized trainer keys are no longer supported: "
                    f"{present_legacy_keys}. Use max_optimizer_steps/log_every_n_steps/"
                    "val_check_interval_steps/checkpoint_every_n_steps, step-based scheduler keys, "
                    "regulariser_update_interval_steps, and target_global_tasks_per_step instead."
                )
            for key in ["lr", "max_optimizer_steps", "target_global_tasks_per_step"]:
                if not hasattr(trainer, key):
                    raise ValueError(f"Missing required trainer key: trainer.{key}")

    def _maybe_fill_model_num_nodes(cfg_obj: DictConfig) -> None:
        model_cfg = cfg_obj.model
        if hasattr(model_cfg, "num_nodes"):
            return

        inferred = None
        data_cfg = getattr(cfg_obj, "data", None)
        if data_cfg is not None:
            # Prefer explicit shared key when present.
            for k in ["n_nodes", "num_nodes"]:
                if hasattr(data_cfg, k):
                    inferred = int(getattr(data_cfg, k))
                    break
            if inferred is None and hasattr(data_cfg, "train_family"):
                train_family = getattr(data_cfg, "train_family")
                if hasattr(train_family, "n_nodes"):
                    inferred = int(getattr(train_family, "n_nodes"))

        if inferred is None:
            raise ValueError(
                "model.num_nodes is required when it cannot be inferred from data.*.n_nodes"
            )

        with open_dict(cfg_obj):
            cfg_obj.model.num_nodes = inferred

    def _expand_shared_architecture(cfg_obj: DictConfig) -> None:
        """Allow shared amortized architecture under `model.arch.*`."""

        model_cfg = cfg_obj.model
        model_type = str(getattr(model_cfg, "type", "")).lower()
        if model_type not in {"avici", "bcnp"}:
            return
        if not hasattr(model_cfg, "arch"):
            return

        arch_cfg = getattr(model_cfg, "arch")
        if arch_cfg is None:
            return

        with open_dict(cfg_obj):
            for k, v in arch_cfg.items():
                if not hasattr(cfg_obj.model, k):
                    setattr(cfg_obj.model, k, v)
            # Avoid passing nested dicts to model constructors.
            try:
                del cfg_obj.model["arch"]
            except Exception:
                pass

    # 0. Validate config early (fail fast on cluster)
    _validate_experiment_config(cfg)
    _maybe_fill_model_num_nodes(cfg)
    _expand_shared_architecture(cfg)
    _apply_model_specific_trainer_profile(cfg)
    maybe_fill_edge_prior(cfg)

    # 1. Distributed setup
    dist_ctx = DistributedContext.setup()
    is_distributed = dist_ctx.is_distributed
    local_rank = dist_ctx.local_rank
    use_wandb = cfg.get("logger", {}).get("wandb", {}).get("enabled", False)
    logger: BaseLogger | None = None

    # Use Hydra's per-job output directory as the single artifact root.
    base_output_dir = resolve_output_dir(cfg)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Configure logging to be less verbose on non-zero ranks
        if is_distributed and dist_ctx.rank != 0:
            logging.getLogger().setLevel(logging.WARNING)

        if (not is_distributed) or dist_ctx.is_main_process:
            log.info("Starting Pipeline...")
            log_environment_info()

        # 2. Seed process RNGs (dataset workers handle their own seeding internally)
        seed = get_experiment_seed(cfg, fallback=0)
        seed_everything(seed, deterministic=bool(cfg.get("deterministic", False)))

        # 2b. Optional TF32 (perf knob for transformer-heavy workloads)
        tf32_enabled = bool(cfg.get("trainer", {}).get("tf32", False))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
            torch.backends.cudnn.allow_tf32 = tf32_enabled

        # 3. Setup Data
        if (not is_distributed) or dist_ctx.is_main_process:
            log.info("Initializing Data Module...")

        data_module = CausalMetaModule.from_config(cfg.data)

        model_specific_cfg = cfg.model
        model_id = str(
            getattr(
                model_specific_cfg, "id", getattr(model_specific_cfg, "type", "model")
            )
        )
        if (not is_distributed) or dist_ctx.is_main_process:
            _attach_file_logger(base_output_dir, filename="main.log")
        if (not is_distributed) or dist_ctx.is_main_process:
            log.info(f"=== Running Model: {model_id} ===")

        dist_ctx.barrier()

        # Check for existing W&B run ID from checkpoint for resumption
        resume_run_id: str | None = None
        if use_wandb:
            resume_run_id = _load_wandb_run_id_from_checkpoint(base_output_dir)
            if resume_run_id and ((not is_distributed) or dist_ctx.is_main_process):
                log.info(f"Found W&B run ID in checkpoint: {resume_run_id}")

        # Initialize Logger once per run (Hydra multirun launches separate jobs per model).
        if (not is_distributed) or dist_ctx.is_main_process:
            if use_wandb:
                log.info("Initializing WandB Logger...")
                try:
                    logger = WandbLogger(
                        cfg,
                        output_dir=str(base_output_dir),
                        resume_run_id=resume_run_id,
                    )
                except Exception as exc:
                    log.warning(
                        "W&B initialization failed (%s). Falling back to LocalLogger. "
                        "Use logger.wandb.enabled=false or logger.wandb.mode=offline "
                        "to suppress this warning.",
                        exc,
                    )
                    logger = LocalLogger()
            else:
                log.info("Initializing Local Logger...")
                logger = LocalLogger()
        else:
            logger = LocalLogger()

        try:
            # Instantiate Model
            model_params = {
                k: v
                for k, v in model_specific_cfg.items()
                if k not in {"trainer", "inference", "arch"}
            }

            model_type = str(model_params.get("type", "")).lower()
            explicit_override = _resolve_explicit_model_inference_params(
                cfg,
                model_type,
            )
            if explicit_override:
                model_params.update(explicit_override)

            if model_type == "bayesdag" and "save_dir" not in model_params:
                model_params["save_dir"] = str(base_output_dir / "bayesdag_output")

            model = ModelFactory.create(model_params)

            device = dist_ctx.device
            model.to(device)

            if is_distributed:
                use_cuda_ddp = dist_ctx.device.type == "cuda"
                ddp_find_unused_parameters = model_type == "bcnp"
                model = DDP(
                    model,
                    device_ids=[local_rank] if use_cuda_ddp else None,
                    output_device=local_rank if use_cuda_ddp else None,
                    find_unused_parameters=ddp_find_unused_parameters,
                )

            # 5/6. Pre-training (amortized models only)
            model_unwrapped_raw = model.module if is_distributed else model
            model_unwrapped = cast(BaseModel, model_unwrapped_raw)
            if not isinstance(model_unwrapped, BaseModel):
                raise TypeError(
                    "Expected model to be a BaseModel or DDP-wrapped BaseModel."
                )

            loaded_best_checkpoint_for_eval = _maybe_load_best_checkpoint_for_eval(
                cfg,
                model_unwrapped,
                output_dir=base_output_dir,
            )

            if model_unwrapped.needs_pretraining:
                if loaded_best_checkpoint_for_eval:
                    if (not is_distributed) or dist_ctx.is_main_process:
                        log.info(
                            "--- Phase 1: Pre-Training Skipped (loaded best checkpoint) ---"
                        )
                else:
                    pre_training.run(
                        cfg,
                        model,
                        data_module,
                        logger=logger,
                        output_dir=base_output_dir,
                    )
                    if (not is_distributed) or dist_ctx.is_main_process:
                        log.info("--- Phase 1: Pre-Training ---")

            # 7. Evaluation
            if (not is_distributed) or dist_ctx.is_main_process:
                if model_unwrapped.needs_pretraining:
                    log.info("--- Phase 2: Evaluation ---")
                else:
                    log.info("--- Evaluation ---")
            evaluation.run(
                cfg,
                model,
                data_module,
                logger=logger,
                output_dir=base_output_dir,
            )
        finally:
            if logger is not None:
                logger.finish()

        if (not is_distributed) or dist_ctx.is_main_process:
            log.info("Pipeline Finished.")
    except Exception:
        if (not is_distributed) or dist_ctx.is_main_process:
            log.error("Pipeline failed.")
            print("[causal_meta.main][ERROR] Pipeline failed.", flush=True)
            traceback.print_exc()
        raise
    finally:
        # Cleanup distributed context at the very end
        try:
            DistributedContext.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()
