from __future__ import annotations

import contextlib
import inspect
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.models.base import BaseModel
from causal_meta.runners.logger.base import BaseLogger
from causal_meta.runners.metrics.graph import Metrics
from causal_meta.runners.tasks.utils import infer_device, unwrap_model
from causal_meta.runners.utils.artifacts import resolve_output_dir
from causal_meta.runners.utils.distributed import DistributedContext
from causal_meta.runners.utils.seeding import get_experiment_seed

log = logging.getLogger(__name__)


def _is_avici_model_type(model_type: str) -> bool:
    return model_type.strip().lower() == "avici"


@dataclass(frozen=True)
class _TrainingPlan:
    train_batch_size: int
    accumulate_grad_batches: int
    global_tasks_per_step: int
    max_optimizer_steps: int
    log_every_n_steps: int
    val_check_interval_steps: int
    checkpoint_every_n_steps: int


DEFAULT_VALIDATION_METRICS = (
    "e-edgef1",
    "ne-sid",
    "ne-shd",
    "valid_dag_pct",
)
AVICI_SAMPLED_VALIDATION_METRICS = (
    "sampled-e-edgef1",
    "sampled-ne-sid",
    "sampled-ne-shd",
    "sampled-valid_dag_pct",
    "sampled-graph_nll",
    "sampled-graph_nll_per_edge",
    "sampled-edge_entropy",
    "sampled-ece",
)
DEFAULT_VALIDATION_GROUP_PREFIXES = {
    "id": ("id_",),
    "ood_graph": ("ood_graph_",),
    "ood_mech": ("ood_mech_",),
    "ood_noise": ("ood_noise_",),
    "ood_both": ("ood_both_",),
    "ood_nodes": ("ood_nodes_",),
    "ood_samples": ("ood_samples_",),
}


def run(
    cfg: DictConfig,
    model: nn.Module,
    data_module: CausalMetaModule,
    *,
    logger: BaseLogger | None = None,
    output_dir: Path | None = None,
) -> None:
    """
    Executes the pre-training loop for meta-learning models (e.g., Avici, BCNP).

    This function handles:
    1.  DDP setup and model wrapping.
    2.  Loading the infinite training stream (`train_dataloader`).
    3.  Optimization loop (forward pass, loss calculation, backward pass).
    4.  Periodic validation and checkpointing (best/last).
    5.  Logging to the provided logger (e.g., WandB).

    Args:
        cfg: Experiment configuration.
        model: The model instance to train.
        data_module: The data module providing training and validation dataloaders.
        logger: Optional logger instance (BaseLogger).
        output_dir: Directory to save checkpoints and logs.
    """
    # 0. DDP Utilities
    dist_ctx = DistributedContext.current()
    is_distributed = dist_ctx.is_distributed
    rank = dist_ctx.rank
    world_size = dist_ctx.world_size

    # 1. Setup
    if rank == 0:
        log.info(f"Starting experiment: {cfg.name}")
        log.info(f"Working directory: {os.getcwd()}")

    # Device is already set in pipe.py, but we can double check or get it from model
    device = infer_device(model, dist_ctx)

    train_batch_size_cap = int(getattr(cfg.data, "batch_size_train", 1))
    if train_batch_size_cap < 1:
        raise ValueError("data.batch_size_train must be >= 1")

    training_plan = _resolve_training_plan(
        cfg,
        max_per_device_batch_size=train_batch_size_cap,
        world_size=world_size,
    )
    train_batch_size = training_plan.train_batch_size
    accumulate_grad_batches = training_plan.accumulate_grad_batches
    global_tasks_per_step = training_plan.global_tasks_per_step

    # 2. Data
    # train_dataloader handles DDP seeding internally
    train_loader = data_module.train_dataloader(batch_size_override=train_batch_size)

    # Unwrap model for attribute access
    model_unwrapped = unwrap_model(model)

    total_params, trainable_params = _count_parameters(model_unwrapped)

    if rank == 0:
        log.info(
            "Model parameters | total=%s | trainable=%s",
            f"{total_params:,}",
            f"{trainable_params:,}",
        )
        log.info(
            "Training plan | batch/device=%d | grad_accum=%d | global_tasks/step=%d | max_steps=%d",
            train_batch_size,
            accumulate_grad_batches,
            global_tasks_per_step,
            training_plan.max_optimizer_steps,
        )
        if logger:
            logger.log_hyperparams(
                {
                    "model/total_parameters": total_params,
                    "model/trainable_parameters": trainable_params,
                    "trainer/train_batch_size_per_device": train_batch_size,
                    "trainer/accumulate_grad_batches": accumulate_grad_batches,
                    "trainer/global_tasks_per_step": global_tasks_per_step,
                    "trainer/max_optimizer_steps": training_plan.max_optimizer_steps,
                }
            )

    if not model_unwrapped.needs_pretraining:
        if rank == 0:
            log.info("Model does not require pre-training. Exiting.")
        return

    # 3. Optimizer
    lr = float(cfg.trainer.lr)
    weight_decay = float(cfg.trainer.get("weight_decay", 1e-4))
    betas = _maybe_parse_betas(cfg.trainer.get("optimizer_betas", None))
    optimizer_eps = _maybe_parse_eps(cfg.trainer.get("optimizer_eps", None))

    optimizer_kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
    if betas is not None:
        optimizer_kwargs["betas"] = betas
    if optimizer_eps is not None:
        optimizer_kwargs["eps"] = optimizer_eps
    if device.type == "cuda" and _adamw_supports_fused():
        optimizer_kwargs["fused"] = bool(cfg.trainer.get("optimizer_fused", True))
    optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)

    scheduler = _build_scheduler(optimizer, cfg)

    amp_enabled = bool(cfg.trainer.get("amp", False)) and device.type == "cuda"
    amp_dtype_name = str(cfg.trainer.get("amp_dtype", "bf16")).lower()
    if amp_dtype_name in {"bf16", "bfloat16"}:
        amp_dtype = torch.bfloat16
        scaler = GradScaler("cuda", enabled=False)
    elif amp_dtype_name in {"fp16", "float16"}:
        amp_dtype = torch.float16
        scaler = GradScaler("cuda", enabled=amp_enabled)
    else:
        raise ValueError("trainer.amp_dtype must be one of: bf16, fp16")

    # 4. Training Loop
    max_optimizer_steps = training_plan.max_optimizer_steps
    log_every_n_steps = training_plan.log_every_n_steps
    val_check_interval_steps = training_plan.val_check_interval_steps
    checkpoint_every_n_steps = training_plan.checkpoint_every_n_steps
    regulariser_update_interval = int(
        cfg.trainer.get("regulariser_update_interval_steps", 0)
    )
    grad_clip_norm = float(cfg.trainer.get("grad_clip_norm", 1.0))
    validation_n_samples = int(
        cfg.trainer.get(
            "validation_n_samples",
            cfg.get("inference", {}).get("n_samples", 10),
        )
    )
    validation_metrics = _validation_metrics_from_config(cfg)
    validation_log_per_family = bool(
        cfg.trainer.get("validation_log_per_family", False)
    )
    validation_group_prefixes = _validation_group_prefixes_from_config(cfg)
    validation_selection_metric = str(
        cfg.trainer.get("validation_selection_metric", "mean_id_e-edgef1")
    )
    validation_selection_mode = _validation_selection_mode_from_config(cfg)

    best_val_metric = (
        float("-inf")
        if _should_maximize_selection_metric(
            validation_selection_metric,
            validation_selection_mode,
        )
        else float("inf")
    )
    best_val_metric_name: str | None = None
    step = 0
    tasks_seen = 0

    if rank == 0:
        log.info("Starting training loop...")

    # Determine output directory
    output_dir = resolve_output_dir(cfg, output_dir)

    path = output_dir / "checkpoints"
    if rank == 0:
        path.mkdir(parents=True, exist_ok=True)

    # RESUME LOGIC
    start_step = 0
    resume_path = path / "last.pt"
    train_stream_initial_base_seed = int(getattr(cfg.data, "base_seed", 0))
    if resume_path.exists():
        if rank == 0:
            log.info(f"Resuming from checkpoint: {resume_path}")
        # Load to CPU for portability across nodes/devices.
        checkpoint = torch.load(resume_path, map_location="cpu")

        model_unwrapped.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        saved_best_val_metric = checkpoint.get("best_val_metric")
        if saved_best_val_metric is not None:
            saved_selection_metric = checkpoint.get("validation_selection_metric")
            saved_selection_mode = checkpoint.get("validation_selection_mode")
            can_restore_best = (
                saved_selection_metric is None
                or str(saved_selection_metric) == validation_selection_metric
            ) and (
                saved_selection_mode is None
                or str(saved_selection_mode) == validation_selection_mode
            )
            if can_restore_best:
                best_val_metric = float(saved_best_val_metric)
                saved_best_name = checkpoint.get("best_val_metric_name")
                best_val_metric_name = (
                    str(saved_best_name)
                    if saved_best_name is not None
                    else validation_selection_metric
                )
                if rank == 0 and math.isfinite(best_val_metric):
                    log.info(
                        "Restored best validation metric from checkpoint: %s=%.4f",
                        best_val_metric_name,
                        best_val_metric,
                    )
            elif rank == 0:
                log.warning(
                    "Checkpoint best validation metric was not restored because "
                    "trainer.validation_selection_metric/mode changed."
                )

        start_step = int(checkpoint["step"])
        step = start_step

        train_stream_initial_base_seed = int(
            checkpoint.get(
                "train_stream_initial_base_seed", train_stream_initial_base_seed
            )
        )

        if rank == 0:
            log.info(f"Resumed at step {step}")

        saved_global_tasks_per_step = int(
            checkpoint.get("train_stream_global_tasks_per_step", global_tasks_per_step)
        )
        if saved_global_tasks_per_step != global_tasks_per_step and rank == 0:
            log.warning(
                "Checkpoint global_tasks_per_step=%d differs from current global_tasks_per_step=%d; "
                "training data stream resume will be best-effort.",
                saved_global_tasks_per_step,
                global_tasks_per_step,
            )

        saved_world_size = int(checkpoint.get("train_stream_world_size", world_size))
        if saved_world_size != world_size and rank == 0:
            log.warning(
                f"Checkpoint world_size={saved_world_size} differs from current world_size={world_size}; "
                "training data stream resume will be best-effort."
            )

        saved_batch_size = int(
            checkpoint.get("train_stream_batch_size_train", train_batch_size)
        )
        if saved_batch_size != train_batch_size and rank == 0:
            log.warning(
                f"Checkpoint batch_size_train={saved_batch_size} differs from current batch_size_train={train_batch_size}; "
                "training data stream resume will be best-effort."
            )

        saved_accum = int(
            checkpoint.get(
                "train_stream_accumulate_grad_batches", accumulate_grad_batches
            )
        )
        if saved_accum != accumulate_grad_batches and rank == 0:
            log.warning(
                f"Checkpoint accumulate_grad_batches={saved_accum} differs from current accumulate_grad_batches={accumulate_grad_batches}; "
                "training data stream resume will be best-effort."
            )

        if getattr(cfg.data, "num_workers", 0) > 0:
            log.warning(
                "Resuming with an IterableDataset stream is not strictly reproducible when num_workers>0; "
                "the data stream may differ from the original run."
            )

        tasks_seen = int(
            checkpoint.get(
                "tasks_seen",
                start_step * saved_global_tasks_per_step,
            )
        )
        if rank == 0:
            log.info("Resumed at tasks_seen=%d", tasks_seen)

    # Deterministic-ish resume for the streaming dataset when DataLoader uses num_workers=0.
    # With num_workers=0, MetaIterableDataset uses a single stream per rank with stride=world_size.
    # If batch_size_train or gradient accumulation is used, each optimizer step consumes
    # multiple stream items per rank.
    if (
        hasattr(data_module, "train_dataset")
        and data_module.train_dataset is not None
        and getattr(cfg.data, "num_workers", 0) == 0
    ):
        data_module.train_dataset.base_seed = (
            train_stream_initial_base_seed + tasks_seen
        )
        if rank == 0 and start_step > 0:
            log.info(
                f"Resumed train stream base_seed={data_module.train_dataset.base_seed} "
                f"(initial={train_stream_initial_base_seed}, tasks_seen={tasks_seen}, world_size={world_size}, "
                f"batch_size_train={train_batch_size}, accumulate_grad_batches={accumulate_grad_batches})."
            )

    next_log_step = _next_step_threshold(step, log_every_n_steps)
    next_val_step = _next_step_threshold(step, val_check_interval_steps)
    next_checkpoint_step = _next_step_threshold(step, checkpoint_every_n_steps)

    train_iter = iter(train_loader)

    while step < max_optimizer_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Loss is tracked as a scalar mean across micro-batches.
        loss_accum = torch.tensor(0.0, device=device)

        next_step = step + 1
        update_regulariser = (
            regulariser_update_interval > 0
            and next_step % regulariser_update_interval == 0
        )
        grad_norm_value: float | None = None

        for micro in range(accumulate_grad_batches):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_data, node_mask = _prepare_model_batch(batch, device)
            adjacency_matrix = batch["adjacency"].to(device)

            # Skip redundant DDP gradient sync on non-final micro-batches.
            is_last_micro = micro == accumulate_grad_batches - 1
            sync_ctx: contextlib.AbstractContextManager[None] = (
                contextlib.nullcontext()
                if (not is_distributed or is_last_micro)
                else cast(DDP, model).no_sync()
            )

            with sync_ctx:
                if amp_enabled:
                    with torch.autocast(
                        device_type="cuda", dtype=amp_dtype, enabled=True
                    ):
                        output = model(input_data, mask=node_mask)
                        loss_vec = model_unwrapped.calculate_loss(
                            output,
                            adjacency_matrix,
                            node_mask=node_mask,
                            update_regulariser=update_regulariser and is_last_micro,
                        )
                        loss = loss_vec.mean() / float(accumulate_grad_batches)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    output = model(input_data, mask=node_mask)
                    loss_vec = model_unwrapped.calculate_loss(
                        output,
                        adjacency_matrix,
                        node_mask=node_mask,
                        update_regulariser=update_regulariser and is_last_micro,
                    )
                    loss = loss_vec.mean() / float(accumulate_grad_batches)
                    loss.backward()

            loss_accum = loss_accum + loss.detach()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            grad_norm_value = _compute_grad_norm(model.parameters())
            _maybe_clip_grad_norm(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm_value = _compute_grad_norm(model.parameters())
            _maybe_clip_grad_norm(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        step += 1
        tasks_seen += global_tasks_per_step

        # Logging
        if (
            log_every_n_steps > 0
            and next_log_step is not None
            and step >= next_log_step
        ):
            loss_value = _reduce_loss_for_logging(loss_accum, world_size)
            if rank == 0:
                current_lr = _current_learning_rate(optimizer)
                grad_norm_str = (
                    f"{grad_norm_value:.4f}"
                    if grad_norm_value is not None and math.isfinite(grad_norm_value)
                    else "nan"
                )
                log_str = (
                    f"Step {step:,}/{max_optimizer_steps:,} | Tasks {tasks_seen:,} | Loss: {loss_value:.4f} | "
                    f"LR: {current_lr:.6g} | GradNorm: {grad_norm_str}"
                )
                log.info(log_str)

                if logger:
                    logger.log_metrics(
                        {
                            "train/loss": loss_value,
                            "train/lr": current_lr,
                            "train/grad_norm": (
                                float(grad_norm_value)
                                if grad_norm_value is not None
                                else float("nan")
                            ),
                            "train/tasks_seen": float(tasks_seen),
                            "train/global_tasks_per_step": float(global_tasks_per_step),
                            "train/optimizer_step": float(step),
                        },
                        step=step,
                    )
            next_log_step = _advance_step_threshold(
                next_log_step,
                log_every_n_steps,
                step,
            )

        # Validation
        if (
            val_check_interval_steps > 0
            and next_val_step is not None
            and step >= next_val_step
        ):
            val_metrics = validate(
                model_unwrapped,
                data_module,
                device,
                num_samples=validation_n_samples,
                metrics=validation_metrics,
                group_prefixes=validation_group_prefixes,
            )

            if rank == 0:
                log.info(
                    "Validation at step=%d (tasks_seen=%d): %s",
                    step,
                    tasks_seen,
                    val_metrics,
                )

                if logger:
                    if validation_log_per_family:
                        per_family_payload = {
                            f"val_family/{k}": v
                            for k, v in val_metrics.items()
                            if "/" in k
                        }
                        if per_family_payload:
                            logger.log_metrics(per_family_payload, step=step)

                    aggregate_payload = {
                        f"val/{k}": v
                        for k, v in val_metrics.items()
                        if k.startswith("mean_") and "/" not in k
                    }
                    if aggregate_payload:
                        logger.log_metrics(aggregate_payload, step=step)

                current_metric_name, current_metric = (
                    _resolve_validation_selection_metric(
                        val_metrics,
                        validation_selection_metric,
                    )
                )
                maximize_metric = _should_maximize_selection_metric(
                    current_metric_name,
                    validation_selection_mode,
                )
                if _is_metric_improved(
                    current_metric,
                    best_val_metric,
                    maximize=maximize_metric,
                ):
                    best_val_metric = current_metric
                    best_val_metric_name = current_metric_name
                    save_checkpoint(
                        cfg,
                        model_unwrapped,
                        optimizer,
                        scheduler,
                        scaler,
                        step,
                        path / "best.pt",
                        tasks_seen=tasks_seen,
                        train_stream_initial_base_seed=train_stream_initial_base_seed,
                        global_tasks_per_step=global_tasks_per_step,
                        world_size=world_size,
                        train_batch_size=train_batch_size,
                        accumulate_grad_batches=accumulate_grad_batches,
                        best_val_metric=best_val_metric,
                        best_val_metric_name=best_val_metric_name,
                        validation_selection_metric=validation_selection_metric,
                        validation_selection_mode=validation_selection_mode,
                        wandb_run_id=logger.run_id if logger else None,
                    )
                    log.info(
                        "New best model saved! %s: %.4f",
                        current_metric_name,
                        best_val_metric,
                    )
            next_val_step = _advance_step_threshold(
                next_val_step,
                val_check_interval_steps,
                step,
            )

        # Periodic "last" checkpoint (rank 0 only)
        if (
            checkpoint_every_n_steps > 0
            and next_checkpoint_step is not None
            and step >= next_checkpoint_step
        ):
            if rank == 0:
                save_checkpoint(
                    cfg,
                    model_unwrapped,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    path / "last.pt",
                    tasks_seen=tasks_seen,
                    train_stream_initial_base_seed=train_stream_initial_base_seed,
                    global_tasks_per_step=global_tasks_per_step,
                    world_size=world_size,
                    train_batch_size=train_batch_size,
                    accumulate_grad_batches=accumulate_grad_batches,
                    best_val_metric=best_val_metric,
                    best_val_metric_name=best_val_metric_name,
                    validation_selection_metric=validation_selection_metric,
                    validation_selection_mode=validation_selection_mode,
                    wandb_run_id=logger.run_id if logger else None,
                )
            next_checkpoint_step = _advance_step_threshold(
                next_checkpoint_step,
                checkpoint_every_n_steps,
                step,
            )

    if rank == 0:
        log.info("Training finished.")
        save_checkpoint(
            cfg,
            model_unwrapped,
            optimizer,
            scheduler,
            scaler,
            step,
            path / "last.pt",
            tasks_seen=tasks_seen,
            train_stream_initial_base_seed=train_stream_initial_base_seed,
            global_tasks_per_step=global_tasks_per_step,
            world_size=world_size,
            train_batch_size=train_batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            best_val_metric=best_val_metric,
            best_val_metric_name=best_val_metric_name,
            validation_selection_metric=validation_selection_metric,
            validation_selection_mode=validation_selection_mode,
            wandb_run_id=logger.run_id if logger else None,
        )

    if is_distributed:
        dist_ctx.barrier()  # Wait for rank 0 to save

    best_path = path / "best.pt"
    if best_path.exists():
        if rank == 0:
            log.info(f"Reloading best model from {best_path}")

        checkpoint = torch.load(best_path, map_location="cpu")
        model_unwrapped.load_state_dict(checkpoint["model_state_dict"])

    if is_distributed:
        dist_ctx.barrier()


def validate(
    model: BaseModel,
    data_module: CausalMetaModule,
    device: torch.device,
    num_samples: int = 10,
    metrics: Sequence[str] | None = None,
    group_prefixes: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    model.eval()
    val_loaders = data_module.val_dataloader()
    is_avici = _is_avici_model_type(type(model).__name__.replace("Model", ""))

    # One metrics handler per family (single-dataset assumption)
    per_family_metrics: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for name, loader in val_loaders.items():
            metrics_handler = Metrics(
                metrics=list(metrics or DEFAULT_VALIDATION_METRICS)
            )
            for batch in loader:
                input_data, node_mask = _prepare_model_batch(batch, device)
                adjacency_matrix = batch["adjacency"].to(device)

                edge_probs = None
                thresholded_graph = None
                if is_avici:
                    edge_probs = model.predict_probabilities(input_data, mask=node_mask)
                    thresholded_graph = (edge_probs > 0.5).to(dtype=edge_probs.dtype)
                    thresholded_graph = thresholded_graph.unsqueeze(0)

                samples = model.sample(
                    input_data, num_samples=num_samples, mask=node_mask
                )
                samples_for_metrics = samples.permute(1, 0, 2, 3)

                metrics_handler.update(
                    adjacency_matrix,
                    samples_for_metrics,
                    thresholded_graph=thresholded_graph,
                    edge_probs=edge_probs,
                )

            family_results = metrics_handler.compute(summary_stats=False)
            per_family_metrics[name] = family_results

    # Build flat output: {family/metric: value} + {metric: global_mean}
    avg_metrics: dict[str, float] = {}
    all_metric_keys: set[str] = set()
    for family_results in per_family_metrics.values():
        all_metric_keys.update(family_results.keys())

    for name, family_results in per_family_metrics.items():
        for k, v in family_results.items():
            avg_metrics[f"{name}/{k}"] = v

    # Compute global means per metric key
    for k in all_metric_keys:
        values = [fm[k] for fm in per_family_metrics.values() if k in fm]
        if values:
            avg_metrics[k] = sum(values) / len(values)

    _augment_validation_group_metrics(
        avg_metrics,
        group_prefixes=group_prefixes or DEFAULT_VALIDATION_GROUP_PREFIXES,
    )

    # Calculate mean F1 across all datasets (Alias for checkpointing)
    if "mean_id_e-edgef1" in avg_metrics:
        avg_metrics["mean_e-edgef1"] = avg_metrics["mean_id_e-edgef1"]
    elif "e-edgef1" in avg_metrics:
        avg_metrics["mean_e-edgef1"] = avg_metrics["e-edgef1"]

    return avg_metrics


def _augment_validation_group_metrics(
    metrics: dict[str, float],
    *,
    group_prefixes: Mapping[str, Sequence[str]],
) -> None:
    """Add grouped ID/OOD means for validation monitoring.

    Args:
        metrics: In-place metric dictionary from ``Metrics.compute``.
    """
    grouped: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        if "/" not in key:
            continue
        family_name, metric_name = key.split("/", 1)
        grouped.setdefault(metric_name, {})[family_name] = float(value)

    for metric_name, family_values in grouped.items():
        all_ood_values: list[float] = []
        for group_name, prefixes in group_prefixes.items():
            normalized_prefixes = tuple(str(prefix) for prefix in prefixes)
            group_values = [
                val
                for family, val in family_values.items()
                if any(family.startswith(prefix) for prefix in normalized_prefixes)
            ]
            if not group_values:
                continue
            metrics[f"mean_{group_name}_{metric_name}"] = float(
                sum(group_values) / len(group_values)
            )
            if group_name.startswith("ood"):
                all_ood_values.extend(group_values)

        if all_ood_values:
            metrics[f"mean_ood_{metric_name}"] = float(
                sum(all_ood_values) / len(all_ood_values)
            )


def _validation_metrics_from_config(cfg: DictConfig) -> list[str]:
    raw_metrics = cfg.trainer.get("validation_metrics", DEFAULT_VALIDATION_METRICS)
    metrics = [str(metric) for metric in raw_metrics]
    model_type = (
        str(getattr(cfg.get("model", {}), "type", "")) if cfg.get("model") else ""
    )
    if _is_avici_model_type(model_type):
        for metric in AVICI_SAMPLED_VALIDATION_METRICS:
            if metric not in metrics:
                metrics.append(metric)
        metrics = [metric for metric in metrics if metric != "threshold_valid_dag_pct"]
    if not metrics:
        raise ValueError("trainer.validation_metrics must contain at least one metric.")
    return metrics


def _validation_group_prefixes_from_config(
    cfg: DictConfig,
) -> dict[str, tuple[str, ...]]:
    raw_prefixes = cfg.trainer.get(
        "validation_group_prefixes", DEFAULT_VALIDATION_GROUP_PREFIXES
    )
    parsed: dict[str, tuple[str, ...]] = {}
    for group_name, prefixes in raw_prefixes.items():
        if isinstance(prefixes, str):
            normalized = (prefixes,)
        else:
            normalized = tuple(str(prefix) for prefix in prefixes)
        if not normalized:
            raise ValueError(
                f"trainer.validation_group_prefixes.{group_name} must not be empty."
            )
        parsed[str(group_name)] = normalized
    if not parsed:
        raise ValueError("trainer.validation_group_prefixes must not be empty.")
    return parsed


def _prepare_model_batch(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prepare a model-ready batch tensor and optional node mask."""
    input_data = batch["data"].to(device)
    intervention_mask = batch.get("intervention_mask")
    if intervention_mask is not None:
        input_data = torch.stack(
            [input_data, intervention_mask.to(device)],
            dim=-1,
        )

    node_mask = batch.get("node_mask")
    if node_mask is not None:
        node_mask = node_mask.to(device)

    return input_data, node_mask


def save_checkpoint(
    cfg: DictConfig,
    model: BaseModel,
    optimizer: optim.Optimizer,
    scheduler: Any | None,
    scaler: GradScaler,
    step: int,
    filepath: Path,
    *,
    tasks_seen: int,
    train_stream_initial_base_seed: int,
    global_tasks_per_step: int,
    world_size: int,
    train_batch_size: int,
    accumulate_grad_batches: int,
    best_val_metric: float | None = None,
    best_val_metric_name: str | None = None,
    validation_selection_metric: str | None = None,
    validation_selection_mode: str | None = None,
    wandb_run_id: str | None = None,
) -> None:
    experiment_seed = get_experiment_seed(
        cfg, fallback=int(train_stream_initial_base_seed)
    )
    state: dict[str, Any] = {
        "step": step,
        "tasks_seen": int(tasks_seen),
        "experiment_seed": int(experiment_seed),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_stream_initial_base_seed": int(train_stream_initial_base_seed),
        "train_stream_global_tasks_per_step": int(global_tasks_per_step),
        "train_stream_world_size": int(world_size),
        "train_stream_batch_size_train": int(train_batch_size),
        "train_stream_accumulate_grad_batches": int(accumulate_grad_batches),
        "train_stream_next_base_seed_if_num_workers_0": int(
            train_stream_initial_base_seed + tasks_seen
        ),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if best_val_metric is not None:
        state["best_val_metric"] = float(best_val_metric)
    if best_val_metric_name is not None:
        state["best_val_metric_name"] = str(best_val_metric_name)
    if validation_selection_metric is not None:
        state["validation_selection_metric"] = str(validation_selection_metric)
    if validation_selection_mode is not None:
        state["validation_selection_mode"] = str(validation_selection_mode)
    if wandb_run_id is not None:
        state["wandb_run_id"] = str(wandb_run_id)
    torch.save(
        state,
        filepath,
    )


def _build_scheduler(
    optimizer: optim.Optimizer,
    cfg: DictConfig,
) -> Any | None:
    scheduler_type = str(cfg.trainer.get("scheduler", "cosine")).lower()
    if scheduler_type in {"none", "off", "false"}:
        return None
    if scheduler_type not in {"cosine", "multistep"}:
        raise ValueError(
            "trainer.scheduler must be one of: cosine, multistep, none. "
            f"Got '{scheduler_type}'."
        )

    max_steps = int(cfg.trainer.max_optimizer_steps)
    if max_steps < 1:
        raise ValueError("trainer.max_optimizer_steps must be >= 1")

    t_max_steps = cfg.trainer.get("scheduler_t_max_steps", None)
    if t_max_steps is None:
        t_max = max_steps
    else:
        t_max = max(1, int(t_max_steps))
    eta_min = float(cfg.trainer.get("scheduler_eta_min", 0.0))

    warmup_steps_cfg = cfg.trainer.get("scheduler_warmup_steps", None)
    if warmup_steps_cfg is None:
        warmup_ratio = float(cfg.trainer.get("scheduler_warmup_ratio", 0.0))
        warmup_steps = int(round(max_steps * max(0.0, warmup_ratio)))
    else:
        warmup_steps = max(0, int(warmup_steps_cfg))

    warmup_steps = max(0, min(warmup_steps, max(0, max_steps - 1)))
    if scheduler_type == "cosine":
        base_scheduler: Any = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max if warmup_steps <= 0 else t_max - warmup_steps),
            eta_min=eta_min,
        )
    else:
        raw_milestones = cfg.trainer.get("scheduler_milestones_steps", [])
        milestone_steps = sorted(
            {max(1, int(step)) for step in raw_milestones if int(step) > 0}
        )
        if warmup_steps > 0:
            milestone_steps = [
                max(1, step - warmup_steps)
                for step in milestone_steps
                if step > warmup_steps
            ]
        gamma = float(cfg.trainer.get("scheduler_gamma", 0.1))
        base_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestone_steps,
            gamma=gamma,
        )

    if warmup_steps <= 0:
        return base_scheduler

    start_factor = float(cfg.trainer.get("scheduler_warmup_start_factor", 1e-3))
    start_factor = min(max(start_factor, 1e-8), 1.0)

    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, base_scheduler],
        milestones=[warmup_steps],
    )


def _validation_selection_mode_from_config(cfg: DictConfig) -> str:
    mode = str(cfg.trainer.get("validation_selection_mode", "auto")).lower()
    if mode not in {"auto", "min", "max"}:
        raise ValueError(
            "trainer.validation_selection_mode must be one of: auto, min, max."
        )
    return mode


def _resolve_validation_selection_metric(
    val_metrics: Mapping[str, float],
    requested_metric: str,
) -> tuple[str, float]:
    if requested_metric in val_metrics:
        return requested_metric, float(val_metrics[requested_metric])
    if "mean_id_e-edgef1" in val_metrics:
        return "mean_id_e-edgef1", float(val_metrics["mean_id_e-edgef1"])
    if "mean_e-edgef1" in val_metrics:
        return "mean_e-edgef1", float(val_metrics["mean_e-edgef1"])
    return requested_metric, 0.0


def _should_maximize_selection_metric(metric_name: str, mode: str) -> bool:
    normalized_mode = str(mode).lower()
    if normalized_mode == "max":
        return True
    if normalized_mode == "min":
        return False

    metric_name_l = str(metric_name).lower()
    minimize_tokens = (
        "sid",
        "shd",
        "nll",
        "loss",
        "error",
        "rmse",
        "mae",
        "mse",
    )
    maximize_tokens = ("f1", "auc", "acc", "precision", "recall", "pct", "score")
    if any(token in metric_name_l for token in minimize_tokens):
        return False
    if any(token in metric_name_l for token in maximize_tokens):
        return True
    return True


def _is_metric_improved(
    current_metric: float,
    best_metric: float,
    *,
    maximize: bool,
) -> bool:
    return current_metric > best_metric if maximize else current_metric < best_metric


def _adamw_supports_fused() -> bool:
    return "fused" in inspect.signature(optim.AdamW).parameters


def _maybe_clip_grad_norm(
    parameters: Iterable[torch.nn.Parameter], max_norm: float
) -> None:
    if max_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def _compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return float(total**0.5)


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    return total, trainable


def _current_learning_rate(optimizer: optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def _reduce_loss_for_logging(loss: torch.Tensor, world_size: int) -> float:
    loss_value = loss.detach()
    if dist.is_available() and dist.is_initialized():
        loss_value = loss_value.clone()
        dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
        if world_size > 0:
            loss_value = loss_value / float(world_size)
    return float(loss_value.item())


def _resolve_training_plan(
    cfg: DictConfig,
    *,
    max_per_device_batch_size: int,
    world_size: int,
) -> _TrainingPlan:
    if max_per_device_batch_size < 1:
        raise ValueError("data.batch_size_train must be >= 1")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")

    target_global_tasks_per_step = int(cfg.trainer.target_global_tasks_per_step)
    if target_global_tasks_per_step < 1:
        raise ValueError("trainer.target_global_tasks_per_step must be >= 1")
    if target_global_tasks_per_step < world_size:
        raise ValueError(
            "trainer.target_global_tasks_per_step must be >= world_size to keep at least one task per device."
        )

    max_train_batch_size = min(
        max_per_device_batch_size,
        target_global_tasks_per_step // world_size,
    )
    for train_batch_size in range(max_train_batch_size, 0, -1):
        microbatch_global = train_batch_size * world_size
        if target_global_tasks_per_step % microbatch_global != 0:
            continue

        accumulate_grad_batches = target_global_tasks_per_step // microbatch_global
        max_optimizer_steps = int(cfg.trainer.max_optimizer_steps)
        if max_optimizer_steps < 1:
            raise ValueError("trainer.max_optimizer_steps must be >= 1")
        log_every_n_steps = int(cfg.trainer.get("log_every_n_steps", 100))
        val_check_interval_steps = int(
            cfg.trainer.get("val_check_interval_steps", 1000)
        )
        checkpoint_every_n_steps = int(
            cfg.trainer.get(
                "checkpoint_every_n_steps",
                val_check_interval_steps,
            )
        )
        return _TrainingPlan(
            train_batch_size=train_batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            global_tasks_per_step=target_global_tasks_per_step,
            max_optimizer_steps=max_optimizer_steps,
            log_every_n_steps=log_every_n_steps,
            val_check_interval_steps=val_check_interval_steps,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
        )

    raise ValueError(
        "trainer.target_global_tasks_per_step must be realizable as "
        "data.batch_size_train * accumulate_grad_batches * world_size with integer accumulation. "
        f"Got target_global_tasks_per_step={target_global_tasks_per_step}, "
        f"data.batch_size_train={max_per_device_batch_size}, world_size={world_size}."
    )


def _next_step_threshold(current_step: int, interval: int) -> int | None:
    if interval <= 0:
        return None
    completed_intervals = current_step // interval
    return int((completed_intervals + 1) * interval)


def _advance_step_threshold(
    next_threshold: int | None,
    interval: int,
    current_step: int,
) -> int | None:
    if next_threshold is None or interval <= 0:
        return None
    while next_threshold <= current_step:
        next_threshold += interval
    return next_threshold


def _maybe_parse_betas(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError("trainer.optimizer_betas must be a sequence of two floats.")


def _maybe_parse_eps(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    pass
