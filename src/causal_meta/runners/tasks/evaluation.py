from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Mapping, cast

import torch
import torch.nn as nn
from omegaconf import DictConfig

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.datasets.generators.configs import (AnyFamilyConfig,
                                                     FamilyConfig,
                                                     RealWorldFamilyConfig)
from causal_meta.datasets.torch_datasets import MetaFixedDataset
from causal_meta.datasets.utils.normalization import normalize_scm_data
from causal_meta.runners.logger.base import BaseLogger
from causal_meta.runners.metrics.graph import Metrics
from causal_meta.runners.metrics.scm import SCMMetrics
from causal_meta.runners.tasks.utils import (infer_device,
                                             resolve_inference_root,
                                             sampling_mode, shard_indices,
                                             unwrap_model)
from causal_meta.runners.utils.artifacts import (
    NpEncoder, atomic_torch_save, cache_settings, cache_suffix, get_model_name,
    prepare_graph_samples_for_cache, resolve_output_dir)
from causal_meta.runners.utils.distributed import DistributedContext
from causal_meta.runners.utils.explicit_profiles import (
    apply_explicit_profile, infer_explicit_profile)

log = logging.getLogger(__name__)


def _is_avici_model(model_name: str) -> bool:
    return model_name.strip().lower() == "avici"


def _prepare_amortized_model_input(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prepare a model input tensor and optional node mask."""
    input_data = batch["data"].to(device)
    intervention_mask = batch.get("intervention_mask")
    if intervention_mask is not None:
        input_data = torch.stack([input_data, intervention_mask.to(device)], dim=-1)

    node_mask = batch.get("node_mask")
    if node_mask is not None:
        node_mask = node_mask.to(device)

    return input_data, node_mask


def _extract_graph_type(family_cfg: AnyFamilyConfig) -> str:
    """Infer the graph type string from a FamilyConfig's graph_cfg."""
    if isinstance(family_cfg, RealWorldFamilyConfig):
        return "real_world"

    graph_cfg = family_cfg.graph_cfg
    type_attr = getattr(graph_cfg, "type", None)
    if type_attr is not None:
        return str(type_attr)
    # Fall back to class name heuristic
    cls_name = type(graph_cfg).__name__.lower()
    for tag in ("er", "scalefree", "sf", "sbm", "mixture"):
        if tag in cls_name:
            # Normalize "scalefree" to "sf"
            return "sf" if tag == "scalefree" else tag
    return "unknown"


def _extract_mech_type(family_cfg: AnyFamilyConfig) -> str:
    """Infer the mechanism type string from a FamilyConfig's mech_cfg."""
    if isinstance(family_cfg, RealWorldFamilyConfig):
        return "real_world"

    mech_cfg = family_cfg.mech_cfg
    type_attr = getattr(mech_cfg, "type", None)
    if type_attr is not None:
        return str(type_attr)
    cls_name = type(mech_cfg).__name__.lower()
    for tag in (
        "linear",
        "mlp",
        "gp",
        "periodic",
        "square",
        "logistic",
        "pnl",
        "mixture",
    ):
        if tag in cls_name:
            return tag
    return "unknown"


def _extract_sparsity_param(family_cfg: AnyFamilyConfig) -> float | None:
    """Extract the sparsity-related parameter from a graph config, if available."""
    if isinstance(family_cfg, RealWorldFamilyConfig):
        return None

    graph_cfg = family_cfg.graph_cfg
    # ER: sparsity or edge_prob
    sparsity = getattr(graph_cfg, "sparsity", None)
    if sparsity is not None:
        return float(sparsity)
    edge_prob = getattr(graph_cfg, "edge_prob", None)
    if edge_prob is not None:
        return float(edge_prob)
    # SF: attachment parameter m
    m = getattr(graph_cfg, "m", None)
    if m is not None:
        return float(m)
    return None


def _build_family_metadata(
    test_family_cfgs: dict[str, AnyFamilyConfig],
    *,
    default_samples_per_task: int | None = None,
    default_inference_n_samples: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a dataset_key -> metadata map for all evaluation families."""
    result: dict[str, dict[str, Any]] = {}
    for fcfg in test_family_cfgs.values():
        entry: dict[str, Any] = {
            "family_name": fcfg.name,
            "n_nodes": fcfg.n_nodes,
            "samples_per_task": (
                int(fcfg.samples_per_task)
                if fcfg.samples_per_task is not None
                else default_samples_per_task
            ),
            "graph_type": _extract_graph_type(fcfg),
            "mech_type": _extract_mech_type(fcfg),
        }
        if isinstance(fcfg, RealWorldFamilyConfig):
            entry["loader"] = fcfg.loader
        family_ins = getattr(fcfg, "inference_n_samples", None)
        entry["inference_n_samples"] = (
            int(family_ins) if family_ins is not None else default_inference_n_samples
        )
        sp = _extract_sparsity_param(fcfg)
        if sp is not None:
            entry["sparsity_param"] = sp
        result[fcfg.name] = entry
    return result


def _resolve_dataset_num_nodes(
    dataset: MetaFixedDataset, family: Any | None
) -> int | None:
    """Infer the node count for a fixed evaluation dataset."""
    if family is not None and hasattr(family, "n_nodes"):
        return int(getattr(family, "n_nodes"))
    if len(dataset) < 1:
        return None
    sample = dataset[0]
    data = sample.get("data")
    if isinstance(data, torch.Tensor) and data.ndim >= 2:
        return int(data.shape[-1])
    return None


def _maybe_set_model_num_nodes(model: nn.Module, num_nodes: int | None) -> bool:
    """Update explicit models that support dynamic node-count overrides."""
    if num_nodes is None:
        return False
    setter = getattr(model, "set_num_nodes", None)
    if not callable(setter):
        return False
    setter(int(num_nodes))
    return True


def _evaluate_explicit_model(
    *,
    model: nn.Module,
    loader: Any,
    dataset_name: str,
    family: Any | None,
    device: torch.device,
    rank: int,
    world_size: int,
    metrics_handler: Metrics,
    scm_metrics_handler: SCMMetrics,
    n_samples: int,
    inference_root: Path,
    model_name: str,
    cache_dir: Any,
    cache_dtype: str,
    cache_n_samples: int | None,
    suffix: str,
    is_avici: bool,
) -> list[float]:
    """Evaluate one explicit model dataset by always sampling live."""
    dataset = cast(MetaFixedDataset, loader.dataset)
    profile = infer_explicit_profile(dataset_name, family)
    profile_applied = apply_explicit_profile(model, profile)
    dataset_num_nodes = _resolve_dataset_num_nodes(dataset, family)
    num_nodes_applied = _maybe_set_model_num_nodes(model, dataset_num_nodes)
    sampling_mode_ = sampling_mode(model)

    if rank == 0 and profile_applied and profile is not None:
        log.info(
            "Applying explicit profile '%s' for dataset '%s'.", profile, dataset_name
        )
    if rank == 0 and num_nodes_applied and dataset_num_nodes is not None:
        log.info(
            "Applying explicit num_nodes=%d for dataset '%s'.",
            dataset_num_nodes,
            dataset_name,
        )

    out_dir = (
        (inference_root / model_name / dataset_name)
        if cache_dir
        else (inference_root / dataset_name)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    inference_times: list[float] = []
    sampling_context_logged = False

    with torch.no_grad():
        for idx in shard_indices(len(dataset), rank=rank, world_size=world_size):
            item = dataset[idx]
            seed = int(item["seed"])
            input_data_raw = item["data"]
            adjacency_matrix_true = item["adjacency"]
            input_data = normalize_scm_data(input_data_raw).to(device).unsqueeze(0)
            adjacency_matrix = adjacency_matrix_true.to(device).unsqueeze(0)

            if rank == 0 and not sampling_context_logged:
                log.info(
                    "Evaluation sampling context: dataset=%s, device=%s, n_samples=%d, mode=%s",
                    dataset_name,
                    device,
                    n_samples,
                    sampling_mode_,
                )
                sampling_context_logged = True

            t0 = time.perf_counter()
            edge_probs = None
            thresholded_graph = None
            if is_avici:
                edge_probs = model.predict_probabilities(input_data)
                thresholded_graph = (edge_probs > 0.5).to(dtype=edge_probs.dtype)
                thresholded_graph = thresholded_graph.unsqueeze(0)
            samples = model.sample(input_data, num_samples=n_samples).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_times.append(time.perf_counter() - t0)
            samples_for_metrics = samples.permute(1, 0, 2, 3)

            cached_samples = prepare_graph_samples_for_cache(
                samples.detach(),
                dtype=cache_dtype,
                max_samples=cache_n_samples,
            )
            atomic_torch_save(
                {
                    "seed": seed,
                    "idx": int(idx),
                    "graph_samples": cached_samples.cpu(),
                    "true_adj": (adjacency_matrix_true.detach() > 0.5)
                    .to(dtype=torch.uint8)
                    .cpu(),
                    "cache_dtype": str(cache_dtype),
                    "cache_n_samples": (
                        int(cache_n_samples) if cache_n_samples is not None else None
                    ),
                    "requested_n_samples": int(n_samples),
                    "num_samples_stored": int(cached_samples.shape[1]),
                },
                out_dir / f"seed_{seed}{suffix}",
            )

            metrics_handler.update(
                adjacency_matrix,
                samples_for_metrics,
                thresholded_graph=thresholded_graph,
                edge_probs=edge_probs,
            )
            if bool(getattr(model, "estimates_scm", False)):
                scm_metrics_handler.update(
                    obs_data=input_data.squeeze(0),
                    graph_samples=samples_for_metrics.squeeze(1),
                    family=family,
                    seeds=[seed],
                )

    return inference_times


def _evaluate_amortized_model(
    *,
    model: nn.Module,
    loader: Any,
    dataset_name: str,
    family: Any | None,
    device: torch.device,
    metrics_handler: Metrics,
    scm_metrics_handler: SCMMetrics,
    n_samples: int,
    inference_root: Path,
    model_name: str,
    cache_dir: Any,
    cache_dtype: str,
    cache_n_samples: int | None,
    suffix: str,
    is_avici: bool,
) -> list[float]:
    """Evaluate one amortized model dataset using batched loader inputs."""
    out_dir = (
        (inference_root / model_name / dataset_name)
        if cache_dir
        else (inference_root / dataset_name)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    inference_times: list[float] = []
    batch_idx = 0

    with torch.no_grad():
        for batch in loader:
            input_data, node_mask = _prepare_amortized_model_input(batch, device)
            adjacency_matrix = batch["adjacency"].to(device)
            seeds = batch.get("seed")
            if seeds is not None and hasattr(seeds, "tolist"):
                seeds = seeds.tolist()

            t0 = time.perf_counter()
            edge_probs = None
            thresholded_graph = None
            if is_avici:
                edge_probs = model.predict_probabilities(input_data, mask=node_mask)
                thresholded_graph = (edge_probs > 0.5).to(dtype=edge_probs.dtype)
                thresholded_graph = thresholded_graph.unsqueeze(0)
            samples = model.sample(
                input_data,
                num_samples=n_samples,
                mask=node_mask,
            ).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_times.append(time.perf_counter() - t0)
            samples_for_metrics = samples.permute(1, 0, 2, 3)

            metrics_handler.update(
                adjacency_matrix,
                samples_for_metrics,
                thresholded_graph=thresholded_graph,
                edge_probs=edge_probs,
            )

            # Save inference artifacts per task (batch_size is always 1 at
            # test time, so each batch element corresponds to a single task).
            batch_seeds = seeds if isinstance(seeds, list) else None
            for b in range(int(input_data.shape[0])):
                seed = int(batch_seeds[b]) if batch_seeds is not None else batch_idx
                adjacency_true = adjacency_matrix[b]

                cached_samples = prepare_graph_samples_for_cache(
                    samples[b : b + 1].detach(),
                    dtype=cache_dtype,
                    max_samples=cache_n_samples,
                )
                atomic_torch_save(
                    {
                        "seed": seed,
                        "idx": batch_idx,
                        "graph_samples": cached_samples.cpu(),
                        "true_adj": (adjacency_true.detach() > 0.5)
                        .to(dtype=torch.uint8)
                        .cpu(),
                        "cache_dtype": str(cache_dtype),
                        "cache_n_samples": (
                            int(cache_n_samples)
                            if cache_n_samples is not None
                            else None
                        ),
                        "requested_n_samples": int(n_samples),
                        "num_samples_stored": int(cached_samples.shape[1]),
                    },
                    out_dir / f"seed_{seed}{suffix}",
                )
                batch_idx += 1

            if bool(getattr(model, "estimates_scm", False)):
                if batch_seeds is None:
                    log.warning(
                        "Skipping SCM metrics because batch seeds are unavailable."
                    )
                else:
                    for b in range(int(input_data.shape[0])):
                        scm_metrics_handler.update(
                            obs_data=input_data[b],
                            graph_samples=samples_for_metrics[:, b],
                            family=family,
                            seeds=[int(batch_seeds[b])],
                        )

    return inference_times


def run(
    cfg: DictConfig,
    model: nn.Module,
    data_module: CausalMetaModule,
    *,
    logger: BaseLogger | None = None,
    output_dir: Path | None = None,
) -> None:
    # DDP Setup
    dist_ctx = DistributedContext.current()
    rank = dist_ctx.rank

    if rank == 0:
        log.info(f"Starting evaluation: {cfg.name}")

    # Device (explicit models may have no parameters)
    device = infer_device(model, dist_ctx)

    # Data
    test_loaders = data_module.test_dataloader()

    # Pre-compute family metadata; distances are pre-computed by data_module.setup()
    dm_config = getattr(data_module, "config", None)
    test_family_cfgs = getattr(dm_config, "test_families", None) or {}
    default_samples_per_task = None
    if (
        dm_config is not None
        and getattr(dm_config, "samples_per_task", None) is not None
    ):
        default_samples_per_task = int(dm_config.samples_per_task)

    # Global default for posterior graph samples to draw.
    # Priority: data.inference_n_samples > inference.n_samples > 100
    default_inference_n_samples: int
    if (
        dm_config is not None
        and getattr(dm_config, "inference_n_samples", None) is not None
    ):
        default_inference_n_samples = int(dm_config.inference_n_samples)
    else:
        default_inference_n_samples = int(cfg.inference.get("n_samples", 100))

    family_metadata = _build_family_metadata(
        test_family_cfgs,
        default_samples_per_task=default_samples_per_task,
        default_inference_n_samples=default_inference_n_samples,
    )
    family_distances = getattr(data_module, "family_distances", {}) or {}

    model.eval()

    # Unwrap if DDP
    model_unwrapped = unwrap_model(model)
    model_unwrapped_raw = model_unwrapped

    all_summary_metrics = {}
    all_raw_metrics = {}

    auc_num_shuffles = int(cfg.inference.get("auc_num_shuffles", 1000))
    auc_balance_classes = bool(cfg.inference.get("auc_balance_classes", True))
    auc_seed = int(cfg.inference.get("auc_seed", 0))

    # Initialize Metrics Handlers (full metric set including opt-in diagnostics)
    metrics = [
        "e-shd",
        "e-edgef1",
        "e-sid",
        "ne-shd",
        "ne-sid",
        "graph_nll",
        "graph_nll_per_edge",
        "edge_entropy",
        "ancestor_f1",
        "auc",
        "fp_count",
        "fn_count",
        "reversed_count",
        "correct_count",
        "sparsity_ratio",
        "skeleton_f1",
        "orientation_accuracy",
        "valid_dag_pct",
        "threshold_valid_dag_pct",
        "ece",
    ]
    model_name = get_model_name(cfg, model_unwrapped_raw)
    is_avici = _is_avici_model(model_name)
    if model_name.lower() == "avici":
        metrics.extend(
            [
                "sampled-e-shd",
                "sampled-e-edgef1",
                "sampled-e-sid",
                "sampled-ne-shd",
                "sampled-ne-sid",
                "sampled-graph_nll",
                "sampled-graph_nll_per_edge",
                "sampled-edge_entropy",
                "sampled-ece",
                "sampled-ancestor_f1",
                "sampled-sparsity_ratio",
                "sampled-skeleton_f1",
                "sampled-orientation_accuracy",
                "sampled-valid_dag_pct",
            ]
        )
        metrics = [m for m in metrics if m != "threshold_valid_dag_pct"]
    metrics_handler = Metrics(
        metrics=metrics,
        auc_num_shuffles=auc_num_shuffles,
        auc_balance_classes=auc_balance_classes,
        auc_seed=auc_seed,
    )
    scm_metrics_handler = SCMMetrics(metrics=["inil", "inil_per_node"])

    # Determine output directory and artifact settings.
    output_dir = resolve_output_dir(cfg, output_dir)
    cache_dir = cfg.inference.get("cache_dir", None)
    inference_root = resolve_inference_root(cfg, output_dir)
    cache_compress, cache_dtype, cache_n_samples_cfg = cache_settings(cfg)
    suffix = cache_suffix(compress=cache_compress)

    for name, loader in test_loaders.items():
        if rank == 0:
            log.info(f"Evaluating on {name}...")

        test_families = getattr(data_module, "test_families", {}) or {}
        family = test_families.get(name)

        # Resolve per-family inference n_samples: prefer per-family config,
        # fall back to the global default (data.inference_n_samples or
        # inference.n_samples).
        family_cfg = test_family_cfgs.get(name)
        family_n_samples = getattr(family_cfg, "inference_n_samples", None)
        n_samples = (
            int(family_n_samples)
            if family_n_samples is not None
            else default_inference_n_samples
        )

        # Adjust cache_n_samples for this family's n_samples.
        cache_n_samples = cache_n_samples_cfg
        if cache_n_samples is not None and cache_n_samples < n_samples:
            if rank == 0:
                log.warning(
                    "cache_n_samples (%d) is smaller than n_samples (%d) for dataset '%s'; "
                    "overriding cache_n_samples to %d so written artifacts preserve all evaluated samples.",
                    cache_n_samples,
                    n_samples,
                    name,
                    n_samples,
                )
            cache_n_samples = n_samples

        metrics_handler.reset()
        scm_metrics_handler.reset()
        if model_unwrapped.needs_pretraining:
            inference_times = _evaluate_amortized_model(
                model=model_unwrapped,
                loader=loader,
                dataset_name=name,
                family=family,
                device=device,
                metrics_handler=metrics_handler,
                scm_metrics_handler=scm_metrics_handler,
                n_samples=n_samples,
                inference_root=inference_root,
                model_name=model_name,
                cache_dir=cache_dir,
                cache_dtype=cache_dtype,
                cache_n_samples=cache_n_samples,
                suffix=suffix,
                is_avici=is_avici,
            )
        else:
            inference_times = _evaluate_explicit_model(
                model=model_unwrapped,
                loader=loader,
                dataset_name=name,
                family=family,
                device=device,
                rank=rank,
                world_size=dist_ctx.world_size,
                metrics_handler=metrics_handler,
                scm_metrics_handler=scm_metrics_handler,
                n_samples=n_samples,
                inference_root=inference_root,
                model_name=model_name,
                cache_dir=cache_dir,
                cache_dtype=cache_dtype,
                cache_n_samples=cache_n_samples,
                suffix=suffix,
                is_avici=is_avici,
            )

        # Compute Summary Stats (Mean + SEM) and Gather Raw Data
        summary = metrics_handler.compute(summary_stats=True)
        final_metrics = metrics_handler.gather_raw_results()

        scm_summary = scm_metrics_handler.compute(summary_stats=True)
        scm_raw = scm_metrics_handler.get_raw_results()

        if rank == 0:
            # Merge
            summary.update(scm_summary)
            for k, v in scm_raw.items():
                final_metrics[k] = v

            # Attach wall-clock inference times for live sampling. When no
            # timing data were recorded, the summary entries are omitted.
            if inference_times:
                final_metrics["inference_time_s"] = inference_times
                mean_t = sum(inference_times) / len(inference_times)
                summary["inference_time_s_mean"] = mean_t
                summary["inference_time_s_total"] = sum(inference_times)

            all_summary_metrics[name] = summary
            all_raw_metrics[name] = final_metrics

            log.info(f"Results for {name}: {summary}")

            if logger:
                # Log means to Logger
                logger.log_metrics({f"test/{name}/{k}": v for k, v in summary.items()})

    # Save
    if rank == 0:
        dm_config = getattr(data_module, "config", None)
        batch_size_test = (
            int(getattr(dm_config, "batch_size_test", 1))
            if dm_config is not None
            else 1
        )
        batch_size_test_interventional = (
            int(getattr(dm_config, "batch_size_test_interventional", 1))
            if dm_config is not None
            else 1
        )

        # Graph-metric raw values are appended per update call.  Batch size
        # is enforced to 1 for test/val loaders, so granularity is always
        # per-task.
        raw_granularity = "per_task"

        metadata = {
            "run_id": output_dir.name,
            "run_name": str(cfg.get("name", "")),
            "model_name": str(model_name),
            "output_dir": str(output_dir),
            "inference_root": str(inference_root.expanduser().resolve()),
            "inference_layout": "model_dataset" if cache_dir else "dataset",
            "inference_n_samples": int(default_inference_n_samples),
            "cache_n_samples": (
                int(cache_n_samples_cfg) if cache_n_samples_cfg is not None else None
            ),
            "batch_size_test": batch_size_test,
            "batch_size_test_interventional": batch_size_test_interventional,
            "raw_granularity": raw_granularity,
        }
        model_results_path = output_dir / "metrics.json"
        with open(model_results_path, "w") as f:
            json.dump(
                {
                    "metadata": metadata,
                    "family_metadata": family_metadata,
                    "distances": family_distances,
                    "summary": all_summary_metrics,
                    "raw": all_raw_metrics,
                },
                f,
                cls=NpEncoder,
                indent=4,
            )

        log.info(f"Metrics saved to {model_results_path}")
