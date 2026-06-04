"""Evaluate source AVICI pretrained checkpoints on thesis test data.

Three-phase pipeline to avoid dependency conflicts between the thesis
environment (PyTorch + numpy 2.x) and ``avici`` (JAX + tensorflow +
numpy 1.x-era deps):

**Phase 1 — generate-data** (main venv, uses ``causal_meta``):
    Generate deterministic test data from thesis SCM families and save as
    ``.npz`` files.

**Phase 2 — run-inference** (avici venv, only needs ``avici`` + ``numpy``):
    Load ``.npz`` data, run source AVICI pretrained models, save edge
    probability matrices as ``.npz`` files.

**Phase 3 — compute-metrics** (main venv, uses ``causal_meta``):
    Load edge probabilities, convert to Bernoulli samples, compute thesis
    metrics, write ``metrics.json``.

Usage::

    # Full pipeline: generate data, run inference (requires avici venv),
    # compute metrics
    PYTHONPATH=src uv run python -m causal_meta.analysis.paper_comparison.eval_source_avici \\
        generate-data --output-root experiments/thesis_runs

    # In avici venv:
    .venv-avici/bin/python -m causal_meta.analysis.paper_comparison.eval_source_avici \\
        run-inference --checkpoint scm-v0 \\
        --data-dir experiments/thesis_runs/avici_pretrained_scm-v0/data

    # Back in main venv:
    PYTHONPATH=src uv run python -m causal_meta.analysis.paper_comparison.eval_source_avici \\
        compute-metrics --checkpoint scm-v0

    # Or run all phases in one command (requires avici importable):
    PYTHONPATH=src uv run python -m causal_meta.analysis.paper_comparison.eval_source_avici \\
        run-all --checkpoints scm-v0

Setup for avici venv::

    uv venv .venv-avici --python 3.11
    uv pip install --python .venv-avici/bin/python avici numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[4]

CONFIG_PATH = (
    REPO_ROOT / "src" / "causal_meta" / "configs" / "dg_2pretrain_multimodel.yaml"
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "thesis_runs"

# Test seeds from the benchmark config.
SEEDS_TEST = list(range(2000, 2025))

# Default number of Bernoulli samples to draw from edge probability matrix.
N_BERNOULLI_SAMPLES = 100

# Metrics matching the thesis evaluation pipeline.
METRICS_LIST = [
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

# Available pretrained checkpoints from the avici package.
# scm-v0 is the most general, trained on the broadest distribution.
AVAILABLE_CHECKPOINTS = ["scm-v0"]


# ── Family specification ───────────────────────────────────────────────────


@dataclass
class FamilySpec:
    """Lightweight specification for a test family, parsed from YAML."""

    name: str
    n_nodes: int
    samples_per_task: int
    graph_cfg: dict[str, Any]
    mech_cfg: dict[str, Any]
    noise_type: str = "gaussian"


def _parse_families_from_yaml(config_path: Path) -> dict[str, FamilySpec]:
    """Parse test family specs directly from the multimodel YAML config.

    Uses PyYAML directly (not Hydra) to avoid initialization overhead.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    default_samples_per_task = int(data_cfg.get("samples_per_task", 500))
    default_n_nodes = int(data_cfg.get("n_nodes", 20))

    test_families_raw = data_cfg.get("test_families", {})
    families: dict[str, FamilySpec] = {}

    for key, fam_cfg in test_families_raw.items():
        name = fam_cfg.get("name", key)
        n_nodes = int(fam_cfg.get("n_nodes", default_n_nodes))
        samples_per_task = int(
            fam_cfg.get("samples_per_task", default_samples_per_task)
        )
        graph_cfg = fam_cfg.get("graph_cfg", {})
        mech_cfg = fam_cfg.get("mech_cfg", {})
        noise_type = fam_cfg.get("noise_type", "gaussian")
        families[name] = FamilySpec(
            name=name,
            n_nodes=n_nodes,
            samples_per_task=samples_per_task,
            graph_cfg=graph_cfg,
            mech_cfg=mech_cfg,
            noise_type=noise_type,
        )

    return families


# ── Family metadata builder ──────────────────────────────────────────────


def _graph_type_from_cfg(graph_cfg: dict[str, Any]) -> str:
    return str(graph_cfg.get("type", "unknown"))


def _mech_type_from_cfg(mech_cfg: dict[str, Any]) -> str:
    return str(mech_cfg.get("type", "unknown"))


def _sparsity_from_cfg(graph_cfg: dict[str, Any]) -> float | None:
    for key in ("sparsity", "edge_prob", "m"):
        val = graph_cfg.get(key)
        if val is not None:
            return float(val)
    return None


def _build_family_metadata(
    families: dict[str, FamilySpec],
) -> dict[str, dict[str, Any]]:
    """Build family_metadata matching the thesis metrics.json format."""
    result: dict[str, dict[str, Any]] = {}
    for spec in families.values():
        entry: dict[str, Any] = {
            "family_name": spec.name,
            "n_nodes": spec.n_nodes,
            "samples_per_task": spec.samples_per_task,
            "graph_type": _graph_type_from_cfg(spec.graph_cfg),
            "mech_type": _mech_type_from_cfg(spec.mech_cfg),
        }
        sp = _sparsity_from_cfg(spec.graph_cfg)
        if sp is not None:
            entry["sparsity_param"] = sp
        result[spec.name] = entry
    return result


# ── JSON encoder ──────────────────────────────────────────────────────────


class _NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Generate Data
# ══════════════════════════════════════════════════════════════════════════


def _build_scm_family(spec: FamilySpec) -> Any:
    """Build an SCMFamily from a FamilySpec using the thesis data pipeline."""
    from causal_meta.datasets.generators.factory import (
        load_graph_config,
        load_mechanism_config,
    )
    from causal_meta.datasets.scm import SCMFamily

    graph_generator = load_graph_config(spec.graph_cfg).instantiate()
    mechanism_factory = load_mechanism_config(spec.mech_cfg).instantiate()

    return SCMFamily(
        name=spec.name,
        n_nodes=spec.n_nodes,
        graph_generator=graph_generator,
        mechanism_factory=mechanism_factory,
        noise_type=spec.noise_type,
    )


def _generate_task_data(
    family: Any,
    seed: int,
    samples_per_task: int,
) -> tuple[Any, Any]:
    """Generate one task's data deterministically, matching MetaFixedDataset.

    Returns:
        (data, adjacency) as torch Tensors — data is (samples_per_task, n_nodes)
        and adjacency is (n_nodes, n_nodes), both float32.
    """
    import torch

    instance = family.sample_task(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        data = instance.sample(samples_per_task)

    adjacency = instance.adjacency_matrix.to(dtype=torch.float32)
    return data, adjacency


def phase_generate_data(
    config_path: Path,
    output_root: Path,
    checkpoints: list[str],
    seeds: list[int],
    *,
    max_families: int | None = None,
    max_seeds: int | None = None,
) -> None:
    """Phase 1: Generate test data and save as .npz files.

    Creates one directory per checkpoint under output_root, each containing
    a ``data/`` subdirectory with ``{family_name}/{seed}.npz`` files.
    Each ``.npz`` contains ``data`` (raw observational) and ``adjacency``
    (ground truth) as numpy arrays.
    """
    families = _parse_families_from_yaml(config_path)
    log.info("Parsed %d test families from %s", len(families), config_path)

    family_names = sorted(families.keys())
    if max_families is not None:
        family_names = family_names[:max_families]

    effective_seeds = seeds
    if max_seeds is not None:
        effective_seeds = seeds[:max_seeds]

    # Save family metadata as JSON (needed by Phase 3).
    for checkpoint in checkpoints:
        ckpt_dir = output_root / f"avici_pretrained_{checkpoint}"
        data_dir = ckpt_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save family metadata for this checkpoint subset.
        selected_families = {n: families[n] for n in family_names}
        metadata = _build_family_metadata(selected_families)
        with open(ckpt_dir / "family_metadata.json", "w") as f:
            json.dump(metadata, f, cls=_NpEncoder, indent=2)

        # Save family specs as JSON for Phase 3.
        specs_dict = {
            n: {
                "name": s.name,
                "n_nodes": s.n_nodes,
                "samples_per_task": s.samples_per_task,
                "graph_cfg": s.graph_cfg,
                "mech_cfg": s.mech_cfg,
                "noise_type": s.noise_type,
            }
            for n, s in selected_families.items()
        }
        with open(ckpt_dir / "family_specs.json", "w") as f:
            json.dump(specs_dict, f, cls=_NpEncoder, indent=2)

        # Save effective seeds list.
        with open(ckpt_dir / "seeds.json", "w") as f:
            json.dump(effective_seeds, f)

        total = len(family_names)
        for fam_idx, fam_name in enumerate(family_names, 1):
            spec = families[fam_name]
            log.info(
                "[%d/%d] Generating data for: %s (d=%d, n=%d, seeds=%d)",
                fam_idx,
                total,
                fam_name,
                spec.n_nodes,
                spec.samples_per_task,
                len(effective_seeds),
            )

            scm_family = _build_scm_family(spec)
            fam_dir = data_dir / fam_name
            fam_dir.mkdir(parents=True, exist_ok=True)

            for seed in effective_seeds:
                data, adjacency = _generate_task_data(
                    scm_family, seed, spec.samples_per_task
                )
                npz_path = fam_dir / f"{seed}.npz"
                np.savez_compressed(
                    npz_path,
                    data=data.numpy(),
                    adjacency=adjacency.numpy(),
                )

        log.info(
            "Phase 1 complete for '%s': %d families × %d seeds -> %s",
            checkpoint,
            len(family_names),
            len(effective_seeds),
            data_dir,
        )


# ══════════════════════════════════════════════════════════════════════════
# Phase 2: Run AVICI Inference
# ══════════════════════════════════════════════════════════════════════════


def _patch_jax_sharding() -> None:
    """Patch jax.sharding to add PositionalSharding if missing.

    avici 1.0.7 imports ``from jax.sharding import PositionalSharding``,
    which was removed in JAX 0.7+.  Since we only run single-device CPU
    inference (``shard_if_possible=False``), the class is never actually
    used — we just need the import to succeed.
    """
    try:
        import jax.sharding as sharding_mod
    except ImportError:
        return

    if not hasattr(sharding_mod, "PositionalSharding"):

        class _PositionalShardingStub:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "PositionalSharding unavailable in this JAX version."
                )

        sharding_mod.PositionalSharding = _PositionalShardingStub  # type: ignore[attr-defined]
        log.debug("Patched jax.sharding.PositionalSharding with stub.")


def phase_run_inference(
    checkpoint: str,
    data_dir: Path,
    output_dir: Path | None = None,
) -> None:
    """Phase 2: Run AVICI inference on pre-generated .npz data.

    This phase is designed to run in an isolated avici venv.  It only needs
    ``avici``, ``numpy``, ``jax``, and their transitive dependencies.

    Reads ``{data_dir}/{family_name}/{seed}.npz`` files and writes
    ``{output_dir}/{family_name}/{seed}.npz`` with ``edge_probs`` arrays.
    """
    _patch_jax_sharding()

    import avici

    if output_dir is None:
        output_dir = data_dir.parent / "edge_probs"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading AVICI checkpoint: %s", checkpoint)
    model = avici.load_pretrained(download=checkpoint)
    log.info("Checkpoint loaded. Data dir: %s", data_dir)

    family_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    total = len(family_dirs)

    for fam_idx, fam_dir in enumerate(family_dirs, 1):
        fam_name = fam_dir.name
        npz_files = sorted(fam_dir.glob("*.npz"))
        log.info(
            "[%d/%d] Running inference: %s (%d tasks)",
            fam_idx,
            total,
            fam_name,
            len(npz_files),
        )

        out_fam_dir = output_dir / fam_name
        out_fam_dir.mkdir(parents=True, exist_ok=True)

        for npz_path in npz_files:
            loaded = np.load(npz_path)
            data = loaded["data"]  # (n, d)

            # Run inference — pass raw data, avici handles standardization.
            g_prob = model(x=data.astype(np.float64), shard_if_possible=False)
            g_prob = np.asarray(g_prob, dtype=np.float64)

            out_path = out_fam_dir / npz_path.name
            np.savez_compressed(out_path, edge_probs=g_prob)

        log.info("  -> Saved edge probs to %s", out_fam_dir)

    log.info("Phase 2 complete for checkpoint '%s'.", checkpoint)


# ══════════════════════════════════════════════════════════════════════════
# Phase 3: Compute Metrics
# ══════════════════════════════════════════════════════════════════════════


def _edge_probs_to_samples(
    g_prob: np.ndarray,
    n_samples: int,
    seed: int,
) -> Any:
    """Convert edge probability matrix to binary Bernoulli samples.

    Returns:
        (n_samples, n_nodes, n_nodes) binary float32 torch Tensor.
    """
    import torch

    rng = np.random.default_rng(seed)
    prob_expanded = np.broadcast_to(g_prob[np.newaxis], (n_samples,) + g_prob.shape)
    samples = rng.random(prob_expanded.shape) < prob_expanded
    return torch.from_numpy(samples.astype(np.float32))


def _build_metrics_handler() -> Any:
    """Create a Metrics handler matching the thesis evaluation config."""
    from causal_meta.runners.metrics.graph import Metrics

    return Metrics(
        metrics=METRICS_LIST,
        auc_num_shuffles=1000,
        auc_balance_classes=True,
        auc_seed=0,
    )


def phase_compute_metrics(
    checkpoint: str,
    output_root: Path,
    *,
    n_bernoulli_samples: int = N_BERNOULLI_SAMPLES,
) -> Path:
    """Phase 3: Compute thesis metrics from edge probabilities.

    Reads edge probs from Phase 2, generates Bernoulli samples, and runs
    the thesis ``Metrics`` class.  Writes ``metrics.json`` in the same
    format as the thesis evaluation pipeline.
    """
    import torch

    ckpt_dir = output_root / f"avici_pretrained_{checkpoint}"
    data_dir = ckpt_dir / "data"
    probs_dir = ckpt_dir / "edge_probs"
    metrics_path = ckpt_dir / "metrics.json"

    if not probs_dir.exists():
        raise FileNotFoundError(
            f"Edge probs directory not found: {probs_dir}\n"
            "Run Phase 2 (run-inference) first."
        )

    # Load family metadata.
    with open(ckpt_dir / "family_metadata.json") as f:
        family_metadata = json.load(f)

    with open(ckpt_dir / "seeds.json") as f:
        seeds = json.load(f)

    family_dirs = sorted(
        [d for d in probs_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    all_summary: dict[str, dict[str, Any]] = {}
    all_raw: dict[str, dict[str, list[float]]] = {}
    total = len(family_dirs)

    t_start = time.perf_counter()

    for fam_idx, fam_dir in enumerate(family_dirs, 1):
        fam_name = fam_dir.name
        npz_files = sorted(fam_dir.glob("*.npz"))
        log.info(
            "[%d/%d] Computing metrics: %s (%d tasks)",
            fam_idx,
            total,
            fam_name,
            len(npz_files),
        )

        metrics_handler = _build_metrics_handler()

        for npz_path in npz_files:
            seed = int(npz_path.stem)

            # Load edge probs.
            probs_data = np.load(npz_path)
            g_prob = probs_data["edge_probs"]  # (d, d)

            # Load ground truth adjacency.
            data_npz = data_dir / fam_name / f"{seed}.npz"
            gt_data = np.load(data_npz)
            adjacency = torch.from_numpy(gt_data["adjacency"]).float()

            # Convert to Bernoulli samples.
            bernoulli_seed = seed * 31 + 7919
            samples = _edge_probs_to_samples(
                g_prob, n_bernoulli_samples, bernoulli_seed
            )

            # Feed into metrics: targets=(1, d, d), samples=(S, 1, d, d).
            targets = adjacency.unsqueeze(0)
            samples_4d = samples.unsqueeze(1)
            metrics_handler.update(targets, samples_4d)

        # Compute summary and raw.
        summary = metrics_handler.compute(summary_stats=True)
        raw = metrics_handler.gather_raw_results()

        all_summary[fam_name] = summary
        all_raw[fam_name] = raw

        auc = summary.get("auc_mean", float("nan"))
        shd = summary.get("e-shd_mean", float("nan"))
        f1 = summary.get("e-edgef1_mean", float("nan"))
        log.info("  -> AUROC=%.4f  E-SHD=%.2f  E-F1=%.4f", auc, shd, f1)

    elapsed = time.perf_counter() - t_start

    metadata = {
        "run_id": f"avici_pretrained_{checkpoint}",
        "run_name": f"avici_pretrained_{checkpoint}",
        "model_name": f"avici_pretrained_{checkpoint}",
        "output_dir": str(ckpt_dir),
        "inference_root": str(ckpt_dir),
        "inference_layout": "dataset",
        "inference_n_samples": n_bernoulli_samples,
        "cache_n_samples": n_bernoulli_samples,
        "batch_size_test": 1,
        "batch_size_test_interventional": 1,
        "raw_granularity": "per_task",
        "source_checkpoint": checkpoint,
        "source_package": "avici (pip install avici)",
        "note": (
            "Source AVICI pretrained model evaluated on thesis test data. "
            "Edge probabilities converted to binary samples via Bernoulli "
            f"sampling ({n_bernoulli_samples} draws). Raw unnormalized data "
            "passed to model (source AVICI handles standardization internally)."
        ),
        "wall_time_metrics_s": elapsed,
    }

    results = {
        "metadata": metadata,
        "family_metadata": family_metadata,
        "distances": {},
        "summary": all_summary,
        "raw": all_raw,
    }

    with open(metrics_path, "w") as f:
        json.dump(results, f, cls=_NpEncoder, indent=4)

    log.info(
        "Wrote %s (%d families, %.1fs total)",
        metrics_path,
        len(all_summary),
        elapsed,
    )
    return metrics_path


# ══════════════════════════════════════════════════════════════════════════
# Run-all: single command for all three phases
# ══════════════════════════════════════════════════════════════════════════


def phase_run_all(
    config_path: Path,
    output_root: Path,
    checkpoints: list[str],
    seeds: list[int],
    *,
    n_bernoulli_samples: int = N_BERNOULLI_SAMPLES,
    max_families: int | None = None,
    max_seeds: int | None = None,
) -> None:
    """Run all three phases sequentially (requires avici importable)."""
    log.info("=== Phase 1: Generating test data ===")
    phase_generate_data(
        config_path=config_path,
        output_root=output_root,
        checkpoints=checkpoints,
        seeds=seeds,
        max_families=max_families,
        max_seeds=max_seeds,
    )

    for checkpoint in checkpoints:
        ckpt_dir = output_root / f"avici_pretrained_{checkpoint}"
        data_dir = ckpt_dir / "data"
        log.info("=== Phase 2: Running inference for '%s' ===", checkpoint)
        phase_run_inference(
            checkpoint=checkpoint,
            data_dir=data_dir,
        )
        log.info("=== Phase 3: Computing metrics for '%s' ===", checkpoint)
        phase_compute_metrics(
            checkpoint=checkpoint,
            output_root=output_root,
            n_bernoulli_samples=n_bernoulli_samples,
        )

    log.info("All phases complete.")


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate source AVICI pretrained checkpoints on thesis test data.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Phase to run.")

    # ── generate-data ──────────────────────────────────────────────────
    p_gen = subparsers.add_parser(
        "generate-data",
        help="Phase 1: Generate test data as .npz files.",
    )
    p_gen.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    p_gen.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_gen.add_argument(
        "--checkpoints",
        nargs="+",
        default=AVAILABLE_CHECKPOINTS,
        choices=AVAILABLE_CHECKPOINTS,
    )
    p_gen.add_argument("--max-families", type=int, default=None)
    p_gen.add_argument("--max-seeds", type=int, default=None)

    # ── run-inference ──────────────────────────────────────────────────
    p_inf = subparsers.add_parser(
        "run-inference",
        help="Phase 2: Run AVICI inference (in avici venv).",
    )
    p_inf.add_argument(
        "--checkpoint",
        required=True,
        choices=AVAILABLE_CHECKPOINTS,
    )
    p_inf.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with .npz data files from Phase 1.",
    )
    p_inf.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for edge probs (default: sibling 'edge_probs/' dir).",
    )

    # ── compute-metrics ────────────────────────────────────────────────
    p_met = subparsers.add_parser(
        "compute-metrics",
        help="Phase 3: Compute thesis metrics from edge probabilities.",
    )
    p_met.add_argument(
        "--checkpoint",
        required=True,
        choices=AVAILABLE_CHECKPOINTS,
    )
    p_met.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_met.add_argument("--n-samples", type=int, default=N_BERNOULLI_SAMPLES)

    # ── run-all ────────────────────────────────────────────────────────
    p_all = subparsers.add_parser(
        "run-all",
        help="Run all three phases (requires avici importable).",
    )
    p_all.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    p_all.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_all.add_argument(
        "--checkpoints",
        nargs="+",
        default=AVAILABLE_CHECKPOINTS,
        choices=AVAILABLE_CHECKPOINTS,
    )
    p_all.add_argument("--n-samples", type=int, default=N_BERNOULLI_SAMPLES)
    p_all.add_argument("--max-families", type=int, default=None)
    p_all.add_argument("--max-seeds", type=int, default=None)

    # ── Common flags on every subparser ──────────────────────────────
    for p in (p_gen, p_inf, p_met, p_all):
        p.add_argument("--verbose", action="store_true", help="Debug logging.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "generate-data":
        phase_generate_data(
            config_path=args.config_path,
            output_root=args.output_root,
            checkpoints=args.checkpoints,
            seeds=SEEDS_TEST,
            max_families=args.max_families,
            max_seeds=args.max_seeds,
        )

    elif args.command == "run-inference":
        phase_run_inference(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )

    elif args.command == "compute-metrics":
        phase_compute_metrics(
            checkpoint=args.checkpoint,
            output_root=args.output_root,
            n_bernoulli_samples=args.n_samples,
        )

    elif args.command == "run-all":
        phase_run_all(
            config_path=args.config_path,
            output_root=args.output_root,
            checkpoints=args.checkpoints,
            seeds=SEEDS_TEST,
            n_bernoulli_samples=args.n_samples,
            max_families=args.max_families,
            max_seeds=args.max_seeds,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
