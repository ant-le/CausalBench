# Causal Meta-Learning Benchmarks

A scalable, Hydra-configured framework for benchmarking Bayesian Causal Discovery and Meta-Learning algorithms.

## Features

- **Meta-Learning Focus:** Infinite streaming datasets (`MetaIterableDataset`) with rank-aware seeding for massive parallel training.
- **Evaluation:** Strict O.O.D. generalization tests with disjoint graph hashing and cached inference artifacts.
- **Scalability:** Distributed Data Parallel (DDP) support, preemption-safe checkpointing, and cluster-ready per-model Slurm scripts.
- **Metrics:** Comprehensive graph metrics (SHD, SID, F1, AUROC) and likelihood proxies.

## Quick Start

### Installation (uv)

```bash
uv sync --extra cluster --extra wandb --frozen --no-editable

# If DiBS should use NVIDIA GPUs on CUDA 12 clusters:
uv pip install --python .venv/bin/python --upgrade "jax[cuda12-local]"
```

Use `--no-editable` for robust cross-platform imports of `causal_meta`.

For full multimodel runs including BayesDAG, use:

```bash
scripts/bootstrap_uv.sh
```

Cluster scripts auto-detect BayesDAG Python from `.bootstrap_env.sh` or
`.venv-bayesdag/bin/python`.

### Running a Smoke Test

```bash
# Run the benchmark-shaped smoke config locally
uv run causal-meta --config-name dg_2pretrain_smoke model=avici
```

Smoke configs use online Weights & Biases logging. Ensure `wandb login` has been
run in your environment.

### Running on a Cluster (Slurm)

```bash
sbatch scripts/run_bcnp.sh
```

Run the main benchmark sweep with:

```bash
scripts/submit_all_models.sh main
```

Run the smoke sweep with the same benchmark layout:

```bash
scripts/submit_all_models.sh smoke
```

Run the ablation suite (AviCi + BCNP across all ablation data configs):

```bash
scripts/submit_ablation_suite.sh
```

## Documentation

- [Runbook](docs/RUNBOOK.md): Detailed guide on running experiments and sweeps.
- [Design](docs/DESIGN.md): Architectural overview.
