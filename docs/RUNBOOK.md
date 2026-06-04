# Runbook: Running Experiments

This guide uses `uv` + Hydra for configuration and per-model Slurm scripts in `scripts/`.

## 1. Environment Model

Use two environments:

- `.venv`: main stack (Avici, BCNP, DiBS)
- `.venv-bayesdag`: BayesDAG legacy stack

## 2. One-Time Setup

```bash
scripts/setup_cluster.sh
```

## 3. Daily Update After Pull

```bash
uv sync --extra cluster --extra wandb --frozen --no-editable
uv pip install --python .venv-bayesdag/bin/python -r requirements-bayesdag.txt
```

Note: do not use `uv pip sync` for `.venv-bayesdag`; it removes packages not
listed in `requirements-bayesdag.txt` (including `causica`, installed from
Project-BayesDAG source by `bootstrap_uv.sh`).

GPU-only default notes:

- `bootstrap_uv.sh` installs CUDA JAX (`jax[cuda12-local]`) by default.

## 4. Local Runs

```bash
uv run causal-meta --config-name default
uv run causal-meta --config-name dg_2pretrain_smoke model=avici
```

## 5. Cluster Runs (No Submitit)

Each model has a dedicated launcher script with hardcoded GPU specs:

- `scripts/run_avici.sh`: 4x A100
- `scripts/run_bcnp.sh`: 4x A100
- `scripts/run_dibs.sh`: 1x A100
- `scripts/run_bayesdag.sh`: 1x A100

Submit one model:

```bash
sbatch scripts/run_bcnp.sh
```

Submit all four models:

```bash
scripts/submit_all_models.sh main
```

Submit all four smoke jobs:

```bash
scripts/submit_all_models.sh smoke
```

`scripts/run_all_models.sh` remains available for sequential execution from an
existing allocation (it does not submit via `sbatch`).

Submit ablations (AviCi + BCNP across all ablation data configs):

```bash
scripts/submit_ablation_suite.sh
```

Customize ablation fanout with environment variables:

```bash
CAUSAL_META_ABLATION_MODELS=avici,bcnp \
CAUSAL_META_ABLATION_DATA_CONFIGS=ablation_linear_only,ablation_linear_mlp \
scripts/submit_ablation_suite.sh dg_2pretrain_multimodel my_ablation_prefix
```

Optional arguments for each script:

```bash
scripts/run_avici.sh <config_name> <run_name> [hydra_overrides...]
scripts/submit_all_models.sh [smoke|main] [run_prefix] [hydra_overrides...]
```

For BayesDAG, `CAUSAL_META_BAYESDAG_PYTHON` defaults to
`.venv-bayesdag/bin/python` when unset.

## 6. Comparability Checklist

Keep these fixed across models:

- `--config-name dg_2pretrain_multimodel`
- same lockfile (`uv.lock`) + frozen sync
- same seeds (`data.base_seed`, evaluation seeds)
- same inference settings (`inference.n_samples`, AUC controls)
- same partition/GPU type/time budget class

If W&B online is unavailable:

```bash
export WANDB_MODE=offline
```

## 7. Output Structure

Run outputs are under `experiments/runs/${name}/`:

- `checkpoints/`
- `metrics.json`
- `inference/`
- `main.log`
- `slurm_<jobid>.out`, `slurm_<jobid>.err`

## 8. Analysis from Run IDs

Generate tables/figures directly from selected run directories:

```bash
uv run python -m causal_meta.analysis.run_analysis experiments/runs \
  --run-id canary_20260317_120000_avici \
  --run-id canary_20260317_120000_bcnp

# Fail fast on missing/invalid analysis prerequisites.
uv run python -m causal_meta.analysis.run_analysis experiments/runs \
  --run-id rq1_full_20260319_120000_avici \
  --strict
```

If no `--run-id`/`--run-dir` is provided, analysis discovers all `metrics.json`
under the supplied runs root.
