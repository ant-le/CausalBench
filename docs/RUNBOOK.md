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

## 9. Data Acquisition & Licensing (SynTReN)

The `real_syntren` test family (d=20, following the BayesDAG benchmark setting)
is not shipped with this repository. Its loader expects local files placed in
`src/causal_meta/datasets/real_world/_cache/syntren/` (or a custom `data_dir`):

- `data.csv` / `data.tsv` —  expression matrix (rows=samples, cols=genes)
- `adjacency.csv` / `adjacency.tsv` —  ground-truth DAG (d × d)
- or a single `data.npz` with `data` and `adjacency` arrays.

**Acquisition.** SynTReN is a Java-based generator published as supplementary
software for Van den Bulcke et al. (2006). Download `SynTReN.zip` (v1.2,
2007-06-08) from the authors' page
(<http://bioinformatics.intec.ugent.be/kmarchal/SynTReN/>) and run
`java -jar SynTReN.jar`, then export/save the resulting network adjacency and
expression matrix in the layout above. The BayesDAG/DiBS papers also used
SynTReN datasets; this repository intentionally does not bundle them.

**Licensing.** The SynTReN paper is open access (CC BY 2.0, BioMed Central);
the tool is distributed freely by the authors (publicly listed as freeware).
Cite the source article when using the data:

- Van den Bulcke, T., Van Leemput, K., Naudts, B., van Remortel, P., Ma, H.,
  Verschoren, A., De Moor, B., Marchal, K. (2006). "SynTReN: a generator of
  synthetic gene expression data for design and analysis of structure learning
  algorithms." *BMC Bioinformatics*, 7:43. doi:10.1186/1471-2105-7-43.
