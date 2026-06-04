# Thesis Results Pipeline

## Goal

Rebuild the thesis results section from one curated run per model and write all
machine-generated thesis artifacts into `paper/final_thesis/generated/`.

## Input Contract

The analysis entry point is:

```text
experiments/thesis_runs/
  avici/
  bcnp/
  dibs/
  bayesdag/
```

Each model directory is manually curated by the user and should contain the
selected run artifacts for that model.

Minimum expected contents per model directory:

- `metrics.json`
- `inference/`
- `main.log`

The user can replace any model directory when a better run becomes available.

## Output Contract

The only writable thesis output root is:

```text
paper/final_thesis/generated/
```

This directory is treated as disposable build output and is rebuilt from
scratch on every analysis run.

Generated subdirectories:

```text
paper/final_thesis/generated/
  figures/
  tables/
  data/
  snippets/
  provenance/
```

`paper/final_thesis/graphics/` remains reserved for static, hand-maintained
assets.

## Source of Truth

The thesis results pipeline uses the structured run artifacts from the curated
run folders.

Primary sources:

- `metrics.json` for metadata, family metadata, summary metrics, raw per-task
  metrics, and shift distances
- `inference/` for posterior diagnostics

The pipeline does **not** use W\&B as the thesis backend, and it does **not**
parse Slurm logs for the core quantitative results.

## Generated Outputs

The thesis-specific pipeline currently writes:

- `generated/tables/results_anchor.tex`
- `generated/tables/ood_detection.tex`
- `generated/tables/fixed_ood_appendix.tex`
- `generated/tables/distance_regression.tex`
- `generated/figures/fixed_ood_degradation.pdf`
- `generated/figures/node_transfer.pdf`
- `generated/figures/sample_transfer.pdf`
- `generated/figures/selective_prediction.pdf`
- `generated/figures/failure_modes.pdf`
- `generated/figures/event_probabilities.pdf`
- `generated/figures/posterior_diagnostics.pdf`
- CSV exports under `generated/data/`
- provenance records under `generated/provenance/`
- TeX snippets under `generated/snippets/`

## Thesis Synchronization

`paper/final_thesis/sections/5_Results.tex` now conditionally includes the
generated artifacts using `\IfFileExists`, so the results chapter automatically
updates when the pipeline is rerun.

## Rebuild Command

Run the thesis analysis with:

```bash
PYTHONPATH=src uv run python -m causal_meta.analysis.run_thesis_analysis
```

Optional arguments:

```bash
PYTHONPATH=src uv run python -m causal_meta.analysis.run_thesis_analysis \
  --input-root experiments/thesis_runs \
  --thesis-root paper/final_thesis
```

Use `--best-effort` if optional diagnostics should be skipped instead of causing
the analysis to fail.

If the repository has not been installed editably into the active `uv`
environment, keep the `PYTHONPATH=src` prefix so `causal_meta` resolves from the
checkout.

## Rebuild Semantics

The pipeline writes into a temporary sibling directory first and only replaces
`paper/final_thesis/generated/` after a successful run. This avoids leaving the
thesis with half-written artifacts.

## Git / Repository Rule

`paper/final_thesis/` is a separate Git repository.

Therefore:

- `generated/` is ignored via `paper/final_thesis/.gitignore`
- nothing inside `generated/` should be committed
- all generated outputs are local and reproducible
