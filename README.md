# Out-of-Distribution Generalization in Deep Learning-Based Bayesian Causal Discovery

This repository contains the code and experiments for the Master's thesis **"Out-of-Distribution Generalization in Deep Learning-Based Bayesian Causal Discovery"**.

The framework provides a scalable, Hydra-configured environment for benchmarking Bayesian Causal Discovery and Meta-Learning algorithms under distributional shift. It evaluates amortized inference methods (AviCi, BCNP) against explicit Bayesian methods (DiBS, BayesDAG) to test their robustness and the utility of posterior uncertainty.

## Benchmark Components

The evaluation is built around a synthetic Structural Causal Model (SCM) generator that isolates specific distributional shifts between the training simulator and test environments:

- **Evaluated Models:** Amortized inference (AviCi, BCNP) and explicit dataset-specific inference (DiBS, BayesDAG).
- **Distributional Shifts:** The benchmark evaluates out-of-distribution (OOD) generalization across isolated changes in graph topology, mechanism priors (linear, MLP), exogenous noise, problem scale (node count), and sample sizes.
- **Metrics & Diagnostics:** Comprehensive graph metrics (SHD, SID, F1, AUROC) are used alongside likelihood proxies and marginal posterior uncertainty diagnostics to evaluate structural error and robustness.

## Reproduction

The experiments are managed using `uv` for reproducible environment resolution and Hydra for configuration.

### 1. Environment Setup

```bash
# Install the main environment (AviCi, BCNP, DiBS)
uv sync --extra cluster --extra wandb --frozen --no-editable

# Bootstrap the secondary environment (BayesDAG legacy stack)
scripts/bootstrap_uv.sh
```

### 2. Local Verification (Smoke Tests)

Run a small, benchmark-shaped smoke config locally to verify the pipeline:

```bash
uv run causal-meta --config-name dg_2pretrain_smoke model=avici
```

### 3. Cluster Execution (Slurm)

Full experiments are designed to run on a Slurm cluster. Submission scripts are provided in `scripts/`:

```bash
# Run the main benchmark sweep across all models
scripts/submit_all_models.sh main
```

## Documentation

- [Runbook](docs/RUNBOOK.md): Detailed guide on environment setup, running experiments, sweeps, and reproducing analysis figures.
- [Design](docs/DESIGN.md): Architectural overview of the datasets, models, and evaluation pipeline.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
