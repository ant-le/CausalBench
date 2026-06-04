# Thesis Viability Plan: RQ1-Only Scope

This document outlines the additions needed to make the thesis viable as a
Data Science master's thesis if RQ2 (full SCM inference) is dropped and
full-cluster results confirm that meta-learners perform poorly under OOD
shifts. Each section describes the analytical contribution, what already
exists in the codebase, and what needs to be implemented.

## Architecture Note

Each model evaluation run writes a self-contained **`metrics.json`** with
the structure `{metadata, summary, raw}`. The `metadata` block carries
`run_id`, `run_name`, `model_name`, and `output_dir`. The `summary` block
maps `dataset_key → {metric_mean, metric_sem, metric_std, …}`. The `raw`
block maps `dataset_key → {metric: [per-task values]}`.

Analysis is run post-hoc by selecting run directories (by ID or path) via
`resolve_run_directories()` and loading them into a unified DataFrame via
`load_runs_dataframe()`. There is no shared mutable `overview.json`.

All new analysis code should consume the standard `pd.DataFrame` returned
by `load_runs_dataframe()` (columns: `RunID`, `RunName`, `RunDir`, `Model`,
`Dataset`, `ModelKey`, `DatasetKey`, `Metric`, `Mean`, `SEM`, `Std`).

---

## 1. Posterior Failure Diagnostics

**Goal:** Transform "meta-learners degrade under shift" from a single number
into a characterised set of posterior failure behaviours by analysing where
the model places probability mass under OOD shift (empty graphs, dense
graphs, fragmented skeletons, correct skeleton / wrong orientation, etc.).

### What exists

- `runners/tasks/evaluation.py` and `runners/tasks/inference.py` already cache
  posterior graph samples as `.pt` artifacts under each run's inference
  directory.
- Each artifact contains the sampled graphs and ground-truth adjacency needed
  for post-hoc posterior analysis.
- `metrics.json` already gives run-level metadata and dataset-level summaries,
  but it is not sufficient for posterior-native failure analysis because it
  collapses each task to scalar summaries.

### What needs to be implemented

| Item                                     | Location                                        | Description                                                                                                                                                                                                                                                                                                            |
| ---------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Artifact-based posterior loader**      | New file `analysis/failure_modes.py`            | Load cached inference artifacts (`.pt` / `.pt.gz`) from selected run directories. For each task, recover posterior graph samples and the true adjacency, together with run/dataset metadata from `metrics.json`. This becomes the canonical input for posterior failure analysis.                                      |
| **Posterior diagnostic primitives**      | New helpers in `analysis/failure_modes.py`      | For every posterior sample, compute graph density, density ratio relative to truth, skeleton F1, orientation accuracy, and number of connected components in the predicted skeleton. Do not collapse to a posterior mean graph before computing these diagnostics.                                                     |
| **Posterior event probabilities**        | New helpers in `analysis/failure_modes.py`      | Define sample-level events and report per-task posterior probabilities such as `P(empty)`, `P(dense relative to truth)`, `P(skeleton correct & orientation wrong)`, and `P(fragmented ∣ truth connected)`. These event probabilities replace brittle one-label-per-task classification as the primary analysis object. |
| **Posterior summary statistics**         | New helpers in `analysis/failure_modes.py`      | For each task, compute exact posterior summaries (mean, std, quantiles) for density ratio, orientation accuracy, skeleton F1, and connected-component count. These are the quantities shown in tables/plots.                                                                                                           |
| **Failure diagnostics figure generator** | New functions in `analysis/plots/`              | Replace the grouped stacked bar chart with posterior-native diagnostics: e.g. event-probability bar charts, violin/box plots for density ratio and orientation accuracy, and/or connected-component distributions by OOD condition. Wire these into `generate_all_artifacts_from_runs()`.                              |
| **Optional taxonomy view**               | `analysis/failure_modes.py` + `analysis/plots/` | If a simple categorical summary is still useful for presentation, derive it only as a secondary view from posterior event probabilities (for example by assigning the dominant failure event per task). The thesis should emphasise posterior mass and exact numbers, not thresholded posterior-mean labels.           |
| **Update `DATASET_DESCRIPTION_MAP`**     | `analysis/utils.py:36`                          | Replace the 6 stale smoke-test keys with the 27 family keys from `full.yaml` (e.g. `id_linear_er20`, `ood_graph_sbm_linear`, `ood_mech_pnl_tanh_er40`, etc.) so that full-sweep plots render readable labels.                                                                                                          |

---

## 2. Posterior Calibration Analysis

**Goal:** Determine whether meta-learners become overconfident (low entropy,
wrong answers) or appropriately uncertain under OOD shift. Overconfident-and-
wrong is a safety-critical finding.

### What exists

- `edge_entropy()` in `runners/metrics/graph.py:234` is already computed and
  stored in each run's `metrics.json` summary for every model × dataset.
- `graph_nll` (line 214) is also already computed everywhere.
- `_bernoulli_log_prob()` (line 196) gives per-graph log-probs.
- Raw per-task values for both metrics are in `metrics.json` under `"raw"`.

### What needs to be implemented

| Item                                 | Location                                               | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Entropy-vs-accuracy scatter**      | New function in `analysis/plots/`                      | For each model, plot `edge_entropy` (x) vs `e-shd` (y) with one point per dataset. Colour by ID/OOD category. An ideal model would show high entropy → high SHD (uncertain and wrong) and low entropy → low SHD (confident and right). Meta-learners that are overconfident under OOD will cluster at low-entropy, high-SHD — the dangerous quadrant. Data source: the `pd.DataFrame` from `load_runs_dataframe()` (both metrics are already in the summary). |
| **Edge-level calibration curve**     | New function in `analysis/plots/` or standalone script | Requires raw posterior edge probabilities (available from cached inference `.pt` files). Bin predicted edge probabilities into deciles, compute fraction of true edges in each bin. Plot reliability diagram. One curve per model, one panel per OOD condition. Implementation: load cached `(S,B,N,N)` samples → compute mean edge probs `(B,N,N)` → flatten → bin → plot.                                                                                   |
| **Expected Calibration Error (ECE)** | New metric in `runners/metrics/graph.py`               | Standard ECE: weighted average of `∣accuracy_bin − confidence_bin∣` across bins. Register in `_compute_batch_metrics()` so it flows into `metrics.json`.                                                                                                                                                                                                                                                                                                      |

---

## 3. Training Diversity Ablation

**Goal:** Determine whether OOD failure is caused by insufficient training
diversity or is a fundamental architectural limitation. This turns an
observation into an explanation.

### What exists

- The training distribution is defined in `configs/data/_base.yaml` under
  `train_family`. Currently a mixture of ER+SF graphs and Linear+MLP+GP
  mechanisms.
- The config/factory system makes it trivial to create new data configs via
  YAML alone (`configs/data/`). No code changes needed.
- The current main sweep config (`dg_2pretrain_multimodel.yaml`) defines the benchmark matrix directly.

### What needs to be implemented

| Item                                | Location                          | Description                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ablation data configs**           | New YAML files in `configs/data/` | Create 3 new data configs inheriting `_base.yaml` but overriding `train_family.mech_cfg`: (1) `ablation_linear_only.yaml` — linear mechanisms only, (2) `ablation_linear_mlp.yaml` — Linear+MLP mixture (no GP), (3) keep `full.yaml` as the full-mixture condition. All three must share identical `test_families` so results are comparable. |
| **Ablation sweep config**           | New YAML in `configs/`            | A Hydra multirun config that sweeps over `data: ablation_linear_only, ablation_linear_mlp, full` and `model: bcnp` (or both bcnp and avici). Each run produces its own `metrics.json`.                                                                                                                                                         |
| **Ablation comparison plot**        | New function in `analysis/plots/` | Load the standard DataFrame from `load_runs_dataframe()` for selected ablation run directories. Produce a grouped bar chart: x-axis = OOD condition, groups = training diversity level (identified via `RunName` or `RunID`), y-axis = E-SID.                                                                                                  |
| **Graph topology ablation configs** | New YAML files in `configs/data/` | (1) `ablation_er_only.yaml` — ER-only training graphs, (2) `ablation_er_sf.yaml` — ER+SF mixture (current default). Same test families. Purpose: isolate whether SBM failure is about missing structural diversity.                                                                                                                            |

---

## 4. OOD Detection via Uncertainty Threshold

**Goal:** Provide a practical, actionable takeaway: use posterior uncertainty
to detect when a meta-learner's output should not be trusted, and fall back
to an explicit method.

### What exists

- `edge_entropy` is already computed for every prediction.
- `graph_nll` is already computed.
- Both are in per-run `metrics.json` summaries at the per-dataset level.
- Raw per-task values are in `metrics.json` under `"raw"` as parallel lists
  (e.g. `raw["id_linear_er20"]["edge_entropy"] = [0.12, 0.15, ...]`).

### What needs to be implemented

| Item                                  | Location                                       | Description                                                                                                                                                                                                                                                                                                 |
| ------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Per-task entropy + accuracy pairs** | New helper in `analysis/` or standalone script | Load `metrics.json` `"raw"` blocks from selected run directories. For each task, pair `edge_entropy` with `e-shd`. Label each task as ID or OOD based on dataset key prefix (`id_*` vs `ood_*`).                                                                                                            |
| **OOD detection AUROC**               | New module `analysis/ood_detection.py`         | Treat the problem as binary classification: ID tasks = negative, OOD tasks = positive. Use `edge_entropy` (or `graph_nll`) as the detection score. Compute AUROC and AUPRC for each meta-learner. Report in a small table. If entropy reliably separates ID from OOD, this is a practically useful finding. |
| **Selective prediction analysis**     | Same module                                    | Simulate a policy: "reject predictions where entropy > threshold T, use DiBS instead." Sweep T, plot effective accuracy (E-SID of accepted predictions) vs coverage (fraction of predictions accepted). This produces a **Pareto frontier** of accuracy vs. compute cost.                                   |
| **OOD detection figure**              | New function in `analysis/plots/`              | (1) Histogram of per-task entropy for ID vs OOD tasks (overlaid). (2) Selective prediction Pareto curve. Wire into `generate_all_artifacts_from_runs()`.                                                                                                                                                    |

---

## 5. Quantitative Shift Distance vs. Degradation

**Goal:** Replace categorical OOD labels ("PNL is hard") with a quantitative
relationship between distributional distance and performance degradation.

### What exists

- `CausalMetaModule._compute_spectral_distance()` in
  `datasets/data_module.py:286` computes spectral L1 distance between train
  and each test family. Stored in `self.spectral_distances`.
- `datasets/utils/stats.py` provides `compute_family_distance(metric="spectral"|"kl")`
  and `get_family_stats()` (avg degree, sparsity, spectral radius).
- These distances are computed during `setup()` but are **not currently saved
  to disk or included in `metrics.json`**.

### What needs to be implemented

| Item                                     | Location                                  | Description                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Persist family distances**             | `runners/tasks/evaluation.py`             | After `data_module.setup()`, write `data_module.spectral_distances` into a sidecar file (`family_distances.json`) alongside `metrics.json` in the run directory. Also compute KL degree-distribution distances via `compute_family_distance(metric="kl")` for each test family and save alongside. Alternatively, add a `"distances"` top-level key to `metrics.json` itself. |
| **Mechanism-level distance metric**      | New function in `datasets/utils/stats.py` | The existing distance metrics are graph-only. Add a data-level distance: sample N tasks from train and test families, compute a simple statistic on the generated data (e.g., mean pairwise mutual information, or non-linearity score via comparing linear-fit R² to actual data variance). This quantifies how different the functional relationships are.                  |
| **Distance-vs-degradation scatter plot** | New function in `analysis/plots/`         | Load `family_distances.json` and the standard DataFrame from `load_runs_dataframe()`. For each test family, plot spectral distance (x) vs E-SID degradation relative to ID performance (y). One series per model. Fit a trend line. This is the key figure that upgrades the analysis from categorical to quantitative.                                                       |
| **Multi-distance regression table**      | New function in `analysis/tables/`        | Regress E-SID degradation on (spectral distance, KL degree distance, mechanism distance) using OLS. Report R², coefficients, and p-values. This answers: "which type of shift predicts degradation most strongly?"                                                                                                                                                            |

---

## 6. Per-Graph-Size and Density Breakdown

**Goal:** Show how OOD robustness scales with graph size and density.

### What exists

- **All current configs use `n_nodes: 20` globally** (`_base.yaml:12`). No
  test family overrides this. The ER20/40/60 suffixes refer to **expected
  edge counts** (sparsity levels 0.0526 / 0.1053 / 0.1579), not node counts.
  SF1/2/3 refers to the Barabasi-Albert attachment parameter `m`.
- The existing ER sparsity variation is useful (it tests density sensitivity),
  but it does not test scalability with graph size.
- The `FamilyConfig` dataclass already supports per-family `n_nodes` overrides.

### What needs to be implemented

| Item                               | Location                                                                  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Density-stratified plot**        | New function in `analysis/plots/`                                         | The existing ER20/40/60 families (which vary sparsity at fixed `n_nodes=20`) already enable a density analysis. Plot E-SID vs sparsity level for each model. This is pure plotting code against the standard `load_runs_dataframe()` DataFrame — no new experiments needed.                                                                                                                                                                 |
| **Node-count test configs**        | New families in `configs/data/full.yaml` or a separate `scalability.yaml` | Add test families at `n_nodes: 10, 20, 40` (or 15, 20, 30 if 40 is too expensive for explicit baselines). Each family overrides the global `n_nodes` at the family level. Use a single mechanism (e.g. linear) and single graph type (e.g. ER at matched sparsity) to isolate the size effect. Repeat for one OOD condition (e.g. PNL-tanh). This is YAML-only — no code changes needed — but requires additional cluster time to evaluate. |
| **Size-stratified plot**           | New function in `analysis/plots/`                                         | Once node-count families exist: plot E-SID vs `n_nodes` for each model, with separate line styles for ID vs OOD. If degradation accelerates with graph size, that is a scalability finding.                                                                                                                                                                                                                                                 |
| **Enrich `metrics.json` metadata** | `runners/tasks/evaluation.py`                                             | Extend the `metadata` block to include per-family properties (`n_nodes`, graph type, mechanism type, sparsity). Currently metadata only has `run_id`, `run_name`, `model_name`, `output_dir`. Adding family-level metadata enables richer post-hoc slicing in `load_runs_dataframe()` without name-parsing hacks. One approach: add a `"family_metadata"` key mapping `dataset_key → {n_nodes, graph_type, mech_type, sparsity}`.           |

---

## 7. Paper Restructuring

**Goal:** Reframe the thesis around a single deep RQ1 with sub-questions,
removing the RQ2 promise.

### What needs to change (no code — LaTeX only)

| File                                      | Change                                                                                                                                                                                                                                           |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `sections/1_Introduction.tex` lines 58-69 | Remove RQ2. Rewrite as RQ1 with sub-questions: (a) What are the degradation patterns? (b) What failure modes emerge? (c) Can uncertainty detect OOD? (d) Does training diversity mitigate degradation?                                           |
| `sections/1_Introduction.tex` lines 74-82 | Rewrite contributions: (1) Systematic OOD benchmark, (2) Failure mode taxonomy, (3) Practical uncertainty-based OOD detection rule. Remove "Full SCM Inference Framework."                                                                       |
| `sections/4_Methodology.tex` lines 80-88  | Expand Model Development to describe what is being evaluated (BCNP, Avici), the ablation design, and the OOD detection approach. Remove forward-references to full SCM inference.                                                                |
| `sections/4_Methodology.tex` lines 95-127 | Remove the duplicated OOD description (it repeats section 4.1.3 nearly verbatim). Replace with the ablation experiment design and the calibration / OOD detection methodology.                                                                   |
| `sections/5_Results.tex`                  | Major expansion needed (~3-4x current length). Add subsections for: failure mode analysis, calibration results, ablation results, distance-vs-degradation analysis, per-graph-size results, OOD detection AUROC. Add all new figures and tables. |
| `sections/6_Conclusion.tex`               | Move "Full SCM Inference" from absent-contribution to Future Work (already partially done). Strengthen the conclusions around practical guidance: when to trust meta-learners, when to fall back to explicit methods.                            |
| `sections/2_Background.tex` lines 288-301 | Currently promises quantitative shift distance metrics. Ensure the methodology and results sections deliver on this promise.                                                                                                                     |
| `sections/3_RelatedWork.tex` line 77      | Update the "Ours" row in Table 3 to reflect the actual contributions (robustness benchmarking + failure mode analysis + OOD detection), not "Graph" output.                                                                                      |

---

## Priority and Sequencing

The items above are ordered by impact-to-effort ratio:

1. **Posterior failure diagnostics** (Section 1) — highest impact, moderate
   effort. Transforms the core negative result into a publishable posterior
   diagnostic rather than a thresholded taxonomy.
2. **Posterior calibration** (Section 2) — high impact, low effort.
   `edge_entropy` already exists; the scatter plot is trivial.
3. **OOD detection** (Section 4) — high practical impact, low-moderate effort.
   Provides the actionable takeaway the thesis needs.
4. **Shift distance vs. degradation** (Section 5) — moderate impact, moderate
   effort. Upgrades analysis from categorical to quantitative.
5. **Training diversity ablation** (Section 3) — high explanatory value, but
   requires additional cluster runs (wall-clock cost).
6. **Per-graph-size and density breakdown** (Section 6) — density analysis is
   free (data exists in the full sweep). Node-count analysis requires new
   test configs and additional cluster time.
7. **Paper restructuring** (Section 7) — necessary but not code work.

---

## Implementation Phases

| Phase                          | Work                                                                    | Blocks on                            |
| ------------------------------ | ----------------------------------------------------------------------- | ------------------------------------ |
| **A. Posterior loaders**       | Implement artifact loading + posterior diagnostic primitives            | Nothing — can start now              |
| **B. DATASET_DESCRIPTION_MAP** | Update the 6 stale keys to 27 full.yaml keys (`analysis/utils.py`)      | Nothing — can start now              |
| **C. Metadata enrichment**     | Extend `metrics.json` metadata with family properties (`evaluation.py`) | Nothing — can start now              |
| **D. Distance persistence**    | Persist spectral/KL distances to run directory (`evaluation.py`)        | Nothing — can start now              |
| **E. Analysis plots/tables**   | All new figures and tables (`analysis/plots/`, `analysis/tables/`)      | Phases A-D (needs enriched data)     |
| **F. Posterior diagnostics**   | New `analysis/failure_modes.py` with event probabilities + summaries    | Phase A                              |
| **G. OOD detection module**    | New `analysis/ood_detection.py`                                         | Phase E (needs raw data loading)     |
| **H. Ablation configs**        | YAML-only (`configs/data/`)                                             | Nothing — can start now              |
| **I. Cluster runs**            | Execute full + ablation sweeps                                          | Phases A-D, H (needs code + configs) |
| **J. Paper restructuring**     | LaTeX edits                                                             | Phase I (needs results)              |
