# Paper Comparison Module -- Implementation Plan

## 1. Goal

Build a lightweight sub-package under `src/causal_meta/analysis/paper_comparison/`
that produces an **appendix-only plausibility check**: are our thesis runs of
AVICI, BCNP, DiBS, and BayesDAG producing results in a reasonable range given
what the source papers reported?

This is **not** a reproduction claim or statistical comparison. It is a sanity
check that our implementations behave plausibly under settings that overlap with
published results.

Output target: **Appendix F (Reproducibility Artifacts)** only.

---

## 2. The Overlap

### 2.1 The BCNP paper as Rosetta Stone

The BCNP paper (meta_learning.md) is the single best comparison source because
its Tables 7--15 report results for **all four models** (BCNP, AVICI, DiBS,
BayesDAG) on settings that closely match our thesis families:

| Paper setting                 | Thesis family                | Mismatch      |
| ----------------------------- | ---------------------------- | ------------- |
| Linear, ER20, d=20, n=1000    | `id_linear_er20_d20_n500`    | n=1000 vs 500 |
| Linear, ER40, d=20, n=1000    | `id_linear_er40_d20_n500`    | n=1000 vs 500 |
| Linear, ER60, d=20, n=1000    | `id_linear_er60_d20_n500`    | n=1000 vs 500 |
| NeuralNet, ER20, d=20, n=1000 | `id_neuralnet_er20_d20_n500` | n=1000 vs 500 |
| NeuralNet, ER40, d=20, n=1000 | `id_neuralnet_er40_d20_n500` | n=1000 vs 500 |
| NeuralNet, ER60, d=20, n=1000 | `id_neuralnet_er60_d20_n500` | n=1000 vs 500 |
| GPCDE, ER20, d=20, n=1000     | `id_gpcde_er20_d20_n500`     | n=1000 vs 500 |
| GPCDE, ER40, d=20, n=1000     | `id_gpcde_er40_d20_n500`     | n=1000 vs 500 |
| GPCDE, ER60, d=20, n=1000     | `id_gpcde_er60_d20_n500`     | n=1000 vs 500 |

**9 settings × 4 models = 36 comparison points.**

The only systematic mismatch is **n=1000 (paper) vs n=500 (ours)**. Same d, same
graph type, same edge density convention, same mechanism names.

### 2.2 Comparable metrics

| Our metric key  | Paper column name | Direction        |
| --------------- | ----------------- | ---------------- |
| `auc_mean`      | AUC               | higher is better |
| `e-shd_mean`    | Expected SHD      | lower is better  |
| `e-edgef1_mean` | Expected Edge F1  | higher is better |

The paper also reports "Log Probability" which may correspond to our
`graph_nll_mean` (sign-flipped). Include if the mapping is confirmed; omit if
unclear.

### 2.3 What is NOT comparable (and is excluded)

- AVICI paper's own tables: all d=30 n=1000, different domains (LINEAR/RFF/GRN
  ≠ our linear/neuralnet/gpcde), no d=20 results in the paper.
- DiBS paper's own tables: only Sachs (d=11) in tables; d=20 results are
  box-plot figures with no extractable numbers.
- BayesDAG paper's own tables: d=70/d=100 and SynTReN/Sachs only; d=20 results
  are in appendix figures without numbers.
- All SF (scale-free) families: no paper reports SF results at these settings.
- All OOD families: no paper comparison data.

---

## 3. Reference Data Source (BCNP paper Tables 7--15)

The values below are extracted from the BCNP paper. For each model we use the
row matching that model from the corresponding table. For BCNP itself we use the
"BCNP All Data" variant (trained on all densities, closest to our unified
training setup).

### 3.1 Paper values to encode (per table)

**Table 7** -- Linear ER20 (d=20, n=1000):

- DiBS: AUC=0.73, E-SHD=25.41, E-F1=0.16
- BayesDAG: AUC=0.52, E-SHD=23.02, E-F1=0.03
- AVICI: AUC=0.89, E-SHD=26.61, E-F1=0.31
- BCNP All Data: AUC=0.89, E-SHD=23.35, E-F1=0.44

**Table 8** -- Linear ER40:

- DiBS: AUC=0.53, E-SHD=72.43, E-F1=0.11
- BayesDAG: AUC=0.51, E-SHD=44.07, E-F1=0.03
- AVICI: AUC=0.86, E-SHD=56.22, E-F1=0.28
- BCNP All Data: AUC=0.90, E-SHD=49.01, E-F1=0.40

**Table 9** -- Linear ER60:

- DiBS: AUC=0.40, E-SHD=106.64, E-F1=0.11
- BayesDAG: AUC=0.51, E-SHD=63.51, E-F1=0.04
- AVICI: AUC=0.83, E-SHD=80.90, E-F1=0.31
- BCNP All Data: AUC=0.86, E-SHD=72.15, E-F1=0.36

**Table 10** -- NeuralNet ER20:

- DiBS: AUC=0.69, E-SHD=28.48, E-F1=0.12
- BayesDAG: AUC=0.59, E-SHD=90.21, E-F1=0.09
- AVICI: AUC=0.84, E-SHD=29.51, E-F1=0.24
- BCNP All Data: AUC=0.88, E-SHD=30.05, E-F1=0.36

**Table 11** -- NeuralNet ER40:

- DiBS: AUC=0.67, E-SHD=46.81, E-F1=0.13
- BayesDAG: AUC=0.59, E-SHD=97.25, E-F1=0.15
- AVICI: AUC=0.79, E-SHD=61.11, E-F1=0.24
- BCNP All Data: AUC=0.84, E-SHD=53.44, E-F1=0.37

**Table 12** -- NeuralNet ER60:

- DiBS: AUC=0.64, E-SHD=65.72, E-F1=0.13
- BayesDAG: AUC=0.58, E-SHD=103.95, E-F1=0.19
- AVICI: AUC=0.79, E-SHD=85.15, E-F1=0.29
- BCNP All Data: AUC=0.83, E-SHD=68.34, E-F1=0.39

**Table 13** -- GPCDE ER20:

- DiBS: AUC=0.80, E-SHD=26.33, E-F1=0.18
- BayesDAG: AUC=0.67, E-SHD=99.69, E-F1=0.11
- AVICI: AUC=0.74, E-SHD=36.27, E-F1=0.08
- BCNP All Data: AUC=0.83, E-SHD=38.24, E-F1=0.20

**Table 14** -- GPCDE ER40:

- DiBS: AUC=0.74, E-SHD=45.04, E-F1=0.16
- BayesDAG: AUC=0.67, E-SHD=103.09, E-F1=0.19
- AVICI: AUC=0.72, E-SHD=67.68, E-F1=0.15
- BCNP All Data: AUC=0.78, E-SHD=62.05, E-F1=0.22

**Table 15** -- GPCDE ER60:

- DiBS: AUC=0.67, E-SHD=64.22, E-F1=0.13
- BayesDAG: AUC=0.66, E-SHD=107.46, E-F1=0.24
- AVICI: AUC=0.71, E-SHD=91.33, E-F1=0.21
- BCNP All Data: AUC=0.77, E-SHD=80.28, E-F1=0.25

### 3.2 Standard errors

All paper values include `± std_of_mean`. These should be stored alongside the
point estimates for context but are not needed for the plausibility assessment
itself (we're looking at ballpark agreement, not statistical tests).

---

## 4. Module Layout

```
src/causal_meta/analysis/paper_comparison/
    __init__.py
    PLAN.md                     # ← this file
    reference_data.py           # all paper values in one module (simple dict/list)
    comparison.py               # load thesis metrics, compare, produce report
    tables.py                   # LaTeX table generator for Appendix F
```

Four files of code. No complex schema, no matching logic (the 9 families are
hardcoded), no separate reference files per paper.

---

## 5. Reference Data (`reference_data.py`)

A single module with one flat data structure:

```python
# Each entry: (family_key, model, metric) → (paper_value, paper_std, paper_table)
PAPER_RESULTS: dict[tuple[str, str, str], PaperEntry] = { ... }
```

Where `PaperEntry` is a simple NamedTuple:

```python
class PaperEntry(NamedTuple):
    value: float
    std: float           # std of mean (as reported)
    table: str           # e.g. "BCNP Table 7"
    paper_n: int = 1000  # paper used n=1000
```

This encodes all 36 comparison points × 3 metrics = 108 entries. Flat, auditable,
no indirection.

The 9 family keys and their paper-side labels:

```python
COMPARABLE_FAMILIES: dict[str, str] = {
    "id_linear_er20_d20_n500": "Linear ER20",
    "id_linear_er40_d20_n500": "Linear ER40",
    "id_linear_er60_d20_n500": "Linear ER60",
    "id_neuralnet_er20_d20_n500": "NeuralNet ER20",
    "id_neuralnet_er40_d20_n500": "NeuralNet ER40",
    "id_neuralnet_er60_d20_n500": "NeuralNet ER60",
    "id_gpcde_er20_d20_n500": "GPCDE ER20",
    "id_gpcde_er40_d20_n500": "GPCDE ER40",
    "id_gpcde_er60_d20_n500": "GPCDE ER60",
}
```

---

## 6. Comparison Engine (`comparison.py`)

### 6.1 Inputs

```python
def run_comparison(
    thesis_runs_root: Path,   # experiments/thesis_runs/
) -> ComparisonReport:
```

### 6.2 Steps

1. For each model in {avici, bcnp, dibs, bayesdag}:
   a. Load `{model}/metrics.json` → extract `summary` dict.
   b. For each of the 9 comparable families:
   - Read `auc_mean`, `e-shd_mean`, `e-edgef1_mean` (+ `_std` variants).
   - Look up the corresponding paper value from `reference_data.py`.
   - Compute delta = thesis − paper.
   - Flag plausibility: is the thesis value within a reasonable range?
2. Assemble a flat DataFrame with columns:
   `[model, family, metric, paper_value, paper_std, thesis_value, thesis_std, delta, paper_table]`
3. Return a `ComparisonReport` containing the DataFrame and a note about
   the n=1000 vs n=500 mismatch.

### 6.3 Plausibility heuristic

No formal test. A result is "plausible" if:

- Same order of magnitude as the paper value.
- Direction of model ranking is broadly preserved (e.g. if BCNP > AVICI > DiBS
  in the paper, we expect something similar).

This is communicated in prose in the appendix, not as a pass/fail flag.

---

## 7. LaTeX Table (`tables.py`)

### 7.1 One table per metric (3 tables total)

Each table has:

- Rows: 9 settings (Linear ER20, ..., GPCDE ER60), grouped by mechanism.
- Columns: Model pairs — for each of {AVICI, BCNP, DiBS, BayesDAG}: Paper | Ours.
- A table footnote: "Paper values from [BCNP ref], Tables 7–15. Paper uses
  n=1000; our benchmark uses n=500. All other settings (d=20, graph type, edge
  density, mechanism) match."

### 7.2 Example structure

```
                 AVICI           BCNP            DiBS            BayesDAG
Setting     Paper  Ours    Paper  Ours    Paper  Ours    Paper  Ours
─────────────────────────────────────────────────────────────────────────
Lin ER20    0.89   0.XX    0.89   0.XX    0.73   0.XX    0.52   0.XX
Lin ER40    0.86   0.XX    0.90   0.XX    0.53   0.XX    0.51   0.XX
...
```

### 7.3 Output path

`paper/final_thesis/graphics/F_ReproducibilityArtifacts/paper_plausibility_*.tex`

---

## 8. Integration

Add a call in `run_thesis_analysis.py`:

```python
from causal_meta.analysis.paper_comparison import generate_paper_comparison
generate_paper_comparison(thesis_runs_root, thesis_root)
```

---

## 9. Implementation Order

| Step | File                | Description                                      |
| ---- | ------------------- | ------------------------------------------------ |
| 1    | `reference_data.py` | Encode all 108 paper entries from Section 3.     |
| 2    | `comparison.py`     | Load thesis metrics, build comparison DataFrame. |
| 3    | `tables.py`         | Generate 3 LaTeX tables.                         |
| 4    | `__init__.py`       | Export `generate_paper_comparison`.              |
| 5    | Integration         | Hook into `run_thesis_analysis.py`.              |
| 6    | Test                | `tests/analysis/test_paper_comparison.py`.       |

---

## 10. Known Caveat (to state in appendix text)

> The paper results were obtained with n=1000 observations per dataset, whereas
> our benchmark uses n=500. With fewer observations, we generally expect higher
> E-SHD and lower AUC/F1 for all models. The comparison is therefore directional:
> we expect our results to be somewhat worse than the paper values. A result
> substantially _better_ than the paper at half the sample size would warrant
> investigation, as would a result orders of magnitude worse.
