# Technical Design

## 1. System Overview

The system is a Python-based framework for benchmarking Bayesian Causal Discovery (BCD) methods, specifically focusing on Meta-Learning approaches. It is designed to measure robustness against distribution shifts (o.o.d.) and extend capabilities to full Structural Causal Model (SCM) inference.

The core architecture follows a **Configuration-Driven Pipeline** pattern:
`Config` -> `Data Generation` -> `Model Training` -> `Inference` -> `Evaluation`

## 2. Architecture Layers

### 2.1. Datasets Layer (`src/causal_meta/datasets` / `causal_meta.datasets`)

Responsible for defining the Data Generating Processes (DGPs).

- **Concept: Families vs. Instances**
  - **SCMFamily:** Represents a distribution over SCMs (the "prior"). Uses **composition** to hold generation strategies (`GraphGenerator`, `MechanismFactory`).
  - **SCMInstance:** A concrete realization sampled from a Family. Contains a fixed DAG, fixed mechanism functions, and methods for ancestral sampling.
- **Generators:**
  - `generate_graph.py`: Vectorized algorithms for sampling DAGs (e.g., Erdos-Renyi, Scale-Free).
  - `generate_functions.py`: Factories for mechanism functions (Linear, MLP, etc.).
- **Data Loading Strategy:**
  - **Training:** Uses `MetaIterableDataset` (Infinite Stream). Implements seeding logic inside `__iter__` using `get_worker_info()` (instead of `worker_init_fn`) and rank-aware offsets for massive parallel throughput without duplication.
  - **Evaluation:** Uses `MetaFixedDataset` (Map-style). Backed by a fixed list of seeds to guarantee strict reproducibility across epochs and model runs. Includes **caching** to avoid re-generating SCMs during repeated validation passes.
- **Disjointness:** Enforced via checking structural hashes (e.g. of the adjacency matrix) to ensure O.O.D. sets do not overlap with training sets.

### 2.2. Models Layer (`src/causal_meta/models` / `causal_meta.models`)

Wraps diverse BCD algorithms into a uniform API.

- **BaseCausalModel:** Abstract base class.
  - `train(dataset)`: Updates internal parameters.
  - `sample_graph_posterior(n_samples)`: Returns a list of adjacency matrices.
  - `sample_scm_posterior(n_samples)`: Returns a list of SCM instances (graph + mechanisms).
- **Implementations:**
  - `bncp/`: Bayesian Neural Causal Models (Meta-learning).
  - `dibs/`: Differentiable Bayesian Structure Learning.
  - `avici/`: Amortized Variational Inference.
  - `bayesdag/`: Explicit VI/MCMC baseline.

### 2.3. Runners Layer (`src/causal_meta/runners` / `causal_meta.runners`)

Orchestrates the experiment (Hydra config -> data module -> model -> tasks).

- **Entry point:** `causal_meta.main`
  1. Validates config keys early.
  2. Sets up distributed context (optional).
  3. Builds `CausalMetaModule` (datasets).
  4. Instantiates `Model` via `ModelFactory`.
  5. Runs `pre_training` **or** `inference` depending on `model.needs_pretraining`.
  6. Runs `evaluation`.
- **Metrics:**
  - Graph metrics: E-SHD, E-edgeF1, E-SID, ancestor metrics, edge entropy.
  - Likelihood proxies: `graph_nll` (edge NLL under mean posterior edge probs) and I-NIL via a Linear Gaussian scorer (heuristic for non-linear mechanisms).

## 3. Data Flow

1.  **Initialization:** User provides `experiment.yaml`.
2.  **Data Gen:** `SCMFamily.sample_task(seed)` produces `SCMInstance`.
3.  **Sampling:** `SCMInstance.sample(batch_size)` produces `X_train` (tensor), optionally normalized.
4.  **Training:** `Model.train(X_train)` optimizes variational parameters or weights.
5.  **Inference:** `Model` produces posterior samples $\{G_i, f_i\}$.
6.  **Scoring:** Metrics module compares $\{G_i\}$ vs `G_true` and computes likelihood of `X_test` under $\{f_i\}$.

## 4. Technologies

- **Language:** Python 3.10+
- **Core Libs:** PyTorch (Deep Learning), NetworkX (Graph Utils), NumPy/Pandas.
- **Config:** Hydra or Pydantic-based YAML parsing.
- **Packaging:** `pyproject.toml` (setuptools/hatch).

## 5. Security & Reproducibility

- **Seeds:** Complex seeding strategy required for Meta-Learning.
  - `base_seed` set in config.
  - Training stream seeds derived via `base + rank * workers + worker_id`.
  - Validation seeds are fixed integers.
- **Isolation:** Models run in isolated environments/processes if necessary.
- **Artifacts:** Outputs are written under Hydra run directories (default: `experiments/runs/{name}`), configurable via `CAUSAL_META_RUN_DIR` / `CAUSAL_META_SWEEP_DIR`.

---

# Class Structure

## 1. Datasets Module

### `SCMFamily`

_Defines the distribution from which SCMs are drawn (Meta-Learning Tasks)._

- **Composition:**
  - `graph_generator`: Protocol (Strategy for adjacency sampling).
  - `mechanism_factory`: Protocol (Strategy for mechanism creation).
- **Attributes:**
  - `variable_count`: int
  - `noise_type`: str
  - `mechanism_proportions`: Dict[str, float]
- **Methods:**
  - `sample_task(seed: int) -> SCMInstance`: Pure function using seed to create task.
  - `plot_example(save_path: Optional[str])`: Visualization helper.

### `MetaIterableDataset` (Inherits `IterableDataset`)

_Infinite stream of tasks for training._

- **Attributes:**
  - `scm_family`: SCMFamily
  - `base_seed`: int
- **Methods:**
  - `__iter__()`: Implements worker-aware and rank-aware seeding logic.
  - **Yields:** a pickle-safe dict: `{"seed": int, "data": Tensor, "adjacency": Tensor}`.

### `MetaFixedDataset` (Inherits `Dataset`)

_Fixed set of tasks for validation/testing._

- **Attributes:**
  - `scm_family`: SCMFamily
  - `seeds`: List[int]
  - `samples_per_task`: int
- **Methods:**
  - `__getitem__(idx)`: Returns a dict `{"seed": int, "data": Tensor, "adjacency": Tensor}`.

**Full-SCM training note**

- The dataset outputs intentionally do **not** include `SCMInstance` / mechanism modules (which may be non-picklable under DataLoader workers).
- To do supervised “full SCM” training, reconstruct the teacher SCM inside the training step via `SCMFamily.sample_task(seed)` using the returned `seed`.

### `SCMInstance`

_A specific realization of a causal model._

- **Attributes:**
  - `adjacency_matrix`: torch.Tensor (The DAG)
  - `mechanisms`: List[nn.Module]
  - `topological_order`: List[int]
- **Methods:**
  - `sample(n: int) -> Tensor`: Ancestral sampling.

## 2. Models Module

### `BaseModel` (Inherits `nn.Module`)

_Uniform interface for all causal discovery methods (Meta-Learning & Classical)._

- **Attributes:**
  - `needs_pretraining`: bool (Flag for meta-learners vs instance-optimizers)
- **Methods:**
  - `forward(x: Tensor) -> Any`: Model-specific forward pass (returns logits/params).
  - `sample(x: Tensor, num_samples: int) -> Tensor`: Returns graph samples `(Batch, N, V, V)`.
  - `calculate_loss(output: Any, target: Tensor, **kwargs) -> Tensor`: Computes training loss.

### `BCNP` (Inherits `BaseModel`)

_Bayesian Causal Neural Process implementation._

- **Architecture:** Node Embedding (Set Transformer) -> Edge Prediction (Bilinear/MLP).
- **Components:** `SetTransformerLayer`, `MultiHeadAttention`.

### `AviciModel` (Inherits `BaseModel`)

_Avici-style Amortized Causal Discovery._

- **Architecture:** Sample Aggregation -> Variable Attention (Transformer) -> Edge Prediction.

### `ModelFactory`

- **Methods:**
  - `create(config: Dict) -> BaseModel`: Instantiates registered models.

## 3. Pipeline Module

### `run_pipeline` (in `causal_meta.main`)

Orchestrates the train/eval run (and optional cached inference) based on Hydra config.

- **Responsibilities:**
  - Distributed setup/teardown.
  - Instantiate `CausalMetaModule` and model (`ModelFactory`).
  - Dispatch tasks: `pre_training` or `inference`, then `evaluation`.

### `Metrics`

- **Static Methods:**
  - `compute_shd(true_adj, pred_adj_list) -> float`
  - `compute_auroc(true_adj, marginal_probs) -> float`
  - `compute_nil(test_data, posterior_scms) -> float`

---

# Organisation

The code is organised into two separate parts:

- `./client`: Organises a static website published on GH-pages. This website should explain the thesis and automatically link results of the thesis and some insights in the future.
- `./src`: Here lives the source code of the thesis.
