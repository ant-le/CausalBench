# Goals

1.  **Efficient Meta-Learning Pipeline:** Provide a dataset workflow optimized for `torch` that supports both "infinite" stream-based training and reproducible fixed-set evaluation.
2.  **Descriptive Analysis:** Include functionality for plotting, summary statistics, and inspection of generated SCMs.
3.  **Distributional Control:** Ability to quantify differences between datasets (OOD vs ID) and ensure disjoint sets via hashing.

# Components

## 1. DataModule (Dataset Handler)
> The central entry point that manages configuration, instantiation, and splitting of datasets.

    -   **Responsibilities:**
        -   **Configuration Management:** Accepts and validates explicit `FamilyConfig` objects (e.g., Hydra/OmegaConf). Logs these configs at instantiation to ensure ID/OOD conditions are fully reproducible.
        -   Instantiates `MetaIterableDataset` for training and `MetaFixedDataset` for validation/testing.    -   **Disjoint Set Enforcement:**
        -   Uses graph hashing (e.g., `hash(adjacency_matrix.tobytes())`) to ensure no exact graph overlaps between Training and O.O.D. Testing sets.
        -   **Retry Logic:** If a sampled training task collides with a reserved test hash, re-sample immediately with a new seed.
    -   **Distance Calculation:** Computes distributional distances (e.g., KL divergence approximation or spectral distances) between generic I.D. and O.O.D. families.
        -   **Optimization:** Precomputes family-level statistics (spectral/KL) rather than computing per-batch, as online calculation is expensive.
-   **Multi-GPU Handling:**
    -   **Warning:** Do NOT use `worker_init_fn` for seeding `IterableDataset`.
    -   Seeding logic resides entirely within `MetaIterableDataset.__iter__` to correctly capture distributed rank and worker info at runtime.

## 2. PyTorch Datasets

### 2.1 MetaIterableDataset (Training)
> Optimized for "Infinite" Data. Used for Training.

-   **Type:** `torch.utils.data.IterableDataset`
-   **Behavior:**
    -   Streams a potentially infinite sequence of tasks (SCMs).
    -   No fixed length (`__len__` is optional/virtual).
    -   **Yields:** a pickle-safe dict `{"seed": int, "x": Tensor, "adj": Tensor}`.
-   **Parallelism & Seeding:**
    -   **Crucial:** Implements rank-aware seeding inside `__iter__`.
    -   On `__iter__` (called inside the worker process):
        1.  Detects global rank (Multi-GPU).
        2.  Detects worker ID.
        3.  Derives a unique stream seed: `base_seed + (rank * num_workers) + worker_id`.
    -   Guarantees that every GPU and every worker produces unique SCMs (maximizing throughput).

### 2.2 MetaFixedDataset (Validation/Testing)
> Optimized for Reproducibility. Used for Val/Test.

-   **Type:** `torch.utils.data.Dataset` (Map-style)
-   **Behavior:**
    -   Backed by a fixed list of pre-generated seeds.
    -   `__getitem__(idx)` returns a dict `{"seed": int, "x": Tensor, "adj": Tensor}`.

### 2.4 Full-SCM / Mechanism-Supervised Training (Design Note)
To support arbitrary functional mechanisms safely on clusters (DataLoader workers), dataset items intentionally do **not** include `SCMInstance` or mechanism modules (which may be non-picklable).

Instead, datasets always return `seed`, so training code can reconstruct the teacher SCM on demand via `SCMFamily.sample_task(seed)` and apply functional supervision/distillation against the ground-truth mechanisms.

### 2.3 SCMInstance (Interface)
> The realized task object passed between components.

-   **Attributes:**
    -   `adjacency_matrix`: `torch.Tensor` (The DAG structure).
    -   `mechanisms`: List of Callables/Modules.
    -   `topological_order`: List[int] (Cached calculation for fast sampling).
-   **Methods:**
    -   `sample(num_samples: int) -> torch.Tensor`:
        -   Performs ancestral sampling in topological order.
        -   **Normalization:** Handled in the `collate_fn` (not here) to align with standard PyTorch conventions and allow batch-level statistics if needed.
    -   `visualize()`: Helper for plotting graph and mechanisms.

## 3. SCMFamily (Composition)
> A wrapper that defines a distribution of SCMs. Uses **Composition** instead of Inheritance.

-   **Structure:**
    -   Holds a `GraphGenerator` strategy.
    -   Holds a `MechanismFactory` strategy.
    -   **Attributes:** `n_nodes`, `sparsity_param`, `mechanism_proportions`, `noise_type`.
-   **Methods:**
    -   `sample_task(seed: int) -> SCMInstance`:
        1.  Seeds local RNG.
        2.  Calls `GraphGenerator` -> `adjacency_matrix`.
        3.  Calls `MechanismFactory` -> List of `nn.Module` mechanisms.
        4.  Returns `SCMInstance`.
    -   `plot_example()`: Visualizes a random sample from this family.

## 4. Generator Strategies
> Functional strategies for creating graphs and mechanisms. Should be vectorized where possible.

### 4.1 GraphGenerator (Protocol)
-   **Input:** `n_nodes`, parameters (density, etc.), `seed`.
-   **Output:** `adjacency_matrix` (Topologically sorted or acyclic by construction).
-   **Implementation variants:**
    -   **Erdős-Rényi (ER):** Vectorized sampling of upper-triangular matrix.
    -   **Scale-Free (SF):** Barabási-Albert algorithm.
    -   **Stochastic Block Model (SBM):** Block-diagonal masks.

### 4.2 MechanismFactory (Protocol)
-   **Input:** `adjacency_matrix`, `mechanism_proportions` (e.g., `{"linear": 0.5, "mlp": 0.5}`).
-   **Output:** List of Callables/Modules.
-   **Variants:**
    -   **Linear Heteroskedastic:** $X_j = W_j X_{pa(j)} + \epsilon_j$. Weights sampled from normal; noise variance sampled from Gamma.
    -   **MLP:** 2-layer NN with Leaky ReLU. Noise $\epsilon$ as input (non-additive).
    -   **GPCDE:** Gaussian Process with Latent Variable (complex, non-additive).

# Implementation Notes
-   **Vectorization & Graph Generation:**
    -   **Erdős-Rényi (ER):** Efficiently vectorized using `torch.bernoulli` on masked tensors.
    -   **Scale-Free (SF) / SBM:** Inherently requires sequential attachment (Barabási-Albert). Use NumPy/NetworkX on CPU for generation, then convert to Tensor. The overhead is negligible as generation is infrequent relative to sampling.

## 5. Analysis & Diagnostics
> A helper module for descriptive analysis, debugging, and experiment logging.

-   **Capabilities:**
    -   **Structural Inspection:** Samples a batch of graphs per family to plot degree histograms and visualize adjacency matrices.
    -   **Metric Logging:** Computes and logs chosen distance metrics (spectral/KL) between families.
    -   **Registry Output:** Writes seed lists, hash registries, and computed distances to disk for experiment tracking.
-   **Device Management:**
    -   Generators run on CPU (in `DataLoader` workers).
    -   Resulting `Tensor` data is pinned and transferred to GPU in the training loop.
