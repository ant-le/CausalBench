# Research Workflow: Auditing Meta-Learner Robustness

## 1. Training Set: "The Competence Curriculum" (BCNP Replication)
[cite_start]**Goal:** Exact replication of the BCNP paper's "All Data" training regime[cite: 296]. The model is trained on a mixture of standard causal mechanisms and graph topologies to learn the fundamental "grammar" of causality (screening off, conditional independence).

### A. Graph Topology Generators
[cite_start]*The paper trains on a mixture of Erdős-Rényi and Scale-Free graphs to cover both random and hub-based structures[cite: 583].*

* **Erdős-Rényi (ER):**
    * **Settings:** Edges sampled with probability $p$. [cite_start]Densities vary to produce expected edge counts of 20, 40, and 60[cite: 498].
    * **Role:** Represents standard, sparse random graphs.
* **Scale-Free (SF):**
    * [cite_start]**Settings:** Generated using the Barabási-Albert model[cite: 583].
    * [cite_start]**Role:** Represents graphs with "hubs" (highly connected nodes), crucial for learning V-structures and super-nodes found in biological networks (like the Syntren dataset)[cite: 549].

### B. Function Generators (Mechanisms)
[cite_start]*The paper trains on a balanced mixture of the following three mechanism classes[cite: 288]. [cite_start]Note: For the Syntren experiment, the paper explicitly mentions mixing NeuralNet and GPCDE with equal probability [cite: 584][cite_start], while the "All Data" model includes Linear as well[cite: 296].*

1.  [cite_start]**Linear-Heteroscedastic (~33%)** [cite: 288, 499]
    * **Equation:** $X_j \sim \mathcal{N}(W^T PA_j, \sigma_i^2)$
    * **Details:** Weights $W \sim \mathcal{N}(0, 10)$. [cite_start]Crucially, the noise variance $\sigma_i$ is sampled from a Gamma distribution for each observation, creating **heteroscedastic noise** (variance changes per sample)[cite: 501, 502].
    * **Training Signal:** Teaches the model to handle varying noise levels and linear dependencies.

2.  [cite_start]**Random MLP (NeuralNet) (~33%)** [cite: 288, 509]
    * **Equation:** $X_j = \text{MLP}(PA_j, \epsilon)$
    * [cite_start]**Details:** 2-layer Neural Network with **Leaky ReLU** activation and width 32[cite: 514]. [cite_start]The noise term $\epsilon$ is fed as an input to the network (non-additive interaction)[cite: 511].
    * **Training Signal:** Teaches monotonic, standard non-linear relationships and latent variable interactions.

3.  [cite_start]**GPCDE (Gaussian Process with Latent Variable) (~33%)** [cite: 288, 515]
    * **Equation:** $X_j \sim \mathcal{GP}(PA_j, \epsilon)$
    * [cite_start]**Details:** Uses an **Exponential Gamma Kernel**[cite: 516]. [cite_start]Crucially, a latent noise variable $\epsilon$ is included in the kernel input[cite: 523], making the noise non-additive and complex. [cite_start]Length scales are sampled to vary smoothness[cite: 524].
    * **Training Signal:** Teaches the model to generalize to arbitrary smooth functions and non-additive noise structures.

---

## 2. Test Set: "The Robustness Audit" (O.O.D. Extensions)
**Goal:** Assess robustness against functional, complexity, and structural shifts that strictly violate the priors learned during training.

### A. Graph Structure O.O.D.
* **Stochastic Block Model (SBM) (New Extension)**
    * **Definition:** Graphs with distinct communities (clusters) where $P(\text{edge within}) \gg P(\text{edge between})$.
    * **O.O.D. Argument:** The training set (ER/Scale-Free) assumes relatively homogeneous or hub-based connectivity. SBM introduces **modularity**. Transformers often struggle with "long-range" dependencies if they over-fit to local patterns within the training graphs. This tests if the attention mechanism can resolve causal links across loosely connected clusters.

### B. Functional O.O.D. (The "Invertibility Gap")
* **Case 1: Square (Symmetric)**
    * **Equation:** $X_j = (W^T PA_j)^2 + 0.1\epsilon$
    * **O.O.D. Argument:** The training functions (Linear, Leaky ReLU, GPCDE) are largely **monotonic** or locally smooth, preserving the ordering of inputs. The Square function is **non-injective** (two different $X$ values map to the same $Y$). This tests if the model has learned "causality" or just "invertibility heuristics."
* **Case 2: Periodic (High Frequency)**
    * **Equation:** $X_j = \sin(4\pi W^T PA_j) + 0.1\epsilon$
    * **O.O.D. Argument:** High-frequency periodicity violates the **smoothness priors** (length scales) learned from the GPCDE kernel during training. It tests if the model can identify directionality when the functional mapping is non-monotonic and repetitive.

### C. Complexity O.O.D. (The "Chaos" Test)
* **Case: Logistic Map**
    * **Equation:** $X_{j} = 4 \cdot \sigma(PA_j) \cdot (1 - \sigma(PA_j))$ (where $\sigma$ is sigmoid).
    * **O.O.D. Argument:** The training data (GPs/MLPs) is fundamentally **stochastic** (driven by noise distributions). The Logistic Map is **deterministic chaos**—it has zero correlation and statistically resembles white noise, yet is purely causal. This tests if the model can distinguish "algorithmic complexity" from "random noise," a distinction never required during standard training.

### D. Statistical O.O.D. (Post-Nonlinear Shift)
* **Case: Post-Nonlinear (PNL)**
    * **Equation:** $X_j = g(f(PA_j) + \epsilon)$, where $g(z) = z^3$ or $g(z) = \text{sigmoid}(z)$.
    * **O.O.D. Argument:** The training set's GPCDE models general noise $f(x, \epsilon)$, but typically assumes the observation is a realization of the process. PNL introduces a specific **measurement distortion** $g(\cdot)$ after the noise is added. This simulates sensor saturation or exponential scaling, a structural assumption (PNL) distinct from the Latent Variable (GPCDE) assumption.

---

## 3. Processing Protocol
* **Normalization:** following Reisach et al. (2021) [cite_start][cite: 408, 495], all variables in both Training and Test sets are normalized to zero mean and unit variance to prevent the model from sorting variables by variance (marginal variance sorting) to infer direction.
