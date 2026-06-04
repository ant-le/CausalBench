# Model Architecture & Design

## Goals
1.  **Modular Storage**: Group each model in a separate subdirectory.
    *   `./avici/`: Wrapper for Avici-like models (Amortized Inference).
    *   `./bcnp/`: Bayesian Causal Neural Process models.
    *   `./bayesDag/`: Custom baselines.
2.  **Unified Interface**: All models must inherit from a common `BaseModel`.
3.  **Meta vs. Classical**: 
    *   Distinguish between models requiring pre-training (Amortized/Deep Learning) and those requiring instance-specific optimization (MCMC/VI).

## Class Structure

### `BaseModel(nn.Module)`
The abstract base class located in `src/causal_meta/models/base.py`.

**Attributes:**
*   `needs_pretraining` (bool): 
    *   `True`: Model is a meta-learner (e.g., Avici, BCNP) and requires a training phase across many tasks.
    *   `False`: Model adapts per instance (e.g., MCMC, VI) and "training" happens during the inference/evaluation phase.

**Key Methods:**
*   `forward(x: torch.Tensor) -> Any`:
    *   Input: `x` of shape `(Batch, Samples, Variables)`.
    *   Output: Model-specific distribution parameters (e.g., edge logits).
*   `sample(x: torch.Tensor, num_samples: int = 1) -> torch.Tensor`:
    *   Input: `x` of shape `(Batch, Samples, Variables)`.
    *   Output: Predicted adjacency matrices `(Batch, num_samples, V, V)`.
    *   *Behavior*: 
        *   If `needs_pretraining=True`: Fast forward pass + sampling from posterior.
        *   If `needs_pretraining=False`: Triggers the optimization/sampling loop (MCMC/VI) for the specific input `x`.

**Training Objective Note:**
`BaseModel` defines `calculate_loss(output, target, **kwargs)` which must be implemented by subclasses. This moves the responsibility of defining the training objective (e.g., NLL, BCE, Acyclicity constraints) to the model itself, allowing for model-specific loss formulations (like DiBS/Avici cyclicity regularization) without bloating the generic runner.

## Directory Structure
```
src/causal_meta/models/
├── base.py          # Abstract base class
├── factory.py       # Config -> Model instantiation
├── avici/           # Avici wrapper
├── bcnp/            # BCNP implementation
└── utils/           # Common neural components
```
