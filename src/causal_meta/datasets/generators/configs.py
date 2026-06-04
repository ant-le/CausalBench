from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from causal_meta.datasets.generators.graphs import (
    ErdosRenyiGenerator,
    GeometricRandomGenerator,
    MixtureGraphGenerator,
    SBMGenerator,
    ScaleFreeGenerator,
    WattsStrogatzGenerator,
)
from causal_meta.datasets.generators.mechanisms import (
    GPMechanismFactory,
    LinearMechanismFactory,
    LogisticMapMechanismFactory,
    MixtureMechanismFactory,
    MLPMechanismFactory,
    PeriodicMechanismFactory,
    PNLMechanismFactory,
    SquareMechanismFactory,
)

from causal_meta.datasets.generators._protocols import Instantiable


@dataclass
class ErdosRenyiConfig:
    edge_prob: Optional[float] = None
    sparsity: Optional[float] = None

    def instantiate(self) -> ErdosRenyiGenerator:
        return ErdosRenyiGenerator(edge_prob=self.edge_prob, sparsity=self.sparsity)


@dataclass
class ScaleFreeConfig:
    m: int = 2

    def instantiate(self) -> ScaleFreeGenerator:
        return ScaleFreeGenerator(m=self.m)


@dataclass
class SBMConfig:
    n_blocks: int
    p_intra: float
    p_inter: float

    def instantiate(self) -> SBMGenerator:
        return SBMGenerator(
            n_blocks=self.n_blocks, p_intra=self.p_intra, p_inter=self.p_inter
        )


@dataclass
class WattsStrogatzConfig:
    k: int = 4
    p: float = 0.3

    def instantiate(self) -> WattsStrogatzGenerator:
        return WattsStrogatzGenerator(k=self.k, p=self.p)


@dataclass
class GeometricRandomConfig:
    radius: float = 0.3
    dim: int = 2

    def instantiate(self) -> GeometricRandomGenerator:
        return GeometricRandomGenerator(radius=self.radius, dim=self.dim)


@dataclass
class MixtureGraphConfig:
    generators: List[Instantiable]
    weights: List[float]

    def instantiate(self) -> MixtureGraphGenerator:
        return MixtureGraphGenerator(
            generators=[g.instantiate() for g in self.generators], weights=self.weights
        )


@dataclass
class LinearMechanismConfig:
    weight_scale: float = 1.0
    noise_concentration: float = 2.0
    noise_rate: float = 2.0

    def instantiate(self) -> LinearMechanismFactory:
        return LinearMechanismFactory(
            weight_scale=self.weight_scale,
            noise_concentration=self.noise_concentration,
            noise_rate=self.noise_rate,
        )


@dataclass
class MLPMechanismConfig:
    hidden_dim: int = 32

    def instantiate(self) -> MLPMechanismFactory:
        return MLPMechanismFactory(hidden_dim=self.hidden_dim)


@dataclass
class MixtureMechanismConfig:
    factories: List[Instantiable]
    weights: List[float]

    def instantiate(self) -> MixtureMechanismFactory:
        return MixtureMechanismFactory(
            factories=[f.instantiate() for f in self.factories], weights=self.weights
        )


@dataclass
class SquareMechanismConfig:
    weight_scale: float = 1.0
    noise_scale: float = 0.1

    def instantiate(self) -> SquareMechanismFactory:
        return SquareMechanismFactory(
            weight_scale=self.weight_scale, noise_scale=self.noise_scale
        )


@dataclass
class PeriodicMechanismConfig:
    weight_scale: float = 1.0
    noise_scale: float = 0.1

    def instantiate(self) -> PeriodicMechanismFactory:
        return PeriodicMechanismFactory(
            weight_scale=self.weight_scale, noise_scale=self.noise_scale
        )


@dataclass
class LogisticMapMechanismConfig:
    weight_scale: float = 1.0
    noise_scale: float = 1.0

    def instantiate(self) -> LogisticMapMechanismFactory:
        return LogisticMapMechanismFactory(
            weight_scale=self.weight_scale,
            noise_scale=self.noise_scale,
        )


@dataclass
class GPMechanismConfig:
    mode: str = "approximate"
    rff_dim: int = 512
    num_kernels: int = 4
    length_scale_range: tuple[float, float] = (0.1, 10.0)
    variance: Optional[float] = None
    variance_range: Optional[tuple[float, float]] = None
    exact_num_kernel_pairs: int = 2
    alpha_range: tuple[float, float] = (0.1, 100.0)
    gamma_range: tuple[float, float] = (1e-5, 0.99999)
    exact_noise_concentration: float = 1.0
    exact_noise_rate: float = 10.0
    exact_jitter: float = 1e-4

    def instantiate(self) -> GPMechanismFactory:
        return GPMechanismFactory(
            mode=self.mode,
            rff_dim=self.rff_dim,
            num_kernels=self.num_kernels,
            length_scale_range=self.length_scale_range,
            variance=self.variance,
            variance_range=self.variance_range,
            exact_num_kernel_pairs=self.exact_num_kernel_pairs,
            alpha_range=self.alpha_range,
            gamma_range=self.gamma_range,
            exact_noise_concentration=self.exact_noise_concentration,
            exact_noise_rate=self.exact_noise_rate,
            exact_jitter=self.exact_jitter,
        )


@dataclass
class PNLMechanismConfig:
    inner_config: Optional[Instantiable] = None
    nonlinearity_type: str = "cube"

    def instantiate(self) -> PNLMechanismFactory:
        inner_factory = self.inner_config.instantiate() if self.inner_config else None
        return PNLMechanismFactory(
            inner_factory=inner_factory, nonlinearity_type=self.nonlinearity_type
        )


# Unions for Type Hinting
GraphConfig = Union[
    ErdosRenyiConfig,
    ScaleFreeConfig,
    SBMConfig,
    WattsStrogatzConfig,
    GeometricRandomConfig,
    MixtureGraphConfig,
    Instantiable,
]
MechanismConfig = Union[
    LinearMechanismConfig,
    MLPMechanismConfig,
    MixtureMechanismConfig,
    SquareMechanismConfig,
    PeriodicMechanismConfig,
    LogisticMapMechanismConfig,
    GPMechanismConfig,
    PNLMechanismConfig,
    Instantiable,
]


@dataclass
class FamilyConfig:
    """Configuration for a single SCM Family (graph distribution + mechanisms)."""

    name: str
    n_nodes: int
    graph_cfg: GraphConfig
    mech_cfg: MechanismConfig
    samples_per_task: Optional[int] = None
    noise_type: str = "gaussian"
    inference_n_samples: Optional[int] = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string.")
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be positive.")
        if self.samples_per_task is not None and self.samples_per_task < 1:
            raise ValueError("samples_per_task must be positive when provided.")
        # Import here to avoid circular imports at module level.
        from causal_meta.datasets.noise import SUPPORTED_NOISE_TYPES

        if self.noise_type not in SUPPORTED_NOISE_TYPES:
            raise ValueError(
                f"Unknown noise_type '{self.noise_type}'. "
                f"Supported: {SUPPORTED_NOISE_TYPES}"
            )


@dataclass
class RealWorldFamilyConfig:
    """Configuration for a real-world dataset with a fixed ground-truth DAG.

    Unlike :class:`FamilyConfig`, this does not specify graph or mechanism
    generators.  Instead it points to a *loader* — a callable string
    identifier (e.g. ``"sachs"``, ``"syntren"``) — that returns the
    observational data matrix and adjacency matrix.
    """

    name: str
    loader: str
    n_nodes: int
    samples_per_task: Optional[int] = None
    loader_kwargs: Optional[Dict[str, str]] = None
    inference_n_samples: Optional[int] = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string.")
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be positive.")
        if self.samples_per_task is not None and self.samples_per_task < 1:
            raise ValueError("samples_per_task must be positive when provided.")


# Union type for test/val families that may be either generative or real-world.
AnyFamilyConfig = Union[FamilyConfig, RealWorldFamilyConfig]


@dataclass
class DataModuleConfig:
    """Top-level configuration for the CausalMetaModule."""

    train_family: FamilyConfig

    # Support multiple named test families (e.g., {"ood_graph": ..., "ood_mech": ...})
    test_families: Dict[str, AnyFamilyConfig]

    seeds_test: List[int]

    # Fixed validation seeds (distinct from train/test)
    seeds_val: List[int]

    # Optional validation families (default is {"id": train_family} when empty)
    val_families: Dict[str, FamilyConfig] = field(default_factory=dict)

    base_seed: int = 0

    samples_per_task: int = 128

    # Number of posterior graph samples drawn at inference time.
    # Individual families can override this via their own inference_n_samples.
    inference_n_samples: int = 100

    samples_per_task_obs: Optional[int] = None

    samples_per_task_int: int = 0

    use_interventional_training: bool = False

    train_p_obs_only: float = 0.0

    intervention_value: float = 0.0

    train_n_nodes: List[int] = field(default_factory=list)

    safety_checks: bool = True

    num_workers: int = 0

    pin_memory: bool = True

    persistent_workers: bool = True

    prefetch_factor: int = 2

    normalize_data: bool = True

    # Per-device dataloader batch caps.  The amortized trainer derives its
    # effective batch size and gradient accumulation from these caps together
    # with trainer.target_global_tasks_per_step.
    batch_size_train: int = 1
    batch_size_val: int = 1
    batch_size_test: int = 1
    batch_size_test_interventional: int = 1

    # If True, include mechanism parameters in graph hashes for disjointness checks.
    # This enables functional generalization testing where the same DAG structure
    # appears with different mechanism parameters in train vs. test.
    hash_mechanisms: bool = False
