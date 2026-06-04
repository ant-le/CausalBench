from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from functools import partial
from typing import Any, Dict, Optional, Sequence, Set

import torch
from torch.utils.data import DataLoader

from causal_meta.datasets.generators.configs import (
    DataModuleConfig,
    FamilyConfig,
    RealWorldFamilyConfig,
)
from causal_meta.datasets.generators.factory import load_data_module_config
from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.torch_datasets import (
    MetaFixedDataset,
    MetaInterventionalDataset,
    MetaIterableDataset,
    RealWorldDataset,
)
from causal_meta.datasets.utils import (
    collate_fn_interventional,
    collate_fn_scm,
    compute_graph_hash,
)
from causal_meta.datasets.utils.sampling import NoPaddingDistributedSampler
from causal_meta.datasets.utils.stats import compute_family_distance

log = logging.getLogger(__name__)


class CausalMetaModule:
    """Dataset orchestrator that enforces disjoint train/val/test families."""

    @classmethod
    def from_config(cls, cfg: Any) -> CausalMetaModule:
        """Build a module from a configuration object (Dict, Hydra, or DataModuleConfig)."""
        data_module_config = load_data_module_config(cfg)
        return cls(data_module_config)

    def __init__(self, config: DataModuleConfig) -> None:
        """Initialize the module with a DataModuleConfig object."""
        self.config = config
        self._train_batch_size = int(getattr(config, "batch_size_train", 1))

        self.train_family: Optional[SCMFamily] = None
        self.train_families_by_n_nodes: Dict[int, SCMFamily] = {}
        self.val_families: Dict[str, SCMFamily] = {}
        self.test_families: Dict[str, SCMFamily] = {}

        self.train_dataset: Optional[MetaIterableDataset] = None
        self.val_datasets: Dict[str, MetaFixedDataset] = {}
        self.test_datasets: Dict[str, MetaFixedDataset] = {}
        self.test_interventional_datasets: Dict[str, MetaInterventionalDataset] = {}
        self._val_loaders: Optional[Dict[str, DataLoader]] = None
        self._test_loaders: Optional[Dict[str, DataLoader]] = None
        self._test_interventional_loaders: Optional[Dict[str, DataLoader]] = None

        self.family_distances: Dict[str, Dict[str, float]] = {}

    def setup(self, *, train_batch_size_override: int | None = None) -> None:
        """Instantiate datasets, enforce disjointness, and pre-compute stats."""
        resolved_train_batch_size = (
            int(train_batch_size_override)
            if train_batch_size_override is not None
            else int(getattr(self.config, "batch_size_train", 1))
        )
        if resolved_train_batch_size < 1:
            raise ValueError("data.batch_size_train must be >= 1")
        self._train_batch_size = resolved_train_batch_size

        self.train_family = self._build_family(self.config.train_family)
        configured_train_nodes = [
            int(n) for n in getattr(self.config, "train_n_nodes", []) if int(n) > 0
        ]
        if not configured_train_nodes:
            configured_train_nodes = [int(self.config.train_family.n_nodes)]
        configured_train_nodes = sorted(set(configured_train_nodes))
        self.train_families_by_n_nodes = {
            n_nodes: self._build_family(
                replace(self.config.train_family, n_nodes=int(n_nodes))
            )
            for n_nodes in configured_train_nodes
        }

        val_family_cfgs = self.config.val_families or {
            self.config.train_family.name: self.config.train_family
        }
        # Re-key by cfg.name so downstream lookups are consistent.
        self._resolved_val_cfgs: Dict[str, FamilyConfig] = {
            cfg.name: cfg for cfg in val_family_cfgs.values()
        }
        self.val_families = {
            cfg.name: self._build_family(cfg) for cfg in val_family_cfgs.values()
        }

        # ----- Separate generative vs. real-world test families -----
        self._resolved_test_cfgs: Dict[str, FamilyConfig] = {}
        self._resolved_rw_test_cfgs: Dict[str, RealWorldFamilyConfig] = {}
        for cfg in self.config.test_families.values():
            if isinstance(cfg, RealWorldFamilyConfig):
                self._resolved_rw_test_cfgs[cfg.name] = cfg
            else:
                self._resolved_test_cfgs[cfg.name] = cfg

        # Build generative SCM families.
        self.test_families = {
            cfg.name: self._build_family(cfg)
            for cfg in self._resolved_test_cfgs.values()
        }

        # Collect hashes from validation and test sets for disjointness check
        all_val_hashes: Set[str] = set()
        if self.config.seeds_val:
            for family in self.val_families.values():
                all_val_hashes.update(
                    self._sample_hashes(family, self.config.seeds_val)
                )

        all_test_hashes: Set[str] = set()
        for family in self.test_families.values():
            all_test_hashes.update(self._sample_hashes(family, self.config.seeds_test))

        all_reserved_hashes = all_val_hashes | all_test_hashes

        if self.config.safety_checks:
            val_test_collisions = all_val_hashes & all_test_hashes
            if val_test_collisions:
                warnings.warn(
                    f"Detected graph hash collisions between validation and test splits: {val_test_collisions}. "
                    "Consider adjusting seeds or family parameters.",
                    RuntimeWarning,
                )

            # Compute distributional distances for each test family
            self.family_distances = self._compute_all_distances()
        else:
            self.family_distances = {}

        self.train_dataset = MetaIterableDataset(
            self.train_family,
            base_seed=self.config.base_seed,
            samples_per_task=self.config.samples_per_task,
            forbidden_hashes=all_reserved_hashes,
            hash_mechanisms=getattr(self.config, "hash_mechanisms", False),
            families_by_n_nodes=(
                self.train_families_by_n_nodes
                if len(self.train_families_by_n_nodes) > 1
                else None
            ),
            batch_size_hint=resolved_train_batch_size,
            samples_per_task_obs=getattr(self.config, "samples_per_task_obs", None),
            samples_per_task_int=int(getattr(self.config, "samples_per_task_int", 0)),
            use_interventional_training=bool(
                getattr(self.config, "use_interventional_training", False)
            ),
            train_p_obs_only=float(getattr(self.config, "train_p_obs_only", 0.0)),
            intervention_value=float(getattr(self.config, "intervention_value", 0.0)),
        )

        self.val_datasets = {
            name: MetaFixedDataset(
                family,
                seeds=self.config.seeds_val,
                samples_per_task=self._family_samples_per_task(
                    self._resolved_val_cfgs[name]
                ),
            )
            for name, family in self.val_families.items()
        }

        # ----- Build test datasets: generative families -----
        self.test_datasets = {
            name: MetaFixedDataset(
                family,
                seeds=self.config.seeds_test,
                samples_per_task=self._family_samples_per_task(
                    self._resolved_test_cfgs[name]
                ),
            )
            for name, family in self.test_families.items()
        }

        # ----- Build test datasets: real-world families -----
        for rw_name, rw_cfg in self._resolved_rw_test_cfgs.items():
            rw_dataset = self._build_real_world_dataset(rw_cfg)
            self.test_datasets[rw_name] = rw_dataset
            log.info(
                "Loaded real-world test family '%s': %d variables, %d observations",
                rw_name,
                rw_dataset._n_nodes,
                rw_dataset._data.shape[0],
            )

        self.test_interventional_datasets = {
            name: MetaInterventionalDataset(
                family,
                seeds=self.config.seeds_test,
                intervention_value=0.0,
                samples_per_task=self._family_samples_per_task(
                    self._resolved_test_cfgs[name]
                ),
            )
            for name, family in self.test_families.items()
        }
        self._val_loaders = None
        self._test_loaders = None
        self._test_interventional_loaders = None

    def train_dataloader(self, *, batch_size_override: int | None = None) -> DataLoader:
        """Return the training DataLoader with GPU optimizations."""
        batch_size = (
            int(batch_size_override)
            if batch_size_override is not None
            else int(getattr(self.config, "batch_size_train", 1))
        )
        if batch_size < 1:
            raise ValueError("data.batch_size_train must be >= 1")

        if self.train_dataset is None or batch_size != self._train_batch_size:
            self.setup(train_batch_size_override=batch_size)
        if self.train_dataset is None:
            raise RuntimeError("Training dataset was not initialized.")

        collate_fn = partial(collate_fn_scm, normalize=self.config.normalize_data)
        train_dataset = self.train_dataset

        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
            **self._dataloader_perf_kwargs(),
        )

    def test_dataloader(self) -> Dict[str, DataLoader]:
        """Return a dictionary of test DataLoaders.

        Batch size is always 1 to ensure per-task metric granularity.
        """
        if self._test_loaders is not None:
            return self._test_loaders

        if not self.test_datasets:
            self.setup()

        loaders = {}

        collate_fn = partial(collate_fn_scm, normalize=self.config.normalize_data)

        configured_bs = int(getattr(self.config, "batch_size_test", 1))
        if configured_bs != 1:
            log.warning(
                "data.batch_size_test=%d overridden to 1 for per-task metric "
                "granularity.",
                configured_bs,
            )
        batch_size = 1

        for name, dataset in self.test_datasets.items():
            loaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.config.num_workers,  # Use configured workers
                pin_memory=self.config.pin_memory,
                collate_fn=collate_fn,
                sampler=self._evaluation_sampler(dataset),
                **self._dataloader_perf_kwargs(for_evaluation=True),
            )
        self._test_loaders = loaders
        return self._test_loaders

    def test_interventional_dataloader(self) -> Dict[str, DataLoader]:
        """Return a dictionary of interventional test DataLoaders.

        Batch size is always 1 to ensure per-task metric granularity.
        """
        if self._test_interventional_loaders is not None:
            return self._test_interventional_loaders

        if not self.test_interventional_datasets:
            self.setup()

        loaders = {}

        collate_fn = partial(
            collate_fn_interventional, normalize=self.config.normalize_data
        )

        configured_bs = int(getattr(self.config, "batch_size_test_interventional", 1))
        if configured_bs != 1:
            log.warning(
                "data.batch_size_test_interventional=%d overridden to 1 for "
                "per-task metric granularity.",
                configured_bs,
            )
        batch_size = 1

        for name, dataset in self.test_interventional_datasets.items():
            loaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                collate_fn=collate_fn,
                sampler=self._evaluation_sampler(dataset),
                **self._dataloader_perf_kwargs(for_evaluation=True),
            )
        self._test_interventional_loaders = loaders
        return self._test_interventional_loaders

    def val_dataloader(self) -> Dict[str, DataLoader]:
        """Return a dictionary of validation DataLoaders.

        Batch size is always 1 to ensure per-task metric granularity.
        """
        if self._val_loaders is not None:
            return self._val_loaders

        if not self.val_datasets:
            self.setup()

        if not self.config.seeds_val:
            raise ValueError(
                "Validation split is not configured. Provide non-empty 'seeds_val' in the data config."
            )

        loaders = {}

        collate_fn = partial(collate_fn_scm, normalize=self.config.normalize_data)

        configured_bs = int(getattr(self.config, "batch_size_val", 1))
        if configured_bs != 1:
            log.warning(
                "data.batch_size_val=%d overridden to 1 for per-task metric "
                "granularity.",
                configured_bs,
            )
        batch_size = 1

        for name, dataset in self.val_datasets.items():
            loaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.config.num_workers,  # Use configured workers
                pin_memory=self.config.pin_memory,
                collate_fn=collate_fn,
                sampler=self._evaluation_sampler(dataset),
                **self._dataloader_perf_kwargs(for_evaluation=True),
            )
        self._val_loaders = loaders
        return self._val_loaders

    def _dataloader_perf_kwargs(
        self, *, for_evaluation: bool = False
    ) -> Dict[str, Any]:
        """Return optional DataLoader kwargs that improve host-side throughput."""
        num_workers = int(self.config.num_workers)
        if num_workers <= 0:
            return {}

        prefetch_factor = int(getattr(self.config, "prefetch_factor", 2))
        if prefetch_factor < 1:
            raise ValueError("data.prefetch_factor must be >= 1 when num_workers > 0")

        persistent_workers = bool(getattr(self.config, "persistent_workers", True))
        if for_evaluation:
            persistent_workers = False

        return {
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
        }

    def _evaluation_sampler(self, dataset: Any) -> Any:
        """Build the distributed sampler used for validation and test datasets."""
        is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if not is_distributed:
            return None
        return NoPaddingDistributedSampler(dataset, shuffle=False)

    def _sample_hashes(self, family: SCMFamily, seeds: Sequence[int]) -> Set[str]:
        """
        Compute a set of graph hashes for a given list of seeds to ensure disjointness.

        If ``config.hash_mechanisms`` is True, mechanism parameters are included in the
        hash, enabling functional generalization testing on identical DAG structures.
        """
        hashes: Set[str] = set()
        # Use provided seeds or probe a default range if empty
        seeds_to_probe = list(seeds) if seeds else self._probe_seeds(seeds)

        include_mechanisms = getattr(self.config, "hash_mechanisms", False)

        for seed in seeds_to_probe:
            if include_mechanisms:
                # sample_task returns the full SCMInstance with mechanisms
                instance = family.sample_task(seed)
                hashes.add(
                    compute_graph_hash(
                        instance.adjacency_matrix,
                        mechanisms=instance.mechanisms,
                        include_mechanisms=True,
                    )
                )
            else:
                # Fast path: only hash graph structure
                adjacency_matrix = family.sample_graph(seed)
                hashes.add(compute_graph_hash(adjacency_matrix))
        return hashes

    def _probe_seeds(self, seeds: Sequence[int], count: int = 3) -> Sequence[int]:
        if seeds:
            return list(seeds)[:count]
        return [self.config.base_seed + i for i in range(count)]

    def _compute_all_distances(self) -> Dict[str, Dict[str, float]]:
        """Pre-compute spectral, KL degree, and mechanism distances for each test family.

        Returns:
            Mapping of ``dataset_key -> {"spectral": float, "kl_degree": float,
            "mechanism": float}``.  Falls back to ``NaN`` for any metric that
            fails to compute.
        """
        if self.train_family is None:
            return {}

        distances: Dict[str, Dict[str, float]] = {}
        for name, test_family in self.test_families.items():
            entry: Dict[str, float] = {}
            for metric_name, metric_key, n_samples in [
                ("spectral", "spectral", 25),
                ("kl", "kl_degree", 25),
                ("mechanism", "mechanism", 12),
            ]:
                try:
                    entry[metric_key] = compute_family_distance(
                        self.train_family,
                        test_family,
                        metric=metric_name,
                        n_samples=n_samples,
                    )
                except Exception:
                    log.warning(
                        "%s distance failed for family %s",
                        metric_key,
                        name,
                        exc_info=True,
                    )
                    entry[metric_key] = float("nan")
            distances[name] = entry
        return distances

    def _build_family(self, cfg: FamilyConfig) -> SCMFamily:
        graph_generator = cfg.graph_cfg.instantiate()
        mechanism_factory = cfg.mech_cfg.instantiate()
        return SCMFamily(
            name=cfg.name,
            n_nodes=cfg.n_nodes,
            graph_generator=graph_generator,
            mechanism_factory=mechanism_factory,
            noise_type=cfg.noise_type,
        )

    def _build_real_world_dataset(self, cfg: RealWorldFamilyConfig) -> RealWorldDataset:
        """Load a real-world dataset by dispatching on ``cfg.loader``."""
        from causal_meta.datasets.real_world import load_real_world_dataset

        data, adjacency = load_real_world_dataset(
            cfg.loader,
            **(cfg.loader_kwargs or {}),
        )
        samples_per_task = (
            int(cfg.samples_per_task)
            if cfg.samples_per_task is not None
            else int(self.config.samples_per_task)
        )
        return RealWorldDataset(
            name=cfg.name,
            data=data,
            adjacency=adjacency,
            seeds=self.config.seeds_test,
            samples_per_task=samples_per_task,
        )

    def _family_samples_per_task(self, cfg: FamilyConfig) -> int:
        if cfg.samples_per_task is not None:
            return int(cfg.samples_per_task)
        return int(self.config.samples_per_task)
