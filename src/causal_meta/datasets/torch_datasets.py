from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.utils import compute_graph_hash

log = logging.getLogger(__name__)


def _distributed_context() -> Tuple[int, int]:
    """Return rank and world size if torch.distributed is initialized."""
    rank = 0
    world_size = 1
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
    except Exception:
        rank = 0
        world_size = 1
    return rank, world_size


class MetaIterableDataset(IterableDataset):
    """Infinite stream of SCM samples with rank/worker-aware seeding."""

    def __init__(
        self,
        family: Optional[SCMFamily],
        base_seed: int,
        samples_per_task: int = 128,
        forbidden_hashes: Optional[Iterable[str]] = None,
        hash_mechanisms: bool = False,
        families_by_n_nodes: Optional[Dict[int, SCMFamily]] = None,
        batch_size_hint: int = 1,
        samples_per_task_obs: Optional[int] = None,
        samples_per_task_int: int = 0,
        use_interventional_training: bool = False,
        train_p_obs_only: float = 0.0,
        intervention_value: float = 0.0,
    ) -> None:
        super().__init__()
        if family is None and not families_by_n_nodes:
            raise ValueError(
                "Either `family` or `families_by_n_nodes` must be provided."
            )

        self.family = family
        self.families_by_n_nodes = {
            int(k): v for k, v in (families_by_n_nodes or {}).items()
        }
        self.family_name = family.name if family is not None else "mixed_train_stream"
        self.base_seed = int(base_seed)
        self.samples_per_task = samples_per_task
        self.samples_per_task_obs = (
            int(samples_per_task_obs)
            if samples_per_task_obs is not None
            else int(samples_per_task)
        )
        self.samples_per_task_int = int(samples_per_task_int)
        self.use_interventional_training = bool(use_interventional_training)
        self.train_p_obs_only = float(train_p_obs_only)
        self.intervention_value = float(intervention_value)
        self.batch_size_hint = max(1, int(batch_size_hint))
        self.forbidden_hashes: Set[str] = set(forbidden_hashes or [])
        self.hash_mechanisms = hash_mechanisms

        if self.samples_per_task_obs < 1:
            raise ValueError("samples_per_task_obs must be >= 1")
        if self.samples_per_task_int < 0:
            raise ValueError("samples_per_task_int must be >= 0")
        if not (0.0 <= self.train_p_obs_only <= 1.0):
            raise ValueError("train_p_obs_only must be between 0 and 1")

        if self.families_by_n_nodes:
            self._n_nodes_schedule = sorted(self.families_by_n_nodes)
        elif self.family is not None:
            self._n_nodes_schedule = [int(self.family.n_nodes)]
        else:
            self._n_nodes_schedule = []

    def _family_for_n_nodes(self, n_nodes: int) -> SCMFamily:
        if self.families_by_n_nodes:
            family = self.families_by_n_nodes.get(int(n_nodes))
            if family is None:
                raise KeyError(f"No train family configured for n_nodes={n_nodes}.")
            return family
        if self.family is None:
            raise RuntimeError("MetaIterableDataset has no configured family.")
        return self.family

    def _sample_interventional_batch(
        self,
        instance: Any,
        *,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_obs = int(self.samples_per_task_obs)
        n_int = int(self.samples_per_task_int)

        use_obs_only = n_int <= 0
        if n_int > 0:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 31)
                use_obs_only = bool(
                    torch.rand((), dtype=torch.float32).item() < self.train_p_obs_only
                )

        if use_obs_only:
            total_obs = n_obs + n_int
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 41)
                data = instance.sample(total_obs)
            intervention_mask = torch.zeros_like(data)
            return data, intervention_mask

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 43)
            x_obs = instance.sample(n_obs)

        n_nodes = int(instance.n_nodes)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 47)
            int_targets = torch.randint(0, n_nodes, (n_int,), dtype=torch.long)

        x_int = torch.zeros(n_int, n_nodes, dtype=x_obs.dtype)
        int_mask = torch.zeros(n_int, n_nodes, dtype=x_obs.dtype)

        unique_targets = torch.unique(int_targets)
        for target in unique_targets.tolist():
            target_idx = int(target)
            rows = torch.nonzero(int_targets == target_idx, as_tuple=False).flatten()
            mutilated = instance.do({target_idx: self.intervention_value})
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 1000 + target_idx)
                sampled = mutilated.sample(int(rows.numel()))
            x_int[rows] = sampled
            int_mask[rows, target_idx] = 1.0

        data = torch.cat([x_obs, x_int], dim=0)
        intervention_mask = torch.cat([torch.zeros_like(x_obs), int_mask], dim=0)
        return data, intervention_mask

    def _sample_observational_batch(
        self,
        instance: Any,
        *,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            data = instance.sample(self.samples_per_task)
        intervention_mask = torch.zeros_like(data)
        return data, intervention_mask

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        rank, world_size = _distributed_context()
        stride = max(1, num_workers * world_size)

        seed = self.base_seed + rank * num_workers + worker_id
        local_index = 0
        while True:
            bucket = (local_index // self.batch_size_hint) % len(self._n_nodes_schedule)
            n_nodes = self._n_nodes_schedule[bucket]
            family = self._family_for_n_nodes(n_nodes)
            instance = family.sample_task(seed)

            if self.forbidden_hashes:
                graph_hash = compute_graph_hash(
                    instance.adjacency_matrix,
                    mechanisms=instance.mechanisms if self.hash_mechanisms else None,
                    include_mechanisms=self.hash_mechanisms,
                )
                if graph_hash in self.forbidden_hashes:
                    seed += stride
                    local_index += 1
                    continue

            if self.use_interventional_training:
                data, intervention_mask = self._sample_interventional_batch(
                    instance,
                    seed=seed,
                )
            else:
                data, intervention_mask = self._sample_observational_batch(
                    instance,
                    seed=seed,
                )

            yield {
                "seed": int(seed),
                "family_name": family.name,
                "data": data,
                "intervention_mask": intervention_mask,
                "adjacency": instance.adjacency_matrix,
                "n_nodes": int(instance.n_nodes),
                "samples_per_task": int(data.shape[0]),
            }
            seed += stride
            local_index += 1


class MetaFixedDataset(Dataset):
    """
    Fixed-seed SCM dataset for deterministic validation/testing.

    This dataset acts as a map-style dataset where each index corresponds to a specific
    deterministic seed from the provided list.
    """

    def __init__(
        self,
        family: SCMFamily,
        seeds: Sequence[int],
        samples_per_task: int = 128,
    ) -> None:
        self.family = family
        self.family_name = family.name
        self.seeds = list(seeds)
        self.samples_per_task = samples_per_task

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Generates and returns an SCM task for the given index.

        Returns:
            dict: {
                "seed": int,
                "data": torch.Tensor (samples),
                "adjacency": torch.Tensor (adjacency matrix)
            }
        """
        seed = int(self.seeds[idx])

        instance = self.family.sample_task(seed)
        # Deterministic sampling based on the seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            data = instance.sample(self.samples_per_task)

        adjacency = instance.adjacency_matrix.to(dtype=torch.float32)
        intervention_mask = torch.zeros_like(data)

        return {
            "seed": seed,
            "family_name": self.family_name,
            "data": data,
            "intervention_mask": intervention_mask,
            "adjacency": adjacency,
            "n_nodes": int(instance.n_nodes),
            "samples_per_task": int(self.samples_per_task),
        }


class MetaInterventionalDataset(Dataset):
    """
    Fixed-seed SCM dataset that provides both observational and interventional data.
    Performs single-node interventions on all nodes.
    """

    def __init__(
        self,
        family: SCMFamily,
        seeds: Sequence[int],
        intervention_value: float = 0.0,
        samples_per_task: int = 128,
    ) -> None:
        self.family = family
        self.family_name = family.name
        self.seeds = list(seeds)
        self.intervention_value = intervention_value
        self.samples_per_task = samples_per_task

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seed = int(self.seeds[idx])

        instance = self.family.sample_task(seed)
        n_nodes = instance.n_nodes

        # 1. Observational Data
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            x_obs = instance.sample(self.samples_per_task)

        obs_data = {"data": x_obs, "adjacency": instance.adjacency_matrix}

        # 2. Interventional Data (Single-node on all nodes)
        interventions = []
        for target_node in range(n_nodes):
            # Do(X_target = val)
            mutilated_instance = instance.do({target_node: self.intervention_value})

            # Sample from mutilated graph
            # We use a derived seed for reproducibility of interventions
            int_seed = seed + (target_node + 1) * 1000
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int_seed)
                x_int = mutilated_instance.sample(self.samples_per_task)

            interventions.append(
                {
                    "target": target_node,
                    "value": self.intervention_value,
                    "data": x_int,
                    "adjacency": mutilated_instance.adjacency_matrix,
                }
            )

        result = {
            "family_name": self.family_name,
            "observational": obs_data,
            "interventions": interventions,
            "seed": seed,
            "samples_per_task": int(self.samples_per_task),
        }

        return result


# ---------------------------------------------------------------------------
# Real-world datasets
# ---------------------------------------------------------------------------


class RealWorldDataset(Dataset):
    """Fixed real-world observational dataset with a known ground-truth DAG.

    Each ``__getitem__`` call returns a dict in exactly the same format as
    :class:`MetaFixedDataset`, making it a drop-in replacement for the
    evaluation pipeline.

    The dataset stores *one* observational matrix and *one* adjacency matrix.
    Because there is only a single graph, ``__len__`` returns the number of
    bootstrap resamples requested (``n_resamples``).  Each resample draws
    ``samples_per_task`` rows *with replacement* from the original data using a
    deterministic seed derived from ``seeds[idx]``, so repeated evaluation is
    reproducible.  When ``n_resamples`` is 1 (default), the dataset returns the
    full observational matrix without resampling.
    """

    def __init__(
        self,
        *,
        name: str,
        data: torch.Tensor,
        adjacency: torch.Tensor,
        seeds: Sequence[int],
        samples_per_task: int | None = None,
    ) -> None:
        """
        Args:
            name: Family/dataset name (e.g. ``"sachs"``).
            data: Observational data matrix, shape ``(N, d)``.
            adjacency: Ground-truth adjacency, shape ``(d, d)``, float32.
            seeds: Seed list — length determines ``__len__``.
            samples_per_task: Number of rows per resample.  If ``None`` or
                equal to ``data.shape[0]``, the full dataset is returned for
                every index (no resampling).
        """
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D, got shape {data.shape}")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"adjacency must be square, got shape {adjacency.shape}")
        if data.shape[1] != adjacency.shape[0]:
            raise ValueError(
                f"data has {data.shape[1]} variables but adjacency is "
                f"{adjacency.shape[0]}×{adjacency.shape[1]}"
            )
        self.family_name = name
        self._data = data.to(dtype=torch.float32)
        self._adjacency = adjacency.to(dtype=torch.float32)
        self._seeds = list(seeds)
        self._n_nodes = int(data.shape[1])

        # Decide whether to resample.
        full_n = int(data.shape[0])
        if samples_per_task is None or int(samples_per_task) == full_n:
            self._samples_per_task = full_n
            self._resample = False
        else:
            self._samples_per_task = int(samples_per_task)
            self._resample = True

    def __len__(self) -> int:
        return len(self._seeds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seed = int(self._seeds[idx])

        if self._resample:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                row_indices = torch.randint(
                    0, self._data.shape[0], (self._samples_per_task,)
                )
            data = self._data[row_indices]
        else:
            data = self._data

        return {
            "seed": seed,
            "family_name": self.family_name,
            "data": data,
            "intervention_mask": torch.zeros_like(data),
            "adjacency": self._adjacency,
            "n_nodes": self._n_nodes,
            "samples_per_task": int(data.shape[0]),
        }
