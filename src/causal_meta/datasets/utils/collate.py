from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

import torch

from causal_meta.datasets.utils.normalization import (
    compute_scm_stats,
    normalize_scm_data,
)

ScmItem = Mapping[str, Any]


def collate_fn_scm(batch: Iterable[ScmItem], normalize: bool = True) -> Dict[str, Any]:
    """
    Collate function for SCM batches that stacks and normalizes data.

    Expected item format:
      {"seed": int, "data": Tensor[S, N], "adjacency": Tensor[N, N]}
    """
    batch_data: List[torch.Tensor] = []
    intervention_masks: List[torch.Tensor] = []
    adjacency_matrices: List[torch.Tensor] = []
    node_counts: List[int] = []
    seeds: List[int | None] = []
    family_names: List[str | None] = []
    samples_per_task: List[int | None] = []

    for item in batch:
        if not isinstance(item, Mapping):
            raise TypeError(f"Expected dict item in collate_fn_scm, got {type(item)}")

        seeds.append(int(item["seed"]) if "seed" in item else None)
        family_names.append(
            str(item["family_name"]) if item.get("family_name") is not None else None
        )
        samples_per_task.append(
            int(item["samples_per_task"])
            if item.get("samples_per_task") is not None
            else None
        )
        data = item["data"].float()
        intervention_mask = item.get(
            "intervention_mask", torch.zeros_like(data)
        ).float()
        if data.shape != intervention_mask.shape:
            raise ValueError(
                "intervention_mask must have the same shape as data in collate_fn_scm."
            )
        adjacency = item["adjacency"].float()
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix in collate_fn_scm.")

        batch_data.append(data)
        intervention_masks.append(intervention_mask)
        adjacency_matrices.append(adjacency)
        node_counts.append(int(data.shape[-1]))

    if normalize and batch_data:
        shapes = {(int(x.shape[0]), int(x.shape[1])) for x in batch_data}
        if len(shapes) == 1:
            stacked = normalize_scm_data(torch.stack(batch_data))
            batch_data = list(stacked.unbind(dim=0))
        else:
            normalized: List[torch.Tensor] = []
            for data in batch_data:
                mean, std = compute_scm_stats(data)
                normalized.append((data - mean) / std)
            batch_data = normalized

    max_samples = max(int(x.shape[0]) for x in batch_data)
    max_nodes = max(int(x.shape[1]) for x in batch_data)

    data_tensor = torch.zeros(
        len(batch_data),
        max_samples,
        max_nodes,
        dtype=batch_data[0].dtype,
    )
    intervention_tensor = torch.zeros_like(data_tensor)
    adjacency_tensor = torch.zeros(
        len(adjacency_matrices),
        max_nodes,
        max_nodes,
        dtype=adjacency_matrices[0].dtype,
    )
    node_mask = torch.ones(len(batch_data), max_nodes, dtype=torch.bool)
    sample_mask = torch.ones(len(batch_data), max_samples, dtype=torch.bool)

    for idx, (data, intervention_mask, adjacency) in enumerate(
        zip(batch_data, intervention_masks, adjacency_matrices)
    ):
        n_samples, n_nodes = data.shape
        data_tensor[idx, :n_samples, :n_nodes] = data
        intervention_tensor[idx, :n_samples, :n_nodes] = intervention_mask
        adjacency_tensor[idx, :n_nodes, :n_nodes] = adjacency
        node_mask[idx, :n_nodes] = False
        sample_mask[idx, :n_samples] = False

    return {
        "seed": seeds if any(s is not None for s in seeds) else None,
        "family_name": (
            family_names if any(name is not None for name in family_names) else None
        ),
        "data": data_tensor,
        "intervention_mask": intervention_tensor,
        "adjacency": adjacency_tensor,
        "node_mask": node_mask,
        "sample_mask": sample_mask,
        "n_nodes": torch.tensor(node_counts, dtype=torch.long),
        "samples_per_task": (
            samples_per_task
            if any(sample_count is not None for sample_count in samples_per_task)
            else None
        ),
    }


def collate_fn_interventional(
    batch: Iterable[Dict[str, Any]], normalize: bool = True
) -> Dict[str, Any]:
    """
    Collate function for interventional batches.
    Normalizes both observational and interventional data using observational statistics.
    """
    items = list(batch)
    if not items:
        raise ValueError("collate_fn_interventional received an empty batch.")

    seeds: List[int] = []
    family_names: List[str | None] = []
    samples_per_task: List[int | None] = []
    obs_data_list: List[torch.Tensor] = []
    obs_adj_list: List[torch.Tensor] = []

    int_target_list: List[torch.Tensor] = []
    int_value_list: List[torch.Tensor] = []
    int_data_list: List[torch.Tensor] = []
    int_adj_list: List[torch.Tensor] = []

    expected_num_interventions: int | None = None

    for item in items:
        seeds.append(int(item["seed"]))
        family_names.append(
            str(item["family_name"]) if item.get("family_name") is not None else None
        )
        samples_per_task.append(
            int(item["samples_per_task"])
            if item.get("samples_per_task") is not None
            else None
        )

        obs = item["observational"]
        obs_data = obs["data"].float()
        obs_adj = obs["adjacency"].float()

        # Stats on observational data: (S, N)
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)
        if normalize:
            mean, std = compute_scm_stats(obs_data)
            obs_data = (obs_data - mean) / std

        obs_data_list.append(obs_data)
        obs_adj_list.append(obs_adj)

        interventions = item["interventions"]
        if expected_num_interventions is None:
            expected_num_interventions = len(interventions)
        elif len(interventions) != expected_num_interventions:
            raise ValueError(
                "All items in an interventional batch must have the same number of interventions."
            )

        targets: List[int] = []
        values: List[float] = []
        x_ints: List[torch.Tensor] = []
        adjs: List[torch.Tensor] = []

        for int_item in interventions:
            targets.append(int(int_item["target"]))
            values.append(float(int_item["value"]))

            x_int = int_item["data"].float()
            if normalize:
                x_int = (x_int - mean) / std

            x_ints.append(x_int)
            adjs.append(int_item["adjacency"].float())

        int_target_list.append(torch.tensor(targets, dtype=torch.long))
        int_value_list.append(torch.tensor(values, dtype=torch.float32))
        int_data_list.append(torch.stack(x_ints, dim=0))
        int_adj_list.append(torch.stack(adjs, dim=0))

    return {
        "seed": torch.tensor(seeds, dtype=torch.long),
        "family_name": (
            family_names if any(name is not None for name in family_names) else None
        ),
        "observational": {
            "data": torch.stack(obs_data_list, dim=0),
            "adjacency": torch.stack(obs_adj_list, dim=0),
        },
        "interventions": {
            "target": torch.stack(int_target_list, dim=0),
            "value": torch.stack(int_value_list, dim=0),
            "data": torch.stack(int_data_list, dim=0),
            "adjacency": torch.stack(int_adj_list, dim=0),
        },
        "samples_per_task": (
            samples_per_task
            if any(sample_count is not None for sample_count in samples_per_task)
            else None
        ),
    }
