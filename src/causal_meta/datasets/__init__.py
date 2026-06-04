from __future__ import annotations

from typing import Any

__all__ = [
    "SCMFamily",
    "SCMInstance",
    "MetaIterableDataset",
    "MetaFixedDataset",
    "RealWorldDataset",
    "collate_fn_scm",
    "compute_graph_hash",
    "DataModuleConfig",
    "FamilyConfig",
    "RealWorldFamilyConfig",
    "CausalMetaModule",
    "get_family_stats",
    "plot_degree_distribution",
    "visualize_adjacency",
    "compute_family_distance",
]


def __getattr__(name: str) -> Any:
    if name == "CausalMetaModule":
        from causal_meta.datasets.data_module import CausalMetaModule

        return CausalMetaModule
    if name in {"DataModuleConfig", "FamilyConfig", "RealWorldFamilyConfig"}:
        from causal_meta.datasets.generators import configs

        return getattr(configs, name)
    if name in {"SCMFamily", "SCMInstance"}:
        from causal_meta.datasets import scm

        return getattr(scm, name)
    if name in {"MetaIterableDataset", "MetaFixedDataset", "RealWorldDataset"}:
        from causal_meta.datasets import torch_datasets

        return getattr(torch_datasets, name)
    if name in {
        "collate_fn_scm",
        "compute_graph_hash",
        "get_family_stats",
        "plot_degree_distribution",
        "visualize_adjacency",
        "compute_family_distance",
    }:
        from causal_meta.datasets import utils

        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
