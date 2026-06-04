from __future__ import annotations

import logging
import math

import numpy as np
import torch
from omegaconf import DictConfig, open_dict

from causal_meta.datasets.generators.configs import (
    ErdosRenyiConfig,
    MixtureGraphConfig,
    SBMConfig,
    ScaleFreeConfig,
)
from causal_meta.datasets.generators.factory import load_data_module_config

log = logging.getLogger(__name__)


def _estimate_sbm_edge_probability(cfg: SBMConfig, n_nodes: int) -> float:
    """Estimate expected directed edge probability for the SBM generator."""
    n_blocks = int(cfg.n_blocks)
    if n_blocks < 1:
        return 0.0

    base_size, remainder = divmod(int(n_nodes), n_blocks)
    block_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_blocks)]

    expected_edges = 0.0
    for i in range(n_blocks):
        size_i = float(block_sizes[i])
        expected_edges += (size_i * (size_i - 1.0) / 2.0) * float(cfg.p_intra)
        for j in range(i + 1, n_blocks):
            size_j = float(block_sizes[j])
            expected_edges += size_i * size_j * float(cfg.p_inter)

    max_edges = float(n_nodes * (n_nodes - 1) / 2)
    if max_edges <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, expected_edges / max_edges)))


def estimate_edge_probability_from_graph_cfg(graph_cfg: object, n_nodes: int) -> float:
    """Estimate expected edge probability from a graph config object.

    Handles ER, ScaleFree, SBM, and Mixture configs analytically.
    Falls back to Monte Carlo probing for unknown config types.
    """
    max_edges = float(n_nodes * (n_nodes - 1) / 2)
    if max_edges <= 0.0:
        return 0.0

    if isinstance(graph_cfg, ErdosRenyiConfig):
        p = graph_cfg.edge_prob
        if p is None:
            p = graph_cfg.sparsity
        if p is None:
            raise ValueError("ErdosRenyiConfig must define edge_prob or sparsity.")
        return float(max(0.0, min(1.0, float(p))))

    if isinstance(graph_cfg, ScaleFreeConfig):
        m = int(graph_cfg.m)
        expected_edges = float(m * n_nodes - (m * (m + 1) / 2.0))
        return float(max(0.0, min(1.0, expected_edges / max_edges)))

    if isinstance(graph_cfg, SBMConfig):
        return _estimate_sbm_edge_probability(graph_cfg, n_nodes)

    if isinstance(graph_cfg, MixtureGraphConfig):
        if len(graph_cfg.generators) != len(graph_cfg.weights):
            raise ValueError("MixtureGraphConfig generators/weights length mismatch.")
        total = float(sum(graph_cfg.weights))
        if total <= 0.0:
            raise ValueError("MixtureGraphConfig requires positive total weight.")
        probs = [
            estimate_edge_probability_from_graph_cfg(sub_cfg, n_nodes)
            for sub_cfg in graph_cfg.generators
        ]
        weighted = sum(float(w) * float(p) for w, p in zip(graph_cfg.weights, probs))
        return float(max(0.0, min(1.0, weighted / total)))

    if not hasattr(graph_cfg, "instantiate"):
        raise ValueError(
            f"Unsupported graph config for random baseline: {type(graph_cfg)}"
        )

    generator = graph_cfg.instantiate()
    probe = 128
    base_seed = 0
    rng = np.random.default_rng(base_seed)
    torch_gen = torch.Generator().manual_seed(base_seed)
    total_edges = 0.0
    for i in range(probe):
        adjacency = generator(
            n_nodes,
            seed=base_seed + i,
            torch_generator=torch_gen,
            rng=rng,
        )
        total_edges += float(adjacency.sum().item())
    return float(max(0.0, min(1.0, total_edges / (probe * max_edges))))


def infer_edge_probability(cfg: DictConfig) -> float:
    """Infer a training-sparsity-matched edge probability for the random baseline.

    Args:
        cfg: Full Hydra config (must contain ``data.train_family``).

    Returns:
        Estimated expected directed edge probability of the training graph family.
    """
    data_cfg = load_data_module_config(cfg.data)
    train_family = data_cfg.train_family
    p_edge = estimate_edge_probability_from_graph_cfg(
        train_family.graph_cfg,
        n_nodes=int(train_family.n_nodes),
    )
    if not (0.0 <= p_edge <= 1.0) or math.isnan(p_edge):
        raise ValueError(f"Invalid inferred random baseline p_edge={p_edge}.")
    return float(p_edge)


def maybe_fill_edge_prior(cfg: DictConfig) -> None:
    """Auto-fill ``model.p_edge`` from training data if not already set.

    Only acts when ``model.type == "random"`` and ``model.p_edge`` is unset.
    Mutates ``cfg`` in-place.
    """
    model_cfg = cfg.model
    model_type = str(getattr(model_cfg, "type", "")).lower()
    if model_type != "random":
        return
    if hasattr(model_cfg, "p_edge") and getattr(model_cfg, "p_edge") is not None:
        return

    inferred_p_edge = infer_edge_probability(cfg)
    with open_dict(cfg):
        cfg.model.p_edge = float(inferred_p_edge)
