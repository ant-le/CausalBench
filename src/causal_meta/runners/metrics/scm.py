from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.utils.normalization import compute_scm_stats
from causal_meta.runners.utils.scoring import LinearGaussianScorer

from .base import BaseMetrics

log = logging.getLogger(__name__)


class SCMMetrics(BaseMetrics):
    """
    Single-dataset metrics handler for SCM-based evaluation (I-NIL).

    Tracks metrics for **one** SCM family. Callers must create a separate
    instance per family or call ``reset()`` between families.

    Each rank accumulates I-NIL scores locally via ``update()``.  At the end
    of the dataset, ``compute()`` / ``gather_raw_results()`` perform DDP-aware
    aggregation so that rank 0 can compute / persist the final values.

    Supports two data modes:
    1. **Pre-generated**: Pass ``interventional_data`` list.
    2. **On-the-fly**: Pass ``family`` and ``seeds``.
    """

    def __init__(self, metrics: List[str] | None = None) -> None:
        super().__init__()
        self.metrics_list = (
            metrics if metrics is not None else ["inil", "inil_per_node"]
        )
        self._warned_linear_gaussian = False

    # ── Interventional data generation ────────────────────────────────

    @staticmethod
    def _generate_interventional_data(
        family: SCMFamily,
        seed: int,
        n_samples: int,
        intervention_value: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Generate single-node interventions for an SCM instance.

        Returns:
            List of ``{"target": int, "value": float, "data": normalised_tensor}``.
        """
        instance = family.sample_task(seed)
        n_nodes = family.n_nodes

        # 1. Normalisation stats from observational data
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            x_obs_raw = instance.sample(n_samples)

        mean, std = compute_scm_stats(x_obs_raw)

        # 2. Generate interventions
        interventional_data: List[Dict[str, Any]] = []
        for target_node in range(n_nodes):
            mutilated = instance.do({target_node: intervention_value})
            int_seed = seed + (target_node + 1) * 1000

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int_seed)
                x_int_raw = mutilated.sample(n_samples)

            x_int_norm = (x_int_raw - mean) / std

            interventional_data.append(
                {"target": target_node, "value": intervention_value, "data": x_int_norm}
            )

        return interventional_data

    # ── Public API ─────────────────────────────────────────────────────

    def update(
        self,
        obs_data: torch.Tensor,
        graph_samples: torch.Tensor,
        interventional_data: List[Dict[str, Any]] | None = None,
        family: Optional[SCMFamily] = None,
        seeds: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> None:
        """Accumulate I-NIL for a single task (one graph / set of samples).

        Args:
            obs_data: Observational data ``(n_obs, n_nodes)``.
            graph_samples: Sampled graphs ``(n_samples, n_nodes, n_nodes)``.
            interventional_data: Pre-generated interventions (optional).
            family: SCMFamily used for on-the-fly generation (optional).
            seeds: Dataset seed(s) for on-the-fly generation (optional).
        """
        enabled_metrics = {
            metric
            for metric in self.metrics_list
            if metric in {"inil", "inil_per_node"}
        }
        if not enabled_metrics:
            return

        current_seeds: List[int] = []
        if seeds is not None:
            current_seeds = (
                seeds.tolist() if isinstance(seeds, torch.Tensor) else list(seeds)
            )

        # Case 1: On-the-fly generation
        if interventional_data is None and family is not None and current_seeds:
            if not self._warned_linear_gaussian:
                log.warning(
                    "I-NIL currently uses a Linear Gaussian scorer. Treat as a heuristic."
                )
                self._warned_linear_gaussian = True

            seed = int(current_seeds[0])
            int_data = self._generate_interventional_data(
                family, seed, n_samples=obs_data.shape[0]
            )
            self._compute_inil_score(obs_data, graph_samples, int_data)

        # Case 2: Pre-generated data
        elif interventional_data is not None:
            if not self._warned_linear_gaussian:
                log.warning("I-NIL using Linear Gaussian scorer.")
                self._warned_linear_gaussian = True
            self._compute_inil_score(obs_data, graph_samples, interventional_data)

    def compute(self, summary_stats: bool = True) -> Dict[str, Any]:
        """Return DDP-aggregated summary statistics."""
        return self.summarize(summary_stats=summary_stats)

    def get_raw_results(self) -> Dict[str, List[float]]:
        """Return gathered raw values across all ranks."""
        return self.gather_raw_results()

    # ── Internal helpers ───────────────────────────────────────────────

    def _compute_inil_score(
        self,
        obs_data: torch.Tensor,
        graph_samples: torch.Tensor,
        interventional_data: List[Dict[str, Any]],
    ) -> None:
        n_graph_samples = int(graph_samples.shape[0])
        task_nlls: List[float] = []
        task_nlls_per_node: List[float] = []

        # Deduplicate graphs for efficiency
        flat = graph_samples.detach().to(dtype=torch.int8).reshape(n_graph_samples, -1)
        unique_flat, inverse = torch.unique(flat, dim=0, return_inverse=True)

        per_unique_nll: List[float] = [float("inf")] * int(unique_flat.shape[0])
        per_unique_nll_per_node: List[float] = [float("inf")] * int(
            unique_flat.shape[0]
        )

        for u in range(int(unique_flat.shape[0])):
            adj = (
                unique_flat[u]
                .view(graph_samples.shape[1], graph_samples.shape[2])
                .to(dtype=graph_samples.dtype, device=graph_samples.device)
            )

            scorer = LinearGaussianScorer(adj, obs_data)
            try:
                scorer.fit()
            except RuntimeError:
                per_unique_nll[u] = float("inf")
                continue

            total_nll_k = 0.0
            total_nll_per_node_k = 0.0
            total_samples = 0

            for item in interventional_data:
                target = item["target"]
                val = item["value"]
                int_data_x = item["data"].to(obs_data.device)

                nll_avg = scorer.score_nll(
                    int_data_x, intervention_target=target, intervention_value=val
                )
                nll_per_node_avg = scorer.score_nll_per_node(
                    int_data_x,
                    intervention_target=target,
                    intervention_value=val,
                )
                total_nll_k += nll_avg * int_data_x.shape[0]
                total_nll_per_node_k += nll_per_node_avg * int_data_x.shape[0]
                total_samples += int_data_x.shape[0]

            if total_samples > 0:
                per_unique_nll[u] = total_nll_k / total_samples
                per_unique_nll_per_node[u] = total_nll_per_node_k / total_samples
            else:
                per_unique_nll[u] = 0.0
                per_unique_nll_per_node[u] = 0.0

        for k in range(n_graph_samples):
            task_nlls.append(per_unique_nll[int(inverse[k].item())])
            task_nlls_per_node.append(per_unique_nll_per_node[int(inverse[k].item())])

        # Aggregate: - log (mean ( exp ( - nlls ) ) )
        for metric_name, task_values in (
            ("inil", task_nlls),
            ("inil_per_node", task_nlls_per_node),
        ):
            if metric_name not in self.metrics_list or not task_values:
                continue

            nlls_t = torch.tensor(task_values)
            valid_nlls = nlls_t[torch.isfinite(nlls_t)]
            if valid_nlls.numel() == 0:
                continue

            neg_nlls = -valid_nlls
            log_sum_exp = torch.logsumexp(neg_nlls, dim=0)
            inil_score = -(log_sum_exp - np.log(int(valid_nlls.numel())))
            self.history[metric_name].append(inil_score.item())
