from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


class BaseMetrics:
    """Base class for single-dataset metric trackers with DDP awareness.

    Each instance tracks metrics for **one** dataset (SCM family).  Callers
    must create a separate instance per family or call ``reset()`` between
    families.

    Public API
    ----------
    - ``reset()``              — clear accumulated history.
    - ``gather_raw_results()`` — DDP-aware all-gather of raw per-batch values.
    - ``summarize()``          — DDP-aware mean / std / sem via all-reduce.
    """

    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = defaultdict(list)

    def reset(self) -> None:
        """Clear accumulated history for reuse on a new dataset."""
        self.history = defaultdict(list)

    # ── Raw results (gather across ranks) ──────────────────────────────

    def gather_raw_results(self) -> Dict[str, List[float]]:
        """All-gather raw history lists from every rank and merge them.

        After this call every rank holds the merged lists.  Useful for
        saving per-task raw values on rank 0.

        Returns:
            Merged ``{metric_key: [values_from_all_ranks]}`` dict.
            Identity (copy of local history) when not distributed.
        """
        local_obj = dict(self.history)

        if not (dist.is_available() and dist.is_initialized()):
            return dict(local_obj)

        world_size = dist.get_world_size()
        gathered: List[Dict[str, List[float]] | None] = [
            None for _ in range(world_size)
        ]
        dist.all_gather_object(gathered, local_obj)

        merged: Dict[str, List[float]] = defaultdict(list)
        for item in gathered:
            if not item:
                continue
            for k, v in item.items():
                merged[k].extend(v)
        return dict(merged)

    # ── Summary statistics (memory-efficient all_reduce) ───────────────

    def summarize(self, *, summary_stats: bool = True) -> Dict[str, Any]:
        """Compute DDP-safe summary statistics over accumulated history.

        Uses ``all_reduce`` on (sum, sum_sq, count) per metric key — never
        transfers raw data across ranks.

        Args:
            summary_stats: If ``True``, returns ``{metric}_mean/_sem/_std``.
                Otherwise returns ``{metric}: mean`` only.

        Returns:
            Dictionary of summarised metrics.
        """
        reduced = self._reduce_stats()
        results: Dict[str, Any] = {}
        for k, stats in reduced.items():
            if summary_stats:
                results[f"{k}_mean"] = stats["mean"]
                results[f"{k}_sem"] = stats["sem"]
                results[f"{k}_std"] = stats["std"]
            else:
                results[k] = stats["mean"]
        return results

    def _reduce_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute mean / std / sem, using ``all_reduce`` when distributed.

        Returns:
            ``{metric_key: {"mean", "std", "sem", "count"}}``
        """
        if not (dist.is_available() and dist.is_initialized()):
            # ── Local fast path ────────────────────────────────────────
            results: Dict[str, Dict[str, float]] = {}
            for k, v in self.history.items():
                if not v:
                    continue
                arr = np.asarray(v, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                results[k] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "sem": (
                        float(arr.std(ddof=1) / np.sqrt(arr.size))
                        if arr.size > 1
                        else 0.0
                    ),
                    "count": float(arr.size),
                }
            return results

        # ── Distributed path ───────────────────────────────────────────
        try:
            backend = dist.get_backend()
        except Exception:
            backend = "gloo"
        if backend == "nccl" and torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")

        # Collect all keys across ranks (may differ due to batch distribution)
        local_keys = set(self.history.keys())
        all_keys_list: List[set[str] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(all_keys_list, local_keys)
        all_keys: set[str] = set()
        for ks in all_keys_list:
            if ks:
                all_keys.update(ks)

        results = {}
        # CRITICAL: iterate in deterministic order so that every rank calls
        # all_reduce for the same key at the same time.  Python sets have
        # randomised iteration order across processes (due to PYTHONHASHSEED),
        # which causes all_reduce to cross-contaminate values between ranks.
        for k in sorted(all_keys):
            v = self.history.get(k, [])
            arr = np.asarray(v, dtype=float) if v else np.array([], dtype=float)
            arr = arr[np.isfinite(arr)]

            local_sum = float(arr.sum()) if arr.size > 0 else 0.0
            local_sum_sq = float((arr**2).sum()) if arr.size > 0 else 0.0
            local_count = float(arr.size)

            stats = torch.tensor([local_sum, local_sum_sq, local_count], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            total_sum = float(stats[0].item())
            total_sum_sq = float(stats[1].item())
            total_count = float(stats[2].item())

            if total_count < 1:
                continue

            mean = total_sum / total_count
            variance = max(0.0, (total_sum_sq / total_count) - (mean**2))

            if total_count > 1:
                std = np.sqrt(variance * total_count / (total_count - 1))
                sem = std / np.sqrt(total_count)
            else:
                std = 0.0
                sem = 0.0

            results[k] = {
                "mean": mean,
                "std": std,
                "sem": sem,
                "count": total_count,
            }

        return results
