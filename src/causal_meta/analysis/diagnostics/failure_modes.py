"""Failure mode classification for causal discovery predictions.

Classifies each per-task prediction into one of the following categories based
on the Phase A decomposition metrics:

- **empty**: Predicted graph has near-zero density (sparsity_ratio ≈ 0).
- **dense**: Predicted graph is over-dense (sparsity_ratio > 2×).
- **reversed**: Skeleton is largely correct but most shared edges have the
  wrong orientation (orientation_accuracy < threshold).
- **sparse**: Model under-predicts edges (sparsity_ratio < 0.5) but not empty.
- **reasonable**: None of the above.

The classification operates on the per-task raw metrics from ``metrics.json``
loaded via :func:`~causal_meta.analysis.utils.load_raw_task_dataframe`.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from causal_meta.analysis.common.thesis import axis_category

log = logging.getLogger(__name__)

# ── Thresholds (tunable) ───────────────────────────────────────────────

EMPTY_SPARSITY_UPPER: float = 0.05
"""sparsity_ratio below this → ``empty``."""

DENSE_SPARSITY_LOWER: float = 2.0
"""sparsity_ratio above this → ``dense``."""

SPARSE_SPARSITY_UPPER: float = 0.5
"""sparsity_ratio below this (but above EMPTY_SPARSITY_UPPER) → ``sparse``."""

REVERSED_ORIENTATION_UPPER: float = 0.4
"""orientation_accuracy below this with skeleton_f1 ≥ REVERSED_SKELETON_LOWER → ``reversed``."""

REVERSED_SKELETON_LOWER: float = 0.5
"""Minimum skeleton_f1 to qualify for ``reversed`` (must have a skeleton to mis-orient)."""

# ── Category labels ────────────────────────────────────────────────────

FAILURE_MODE_CATEGORIES: list[str] = [
    "empty",
    "dense",
    "reversed",
    "sparse",
    "reasonable",
]
"""Ordered list of failure mode categories (used for consistent colouring)."""

FAILURE_MODE_COLORS: dict[str, str] = {
    "empty": "#1f77b4",
    "dense": "#d62728",
    "reversed": "#ff7f0e",
    "sparse": "#9467bd",
    "reasonable": "#2ca02c",
}


# ── Classification ─────────────────────────────────────────────────────


def _pivot_raw_wide(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format raw-task DataFrame to one row per task.

    Expects columns: ``RunID``, ``Model``, ``DatasetKey``, ``TaskIdx``,
    ``Metric``, ``Value``, plus optional enrichment columns.
    """
    if raw_df.empty or "Metric" not in raw_df.columns:
        return pd.DataFrame()

    # Use stable task identifiers only. Optional enrichment columns can contain
    # NaNs for specific shift axes (e.g. ``SparsityParam`` on SBM/WS/GRG), and
    # including them in the pivot index can silently drop those rows.
    id_cols = [
        c
        for c in (
            "RunID",
            "Model",
            "ModelKey",
            "DatasetKey",
            "Dataset",
            "TaskIdx",
        )
        if c in raw_df.columns
    ]

    wide = raw_df.pivot_table(
        index=id_cols,
        columns="Metric",
        values="Value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    enrichment_cols = [
        c
        for c in (
            "GraphType",
            "MechType",
            "NNodes",
            "SamplesPerTask",
            "SparsityParam",
            "SpectralDist",
            "KLDegreeDist",
            "MechanismDist",
        )
        if c in raw_df.columns
    ]
    if enrichment_cols:
        enrichment = raw_df[id_cols + enrichment_cols].drop_duplicates(subset=id_cols)
        wide = wide.merge(enrichment, on=id_cols, how="left")

    return wide


def classify_failure_modes(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a failure mode category to each per-task prediction.

    Args:
        raw_df: Long-format DataFrame from
            :func:`~causal_meta.analysis.utils.load_raw_task_dataframe`
            containing at least metrics ``sparsity_ratio``,
            ``skeleton_f1``, and ``orientation_accuracy``.

    Returns:
        A DataFrame with one row per task and a ``FailureMode`` column.
        Also includes the original metric values and enrichment columns.
    """
    needed_metrics = {"sparsity_ratio", "skeleton_f1", "orientation_accuracy"}
    if raw_df.empty or "Metric" not in raw_df.columns:
        log.warning("Empty raw DataFrame; returning empty failure-mode table.")
        return pd.DataFrame()

    available_metrics = set(raw_df["Metric"].unique())
    if not needed_metrics.issubset(available_metrics):
        missing = needed_metrics - available_metrics
        log.warning(
            "Missing metrics for failure-mode classification: %s; skipping.",
            missing,
        )
        return pd.DataFrame()

    wide = _pivot_raw_wide(raw_df)
    if wide.empty:
        return pd.DataFrame()

    # Ensure required columns exist after pivot
    for col in needed_metrics:
        if col not in wide.columns:
            log.warning("Column %s not found after pivot; skipping.", col)
            return pd.DataFrame()

    # Apply classification rules (order matters — first match wins)
    conditions: list[tuple[str, pd.Series]] = []

    sr = wide["sparsity_ratio"]
    sf1 = wide["skeleton_f1"]
    oa = wide["orientation_accuracy"]

    conditions.append(("empty", sr < EMPTY_SPARSITY_UPPER))
    conditions.append(("dense", sr > DENSE_SPARSITY_LOWER))
    conditions.append(
        (
            "reversed",
            (sf1 >= REVERSED_SKELETON_LOWER) & (oa < REVERSED_ORIENTATION_UPPER),
        )
    )
    conditions.append(
        (
            "sparse",
            (sr >= EMPTY_SPARSITY_UPPER) & (sr < SPARSE_SPARSITY_UPPER),
        )
    )
    # Default
    wide["FailureMode"] = "reasonable"
    # Apply in reverse order so higher-priority (earlier) rules overwrite
    for label, mask in reversed(conditions):
        wide.loc[mask, "FailureMode"] = label

    return wide


# ── Aggregation ────────────────────────────────────────────────────────


def failure_mode_fractions(
    classified_df: pd.DataFrame,
    group_cols: Sequence[str] = ("Model", "DatasetKey"),
) -> pd.DataFrame:
    """Compute failure-mode fractions per group.

    Args:
        classified_df: DataFrame with a ``FailureMode`` column (from
            :func:`classify_failure_modes`).
        group_cols: Columns to group by before computing fractions.

    Returns:
        DataFrame with ``group_cols`` + one column per failure mode category
        (values in [0, 1] summing to 1 per row).
    """
    if classified_df.empty or "FailureMode" not in classified_df.columns:
        return pd.DataFrame()

    valid_groups = [c for c in group_cols if c in classified_df.columns]
    if not valid_groups:
        return pd.DataFrame()

    counts = (
        classified_df.groupby([*valid_groups, "FailureMode"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure all categories are present
    for cat in FAILURE_MODE_CATEGORIES:
        if cat not in counts.columns:
            counts[cat] = 0
    counts = counts[FAILURE_MODE_CATEGORIES]

    fractions = counts.div(counts.sum(axis=1), axis=0)
    return fractions.reset_index()


def ood_category(dataset_key: str, *, binary: bool = False) -> str:
    """Classify a dataset key using the canonical thesis shift taxonomy.

    Args:
        dataset_key: The dataset key string to classify.
        binary: If ``True``, return only ``"ID"`` or ``"OOD"``
            (coarse classification for tables).
    """
    category = axis_category(dataset_key)
    if category == "id":
        return "ID"
    if binary:
        return "OOD"

    label_map = {
        "graph": "OOD-Graph",
        "mechanism": "OOD-Mech",
        "noise": "OOD-Noise",
        "compound": "OOD-Both",
        "nodes": "OOD-Nodes",
        "samples": "OOD-Samples",
    }
    return label_map.get(category, "OOD")
