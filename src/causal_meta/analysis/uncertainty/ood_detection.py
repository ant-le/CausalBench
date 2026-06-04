"""OOD detection and selective prediction analysis.

Treats the problem as binary classification (ID vs OOD) using posterior
uncertainty measures (``edge_entropy``, ``graph_nll_per_edge``) as detection scores.
Computes AUROC / AUPRC per model and generates selective prediction
(accuracy-vs-coverage) Pareto frontiers.

All functions consume the per-task raw DataFrame produced by
:func:`~causal_meta.analysis.utils.load_raw_task_dataframe`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────


def _is_ood(dataset_key: str) -> bool:
    """Return True if the dataset key is OOD (not in-distribution)."""
    dk = dataset_key.lower()
    return not dk.startswith("id_")


def _pivot_raw_wide(
    raw_df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Pivot raw long-format to wide (one row per task) for selected metrics."""
    if raw_df.empty or "Metric" not in raw_df.columns:
        return pd.DataFrame()

    subset = raw_df[raw_df["Metric"].isin(metrics)].copy()
    if subset.empty:
        return pd.DataFrame()

    id_cols = [
        c
        for c in (
            "RunID",
            "Model",
            "ModelKey",
            "DatasetKey",
            "Dataset",
            "TaskIdx",
            "GraphType",
            "MechType",
            "NNodes",
            "SamplesPerTask",
            "SparsityParam",
            "SpectralDist",
            "KLDegreeDist",
        )
        if c in subset.columns
    ]

    wide = subset.pivot_table(
        index=id_cols,
        columns="Metric",
        values="Value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


# ── OOD Detection AUROC/AUPRC ──────────────────────────────────────────


def _roc_auc_manual(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn dependency.

    Uses the Wilcoxon–Mann–Whitney statistic (equivalent to AUC).
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    # Count concordant pairs
    n_concordant = 0
    n_tied = 0
    for p in pos:
        n_concordant += int(np.sum(neg < p))
        n_tied += int(np.sum(neg == p))

    auc = (n_concordant + 0.5 * n_tied) / (len(pos) * len(neg))
    return float(auc)


def _precision_recall_auc_manual(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute area under the Precision-Recall curve without sklearn.

    Uses the trapezoidal rule on sorted thresholds.
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")

    # Sort by descending score
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp_cumsum = np.cumsum(sorted_labels)
    n_predicted = np.arange(1, len(sorted_labels) + 1)
    precisions = tp_cumsum / n_predicted
    recalls = tp_cumsum / labels.sum()

    # Prepend (recall=0, precision=1) for area calculation
    precisions = np.concatenate([[1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])

    # Trapezoidal AUC (np.trapezoid replaces np.trapz in NumPy 2.0+)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz is None:
        raise RuntimeError("No trapezoidal integration function found in numpy.")
    auprc = float(_trapz(precisions, recalls))
    return abs(auprc)  # sign depends on recall ordering


def compute_ood_detection_metrics(
    raw_df: pd.DataFrame,
    score_metric: str = "edge_entropy",
) -> pd.DataFrame:
    """Compute OOD detection AUROC and AUPRC per model.

    Args:
        raw_df: Long-format per-task DataFrame from ``load_raw_task_dataframe``.
        score_metric: Which metric to use as OOD detection score.
            Higher score should indicate more likely OOD.

    Returns:
        DataFrame with columns ``RunID`` (if available), ``Model``,
        ``ScoreMetric``, ``AUROC``, ``AUPRC``, ``N_ID``, ``N_OOD``.
    """
    if raw_df.empty or "Metric" not in raw_df.columns:
        return pd.DataFrame()

    # Pivot selected score metric to one row per task.
    wide = _pivot_raw_wide(raw_df, [score_metric])
    if wide.empty or score_metric not in wide.columns:
        log.warning("Score metric '%s' not found in raw data; skipping.", score_metric)
        return pd.DataFrame()

    if "DatasetKey" not in wide.columns or "Model" not in wide.columns:
        return pd.DataFrame()

    wide["is_ood"] = wide["DatasetKey"].apply(_is_ood).astype(int)

    group_cols = [c for c in ("RunID", "Model") if c in wide.columns]
    if not group_cols:
        group_cols = ["Model"]

    rows: list[dict[str, object]] = []
    for group_key, group_df in wide.groupby(group_cols, dropna=False):
        if isinstance(group_key, tuple):
            key_map = {k: v for k, v in zip(group_cols, group_key)}
        else:
            key_map = {group_cols[0]: group_key}

        model = str(key_map.get("Model", "unknown"))
        mdf = group_df.dropna(subset=[score_metric])
        labels = mdf["is_ood"].to_numpy()
        scores = mdf[score_metric].to_numpy(dtype=float)

        n_id = int((labels == 0).sum())
        n_ood = int((labels == 1).sum())

        if n_id == 0 or n_ood == 0:
            continue

        auroc = _roc_auc_manual(labels, scores)
        auprc = _precision_recall_auc_manual(labels, scores)

        row: dict[str, object] = {
            "Model": model,
            "ScoreMetric": score_metric,
            "AUROC": round(auroc, 4),
            "AUPRC": round(auprc, 4),
            "N_ID": n_id,
            "N_OOD": n_ood,
        }
        if "RunID" in key_map:
            row["RunID"] = str(key_map["RunID"])
        rows.append(row)

    return pd.DataFrame(rows)


# ── Selective Prediction ────────────────────────────────────────────────


def compute_selective_prediction(
    raw_df: pd.DataFrame,
    score_metric: str = "edge_entropy",
    accuracy_metrics: Sequence[str] = ("ne-shd", "ne-sid"),
    n_thresholds: int = 50,
) -> pd.DataFrame:
    """Compute selective prediction curves: accuracy vs coverage at varying thresholds.

    Policy: reject predictions where ``score_metric > threshold``.

    Args:
        raw_df: Long-format per-task DataFrame.
        score_metric: Uncertainty metric (higher = less trustworthy).
        accuracy_metrics: Performance metrics to evaluate for accepted
            predictions. This function expects a multi-metric sequence
            (for example ``["ne-shd", "ne-sid"]``).
        n_thresholds: Number of threshold values to sweep.

    Returns:
        DataFrame with columns ``RunID`` (if available), ``Model``,
        ``AccuracyMetric``, ``Threshold``, ``Coverage``, ``MeanValue``,
        ``MeanAccuracy`` (backward-compatible alias), and ``N_Accepted``.
    """
    if raw_df.empty or "Metric" not in raw_df.columns:
        return pd.DataFrame()

    if isinstance(accuracy_metrics, str):
        raise ValueError("accuracy_metrics must be a sequence, not a string.")
    metric_list = list(dict.fromkeys(accuracy_metrics))
    if len(metric_list) < 2:
        raise ValueError(
            "accuracy_metrics must contain at least two metrics (multi-metric mode)."
        )

    wide = _pivot_raw_wide(raw_df, [score_metric, *metric_list])
    needed = {score_metric, *metric_list}
    if wide.empty or not needed.issubset(wide.columns):
        log.warning(
            "Missing metrics for selective prediction (%s); skipping.",
            needed - set(wide.columns) if not wide.empty else needed,
        )
        return pd.DataFrame()

    group_cols = [c for c in ("RunID", "Model") if c in wide.columns]
    if not group_cols:
        group_cols = ["Model"]

    rows: list[dict[str, object]] = []
    for group_key, group_df in wide.groupby(group_cols, dropna=False):
        if isinstance(group_key, tuple):
            key_map = {k: v for k, v in zip(group_cols, group_key)}
        else:
            key_map = {group_cols[0]: group_key}

        model = str(key_map.get("Model", "unknown"))
        mdf = group_df.dropna(subset=[score_metric, *metric_list])
        if mdf.empty:
            continue

        scores = mdf[score_metric].to_numpy(dtype=float)
        total = len(scores)

        lo, hi = float(scores.min()), float(scores.max())
        if lo == hi:
            thresholds = np.array([lo])
        else:
            thresholds = np.linspace(lo, hi, n_thresholds)

        for accuracy_metric in metric_list:
            accuracy = mdf[accuracy_metric].to_numpy(dtype=float)
            for t in thresholds:
                mask = scores <= t
                n_accepted = int(mask.sum())
                if n_accepted == 0:
                    continue
                mean_value = float(accuracy[mask].mean())
                coverage = n_accepted / total

                row: dict[str, object] = {
                    "Model": model,
                    "ScoreMetric": score_metric,
                    "AccuracyMetric": accuracy_metric,
                    "Threshold": round(float(t), 6),
                    "Coverage": round(coverage, 6),
                    "MeanValue": round(mean_value, 6),
                    "MeanAccuracy": round(mean_value, 6),
                    "N_Accepted": n_accepted,
                }
                if "RunID" in key_map:
                    row["RunID"] = str(key_map["RunID"])
                rows.append(row)

    return pd.DataFrame(rows)


# ── OOD Detection Table ────────────────────────────────────────────────


def generate_ood_detection_table(
    detection_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write a LaTeX table summarising OOD detection AUROC/AUPRC per model.

    Args:
        detection_df: Output of :func:`compute_ood_detection_metrics`.
        output_path: Where to write the ``.tex`` file.
    """
    output_path = Path(output_path)

    if detection_df.empty:
        log.warning("Empty detection DataFrame; skipping OOD detection table.")
        return

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Score} & \textbf{AUROC} & \textbf{AUPRC}"
        r" & \textbf{$N_{\mathrm{ID}}$} & \textbf{$N_{\mathrm{OOD}}$} \\"
    )
    lines.append(r"\midrule")

    # Best values for bold highlighting.
    best_auroc = float(detection_df["AUROC"].max())
    best_auprc = float(detection_df["AUPRC"].max())

    for _, row in detection_df.iterrows():
        model_label = str(row["Model"])
        run_id = row.get("RunID")
        if run_id is not None and str(run_id).strip() and str(run_id).lower() != "nan":
            model_label = f"{model_label} [{run_id}]"
        model_tex = model_label.replace("_", r"\_")
        auroc_s = f"{row['AUROC']:.3f}"
        auprc_s = f"{row['AUPRC']:.3f}"
        if abs(float(row["AUROC"]) - best_auroc) < 1e-9:
            auroc_s = r"\textbf{" + auroc_s + "}"
        if abs(float(row["AUPRC"]) - best_auprc) < 1e-9:
            auprc_s = r"\textbf{" + auprc_s + "}"
        lines.append(
            f"{model_tex} & {row['ScoreMetric']}"
            f" & {auroc_s} & {auprc_s}"
            f" & {row['N_ID']} & {row['N_OOD']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log.info("Saved OOD detection table to %s", output_path)
