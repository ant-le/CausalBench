"""Additional deep-dive analyses for thesis Chapter 5.

This module produces six new figures and tables that complement the
existing RQ1/RQ2/RQ3 analyses with cross-cutting insights:

1. Inference efficiency frontier  (RQ2)
2. Distance–degradation scatter   (RQ1/RQ3)
3. AUC vs Edge-F1 divergence      (RQ1)
4. Sparsity dynamics under shift   (RQ1/RQ2)
5. Accuracy–calibration joint plot (RQ3)
6. Per-axis family win-count table (RQ2)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_meta.analysis.common.plot_style import (COMPACT_SINGLE_FIGSIZE,
                                                    HAT_E, LABEL_FONTSIZE,
                                                    LEGEND_FONTSIZE,
                                                    SINGLE_FIGSIZE,
                                                    SUBTITLE_FONTSIZE,
                                                    TITLE_FONTSIZE,
                                                    TOP_LEGEND_ANCHOR,
                                                    TOP_LEGEND_RECT)
from causal_meta.analysis.common.thesis import (axis_category,
                                                is_fixed_size_task_frame,
                                                metric_sem)
from causal_meta.analysis.utils import (MODEL_COLORS, MODEL_MARKERS,
                                        PAPER_MODEL_LABELS,
                                        EmptyAnalysisDataError,
                                        save_figure_data)

log = logging.getLogger(__name__)

_HAT_E = HAT_E
_TITLE_FONTSIZE = TITLE_FONTSIZE
_PANEL_TITLE_FONTSIZE = SUBTITLE_FONTSIZE
_LABEL_FONTSIZE = LABEL_FONTSIZE
_LEGEND_FONTSIZE = LEGEND_FONTSIZE


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#555555")


def _model_marker(model: str) -> str:
    return MODEL_MARKERS.get(model, "o")


def _canonical_model_order() -> list[str]:
    return list(PAPER_MODEL_LABELS.values())


_AXIS_DISPLAY: dict[str, str] = {
    "id": "ID",
    "graph": "Graph",
    "mechanism": "Mechanism",
    "noise": "Noise",
    "compound": "Compound",
    "nodes": "Nodes",
    "samples": "Samples",
}

_AXIS_COLORS: dict[str, str] = {
    "id": "#2ca02c",
    "graph": "#d62728",
    "mechanism": "#9467bd",
    "noise": "#8c564b",
    "compound": "#e377c2",
    "nodes": "#ff7f0e",
    "samples": "#1f77b4",
}


# =====================================================================
# 1. Inference Efficiency Frontier (RQ2)
# =====================================================================


def generate_efficiency_frontier(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Pareto-style scatter: mean inference time vs mean ne-SID.

    Each model appears as two markers (ID average and OOD average) connected
    by an arrow, revealing the cost–accuracy trade-off and how it shifts
    under distribution shift.
    """
    needed = {"ne-sid", "inference_time_s"}
    subset = raw_df[raw_df["Metric"].isin(needed)].copy()
    if subset.empty:
        raise EmptyAnalysisDataError("No data for efficiency frontier.")

    subset["Split"] = np.where(subset["AxisCategory"].eq("id"), "ID", "OOD")

    agg = (
        subset.groupby(["Model", "Split", "Metric"], dropna=False)["Value"]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=["Model", "Split"], columns="Metric", values="Value"
    ).reset_index()
    pivot.columns.name = None

    if "ne-sid" not in pivot.columns or "inference_time_s" not in pivot.columns:
        raise EmptyAnalysisDataError("Missing columns for efficiency frontier.")

    models = [m for m in _canonical_model_order() if m in pivot["Model"].unique()]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    split_markers = {"ID": "o", "OOD": "^"}
    for model in models:
        mdf = pivot[pivot["Model"] == model]
        color = _model_color(model)

        for _, row in mdf.iterrows():
            split = str(row["Split"])
            ax.scatter(
                row["inference_time_s"],
                row["ne-sid"],
                c=color,
                marker=split_markers.get(split, "o"),
                s=120,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )

        # Draw arrow from ID to OOD for this model.
        id_row = mdf[mdf["Split"] == "ID"]
        ood_row = mdf[mdf["Split"] == "OOD"]
        if not id_row.empty and not ood_row.empty:
            x0 = float(id_row.iloc[0]["inference_time_s"])
            y0 = float(id_row.iloc[0]["ne-sid"])
            x1 = float(ood_row.iloc[0]["inference_time_s"])
            y1 = float(ood_row.iloc[0]["ne-sid"])
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.5,
                    alpha=0.6,
                ),
            )
            # Model label near ID marker.
            ax.annotate(
                model,
                xy=(x0, y0),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    ax.set_xscale("log")
    ax.set_xlabel(
        "Mean inference time per task (seconds, log scale)", fontsize=_LABEL_FONTSIZE
    )
    ax.set_ylabel(
        rf"Mean normalized ${_HAT_E}$-SID $\downarrow$", fontsize=_LABEL_FONTSIZE
    )
    ax.set_title(
        "Inference Efficiency Frontier", fontsize=_TITLE_FONTSIZE, fontweight="bold"
    )
    ax.grid(True, linestyle="--", alpha=0.4)

    # Custom legend for split shapes.
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            linestyle="None",
            markersize=8,
            label="ID",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="grey",
            linestyle="None",
            markersize=8,
            label="OOD",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=_LEGEND_FONTSIZE,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, pivot)
    log.info("Saved efficiency frontier to %s", output_path)
    return pivot


# =====================================================================
# 2. Distance–Degradation Scatter (RQ1 / RQ3)
# =====================================================================


def generate_distance_degradation_scatter(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Scatter of mechanism distance vs. ne-SID for each model.

    Only fixed-size (d=20, n=500) OOD families are included so the
    relationship is not confounded by node-count or sample-count effects.
    A Spearman correlation is annotated per panel.
    """
    from scipy.stats import spearmanr

    needed = {"ne-sid"}
    subset = raw_df[raw_df["Metric"].isin(needed)].copy()
    subset = subset[is_fixed_size_task_frame(subset)]
    subset = subset[subset["AxisCategory"] != "id"]
    if subset.empty:
        raise EmptyAnalysisDataError("No data for distance-degradation scatter.")

    agg = (
        subset.groupby(["Model", "DatasetKey", "AxisCategory"], dropna=False)
        .agg(
            ne_sid=("Value", "mean"),
            MechanismDist=("MechanismDist", "first"),
        )
        .reset_index()
    )
    agg = agg.dropna(subset=["ne_sid", "MechanismDist"])
    if agg.empty:
        raise EmptyAnalysisDataError("No mechanism distance data for scatter.")

    models = [m for m in _canonical_model_order() if m in agg["Model"].unique()]
    n_models = len(models)
    if n_models == 0:
        raise EmptyAnalysisDataError("No models for distance-degradation scatter.")

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        mdf = agg[agg["Model"] == model]

        for cat in sorted(mdf["AxisCategory"].unique()):
            cdf = mdf[mdf["AxisCategory"] == cat]
            ax.scatter(
                cdf["MechanismDist"],
                cdf["ne_sid"],
                c=_AXIS_COLORS.get(cat, "#aaaaaa"),
                label=_AXIS_DISPLAY.get(cat, cat),
                s=60,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )

        if len(mdf) >= 3:
            rho, p_val = spearmanr(mdf["MechanismDist"], mdf["ne_sid"])
            if np.isfinite(rho):
                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                ax.annotate(
                    f"Spearman $\\rho$={rho:.2f}\n({p_str})",
                    xy=(0.03, 0.97),
                    xycoords="axes fraction",
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

        ax.set_xlabel("Mechanism Distance", fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel(rf"Normalized ${_HAT_E}$-SID", fontsize=_LABEL_FONTSIZE)
        ax.set_title(model, fontsize=_PANEL_TITLE_FONTSIZE)
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Shift Axis",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=len(labels),
            fontsize=_LEGEND_FONTSIZE,
            frameon=False,
        )

    fig.suptitle(
        "Mechanism Distance vs. Structural Error",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, agg)
    log.info("Saved distance-degradation scatter to %s", output_path)
    return agg


# =====================================================================
# 3. AUC vs Edge-F1 Divergence (RQ1)
# =====================================================================


def generate_auc_f1_divergence(
    raw_df: pd.DataFrame,
    output_path: Path,
    *,
    summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Scatter of edge-level AUROC vs. expected Edge-F1 per family.

    The diagonal represents perfect agreement between discrimination
    and structural recovery.  Points well above the diagonal indicate
    families where the model ranks edges correctly but cannot recover
    the structure (high AUC, low F1).  Points below indicate the
    reverse.  Coloured by shift axis.  Only fixed-size families.

    AUC is a family-level summary metric (not per-task), so if it is
    not present in *raw_df*, the function falls back to *summary_df*.
    """
    # --- Edge-F1 from raw per-task data ---
    f1_sub = raw_df[raw_df["Metric"].eq("e-edgef1")].copy()
    f1_sub = f1_sub[is_fixed_size_task_frame(f1_sub)]
    if f1_sub.empty:
        raise EmptyAnalysisDataError("No Edge-F1 data for AUC-F1 divergence.")

    f1_agg = (
        f1_sub.groupby(["Model", "DatasetKey", "AxisCategory"], dropna=False)["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Value": "e-edgef1"})
    )

    # --- AUC: try raw first, then summary ---
    auc_col = "auc"
    if auc_col in raw_df["Metric"].unique():
        auc_src = raw_df[raw_df["Metric"].eq(auc_col)].copy()
        auc_src = auc_src[is_fixed_size_task_frame(auc_src)]
        auc_agg = (
            auc_src.groupby(["Model", "DatasetKey", "AxisCategory"], dropna=False)[
                "Value"
            ]
            .mean()
            .reset_index()
            .rename(columns={"Value": "auc"})
        )
    elif summary_df is not None and auc_col in summary_df["Metric"].unique():
        auc_src = summary_df[summary_df["Metric"].eq(auc_col)].copy()
        auc_src = auc_src[is_fixed_size_task_frame(auc_src)]
        if auc_src.empty:
            raise EmptyAnalysisDataError("No AUC data in summary_df.")
        auc_agg = (
            auc_src[["Model", "DatasetKey", "AxisCategory", "Mean"]]
            .rename(columns={"Mean": "auc"})
            .copy()
        )
    else:
        raise EmptyAnalysisDataError("AUC not found in raw_df or summary_df.")

    # --- Merge ---
    pivot = f1_agg.merge(
        auc_agg, on=["Model", "DatasetKey", "AxisCategory"], how="inner"
    )
    pivot = pivot.dropna(subset=["e-edgef1", "auc"])
    if pivot.empty:
        raise EmptyAnalysisDataError("No complete data for AUC-F1 divergence.")

    models = [m for m in _canonical_model_order() if m in pivot["Model"].unique()]
    n_models = len(models)
    if n_models == 0:
        raise EmptyAnalysisDataError("No models for AUC-F1 divergence.")

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        mdf = pivot[pivot["Model"] == model]

        for cat in sorted(mdf["AxisCategory"].unique()):
            cdf = mdf[mdf["AxisCategory"] == cat]
            ax.scatter(
                cdf["auc"],
                cdf["e-edgef1"],
                c=_AXIS_COLORS.get(cat, "#aaaaaa"),
                label=_AXIS_DISPLAY.get(cat, cat),
                s=60,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )

        # Diagonal reference: AUC = F1.
        lims = [0, 1]
        ax.plot(lims, lims, "--", color="grey", alpha=0.5, linewidth=1)

        # Compute the mean gap: AUC - F1 for this model.
        gap = float((mdf["auc"] - mdf["e-edgef1"]).mean())
        ax.annotate(
            f"Mean gap (AUC$-$F1) = {gap:.3f}",
            xy=(0.03, 0.03),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

        ax.set_xlabel("Edge AUROC", fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel(rf"${_HAT_E}$-Edge F1", fontsize=_LABEL_FONTSIZE)
        ax.set_title(model, fontsize=_PANEL_TITLE_FONTSIZE)
        ax.set_xlim(0.35, 1.02)
        ax.set_ylim(-0.02, 0.85)
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Shift Axis",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=len(labels),
            fontsize=_LEGEND_FONTSIZE,
            frameon=False,
        )

    fig.suptitle(
        "Edge Discrimination (AUC) vs. Structural Recovery (F1)",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, pivot)
    log.info("Saved AUC-F1 divergence plot to %s", output_path)
    return pivot


# =====================================================================
# 4. Sparsity Dynamics Under Shift (RQ1 / RQ2)
# =====================================================================


def generate_sparsity_dynamics(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Grouped bar chart of mean sparsity ratio per model per shift axis.

    A sparsity ratio of 1.0 means the predicted graph has the same density
    as the ground truth.  Values > 1 indicate over-connection (more edges
    predicted than present), values < 1 indicate under-connection.
    """
    subset = raw_df[raw_df["Metric"].eq("sparsity_ratio")].copy()
    subset = subset[is_fixed_size_task_frame(subset)]
    if subset.empty:
        raise EmptyAnalysisDataError("No sparsity ratio data.")

    agg = (
        subset.groupby(["Model", "AxisCategory"], dropna=False)["Value"]
        .agg(Mean="mean", SEM=metric_sem)
        .reset_index()
    )

    axes_order = ["id", "graph", "mechanism", "noise", "compound"]
    axes_present = [a for a in axes_order if a in agg["AxisCategory"].unique()]
    models = [m for m in _canonical_model_order() if m in agg["Model"].unique()]

    if not models or not axes_present:
        raise EmptyAnalysisDataError("Insufficient sparsity data for dynamics plot.")

    n_axes = len(axes_present)
    n_models = len(models)
    x = np.arange(n_axes)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * n_axes * n_models), 5))

    for i, model in enumerate(models):
        mdf = agg[agg["Model"] == model]
        means = []
        sems = []
        for axis in axes_present:
            row = mdf[mdf["AxisCategory"] == axis]
            if row.empty:
                means.append(0.0)
                sems.append(0.0)
            else:
                means.append(float(row.iloc[0]["Mean"]))
                sems.append(float(row.iloc[0]["SEM"]))
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=sems,
            label=model,
            color=_model_color(model),
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
            alpha=0.85,
        )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.annotate(
        "ground truth density",
        xy=(n_axes - 0.5, 1.0),
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=8,
        color="black",
        alpha=0.7,
        ha="right",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([_AXIS_DISPLAY.get(a, a) for a in axes_present], fontsize=10)
    ax.set_ylabel("Mean Sparsity Ratio", fontsize=_LABEL_FONTSIZE)
    ax.set_title(
        "Predicted Graph Density Relative to Ground Truth",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.legend(fontsize=_LEGEND_FONTSIZE, loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, agg)
    log.info("Saved sparsity dynamics to %s", output_path)
    return agg


# =====================================================================
# 5. Accuracy–Calibration Joint Plot (RQ3)
# =====================================================================


def generate_accuracy_calibration_joint(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Scatter of ECE vs ne-SID per task family, one panel per model.

    Reveals whether calibration and accuracy co-degrade, or whether
    calibration can remain good even as structural recovery collapses.
    """
    from scipy.stats import spearmanr

    needed = {"ne-sid", "ece"}
    subset = raw_df[raw_df["Metric"].isin(needed)].copy()
    subset = subset[is_fixed_size_task_frame(subset)]
    if subset.empty:
        raise EmptyAnalysisDataError("No data for accuracy-calibration joint.")

    agg = (
        subset.groupby(["Model", "DatasetKey", "AxisCategory", "Metric"], dropna=False)[
            "Value"
        ]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=["Model", "DatasetKey", "AxisCategory"],
        columns="Metric",
        values="Value",
    ).reset_index()
    pivot.columns.name = None

    if "ece" not in pivot.columns or "ne-sid" not in pivot.columns:
        raise EmptyAnalysisDataError("Missing ECE or ne-SID for joint plot.")

    pivot = pivot.dropna(subset=["ece", "ne-sid"])
    models = [m for m in _canonical_model_order() if m in pivot["Model"].unique()]
    if not models:
        raise EmptyAnalysisDataError("No models for accuracy-calibration joint.")

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        mdf = pivot[pivot["Model"] == model]

        for cat in sorted(mdf["AxisCategory"].unique()):
            cdf = mdf[mdf["AxisCategory"] == cat]
            ax.scatter(
                cdf["ece"],
                cdf["ne-sid"],
                c=_AXIS_COLORS.get(cat, "#aaaaaa"),
                label=_AXIS_DISPLAY.get(cat, cat),
                s=60,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )

        if len(mdf) >= 3:
            rho, p_val = spearmanr(mdf["ece"], mdf["ne-sid"])
            if np.isfinite(rho):
                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                ax.annotate(
                    f"Spearman $\\rho$={rho:.2f}\n({p_str})",
                    xy=(0.97, 0.03),
                    xycoords="axes fraction",
                    fontsize=9,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

        ax.set_xlabel("Expected Calibration Error (ECE)", fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel(rf"Normalized ${_HAT_E}$-SID", fontsize=_LABEL_FONTSIZE)
        ax.set_title(model, fontsize=_PANEL_TITLE_FONTSIZE)
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Shift Axis",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=len(labels),
            fontsize=_LEGEND_FONTSIZE,
            frameon=False,
        )

    fig.suptitle(
        "Calibration Error vs. Structural Error",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, pivot)
    log.info("Saved accuracy-calibration joint to %s", output_path)
    return pivot


# =====================================================================
# 6. Per-Axis Family Win-Count Table (RQ2)
# =====================================================================


def generate_win_count_table(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """LaTeX table counting per-family ne-SID wins per model per shift axis.

    For each task family, the model with the lowest mean ne-SID is the
    ``winner''.  The table reports the number of family-level wins per
    shift axis for each model.
    """
    subset = raw_df[raw_df["Metric"].eq("ne-sid")].copy()
    if subset.empty:
        raise EmptyAnalysisDataError("No ne-SID data for win-count table.")

    # Compute family-level means per model.
    family_means = (
        subset.groupby(["Model", "DatasetKey", "AxisCategory"], dropna=False)["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Value": "MeanSID"})
    )

    # Identify winner per family.
    best_idx = family_means.groupby("DatasetKey")["MeanSID"].idxmin()
    winners = family_means.loc[best_idx].copy()

    # Count wins per (Model, AxisCategory).
    win_counts = (
        winners.groupby(["Model", "AxisCategory"]).size().reset_index(name="Wins")
    )

    axes_order = ["id", "graph", "mechanism", "noise", "compound", "nodes", "samples"]
    axes_present = [a for a in axes_order if a in win_counts["AxisCategory"].unique()]
    models = [m for m in _canonical_model_order() if m in win_counts["Model"].unique()]

    # Also compute total families per axis for context.
    total_families = (
        family_means.groupby("AxisCategory")["DatasetKey"].nunique().to_dict()
    )

    # Also compute overall totals.
    overall_wins = winners.groupby("Model").size().to_dict()

    # Build LaTeX table.
    col_spec = "l" + "c" * len(axes_present) + "c"
    lines = [
        r"\footnotesize",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Header.
    header_cells = [r"\textbf{Model}"]
    for axis in axes_present:
        n_fam = total_families.get(axis, "?")
        header_cells.append(
            r"\textbf{" + _AXIS_DISPLAY.get(axis, axis) + "}" + f" ({n_fam})"
        )
    header_cells.append(r"\textbf{Total}")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Identify per-axis best (most wins).
    axis_best: dict[str, int] = {}
    for axis in axes_present:
        axis_wins = win_counts[win_counts["AxisCategory"] == axis]["Wins"]
        axis_best[axis] = int(axis_wins.max()) if not axis_wins.empty else 0
    total_best = max(overall_wins.values()) if overall_wins else 0

    for model in models:
        cells = [model]
        for axis in axes_present:
            row = win_counts[
                (win_counts["Model"] == model) & (win_counts["AxisCategory"] == axis)
            ]
            wins = int(row.iloc[0]["Wins"]) if not row.empty else 0
            cell = str(wins)
            if wins == axis_best.get(axis, -1) and wins > 0:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)

        total = overall_wins.get(model, 0)
        total_cell = str(total)
        if total == total_best and total > 0:
            total_cell = r"\textbf{" + total_cell + "}"
        cells.append(total_cell)

        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n")
    log.info("Saved win-count table to %s", output_path)
    return win_counts


# =====================================================================
# 7. SID–SHD Metric Disagreement (RQ2)
# =====================================================================


def generate_metric_disagreement(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Scatter of family-level ne-SID rank vs. ne-SHD rank vs. E-Edge F1 per model.

    For each task family the five models are ranked on ne-SID, ne-SHD, and
    E-Edge F1.  The left panel shows how often the family-level *winner*
    disagrees between the metrics.  The right panel plots per-model
    Spearman correlation between each pair of ranking vectors, quantifying
    how reliably the metrics agree for each method.
    """
    from scipy.stats import spearmanr

    needed = {"ne-sid", "ne-shd", "e-edgef1"}
    subset = raw_df[raw_df["Metric"].isin(needed)].copy()
    if subset.empty:
        raise EmptyAnalysisDataError("No data for metric disagreement.")

    # Family-level means per model per metric.
    agg = (
        subset.groupby(["Model", "DatasetKey", "Metric"], dropna=False)["Value"]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=["Model", "DatasetKey"], columns="Metric", values="Value"
    ).reset_index()
    pivot.columns.name = None

    for col in ("ne-sid", "ne-shd", "e-edgef1"):
        if col not in pivot.columns:
            raise EmptyAnalysisDataError(f"Missing '{col}' for metric disagreement.")

    pivot = pivot.dropna(subset=["ne-sid", "ne-shd", "e-edgef1"])

    # Metric definitions: (column, display_label, lower_is_better)
    _METRICS = [
        ("ne-sid", rf"n${_HAT_E}$-SID", True),
        ("ne-shd", rf"n${_HAT_E}$-SHD", True),
        ("e-edgef1", rf"${_HAT_E}$-Edge F1", False),
    ]

    # Per-family: identify winner under each metric.
    families = sorted(pivot["DatasetKey"].unique())

    records: list[dict] = []
    for fam in families:
        fdf = pivot[pivot["DatasetKey"] == fam]
        if fdf.empty:
            continue
        row: dict = {"DatasetKey": fam}
        for col, label, lower_is_better in _METRICS:
            if lower_is_better:
                row[f"{col}_winner"] = fdf.loc[fdf[col].idxmin(), "Model"]
            else:
                row[f"{col}_winner"] = fdf.loc[fdf[col].idxmax(), "Model"]
        records.append(row)

    disagree_df = pd.DataFrame(records)
    total = len(disagree_df)

    # Count pairwise disagreement
    pairs = [("ne-sid", "ne-shd"), ("ne-sid", "e-edgef1"), ("ne-shd", "e-edgef1")]
    pair_disagree: dict[tuple[str, str], int] = {}
    for m1, m2 in pairs:
        pair_disagree[(m1, m2)] = int(
            (disagree_df[f"{m1}_winner"] != disagree_df[f"{m2}_winner"]).sum()
        )

    # Per-model Spearman ρ between metric pairs.
    models = [m for m in _canonical_model_order() if m in pivot["Model"].unique()]
    model_corrs: dict[str, dict[str, float]] = {m: {} for m in models}
    for model in models:
        mdf = pivot[pivot["Model"] == model].dropna(
            subset=["ne-sid", "ne-shd", "e-edgef1"]
        )
        for m1, m2 in pairs:
            if len(mdf) >= 3:
                rho, _ = spearmanr(mdf[m1], mdf[m2])
                model_corrs[model][f"{m1} vs {m2}"] = (
                    float(rho) if np.isfinite(rho) else np.nan
                )
            else:
                model_corrs[model][f"{m1} vs {m2}"] = np.nan

    # ── Figure: single panel (family-level winners) ──
    fig, ax = plt.subplots(figsize=COMPACT_SINGLE_FIGSIZE)

    metric_wins: dict[str, dict[str, int]] = {}
    for col, label, _ in _METRICS:
        wins: dict[str, int] = {}
        wcol = f"{col}_winner"
        for model in models:
            wins[model] = int((disagree_df[wcol] == model).sum())
        metric_wins[label] = wins

    x = np.arange(len(models))
    n_bars = len(_METRICS)
    w = 0.8 / n_bars
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for bar_idx, (col, label, _) in enumerate(_METRICS):
        offset = (bar_idx - n_bars / 2 + 0.5) * w
        vals = [metric_wins[label].get(m, 0) for m in models]
        ax.bar(
            x + offset,
            vals,
            w,
            label=f"Best on {label}",
            color=bar_colors[bar_idx],
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Number of families won", fontsize=10)
    sid_shd_dis = pair_disagree[("ne-sid", "ne-shd")]
    ax.set_title(
        f"Family-Level Winners\n(SID vs SHD disagree in {sid_shd_dis}/{total} families)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, disagree_df)
    log.info("Saved metric disagreement to %s", output_path)
    return disagree_df


# =====================================================================
# 7b. Metric vs DAG Accuracy (RQ2)
# =====================================================================


def generate_metric_dag_accuracy(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Scatter of AviCi DAG-validity gap vs. structural metrics (1x3 grid).

    The *validity gap* is defined as
    ``threshold_valid_dag_pct − valid_dag_pct`` per task family.  A large gap
    means the posterior-mean graph at p > 0.5 is almost always acyclic even
    though the individual posterior samples are often cyclic — i.e. the
    thresholded diagnostic hides substantial posterior uncertainty about
    cyclicity.

    Each dot is one task family (family-level mean), coloured by shift axis
    category.  The figure reveals whether families with a large validity gap
    are also the ones with worse structural accuracy.
    """
    needed = {
        "ne-sid",
        "ne-shd",
        "e-edgef1",
        "valid_dag_pct",
        "threshold_valid_dag_pct",
    }
    subset = raw_df[raw_df["Metric"].isin(needed) & raw_df["Model"].eq("AviCi")].copy()
    if subset.empty:
        raise EmptyAnalysisDataError("No AviCi data for metric-DAG accuracy plot.")

    agg = (
        subset.groupby(["Model", "DatasetKey", "AxisCategory", "Metric"], dropna=False)[
            "Value"
        ]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=["Model", "DatasetKey", "AxisCategory"],
        columns="Metric",
        values="Value",
    ).reset_index()
    pivot.columns.name = None

    for col in ("ne-sid", "ne-shd", "e-edgef1", "valid_dag_pct"):
        if col not in pivot.columns:
            raise EmptyAnalysisDataError(
                f"Missing '{col}' for metric-DAG accuracy plot."
            )

    has_threshold = "threshold_valid_dag_pct" in pivot.columns
    if not has_threshold:
        raise EmptyAnalysisDataError(
            "Missing 'threshold_valid_dag_pct' — cannot compute validity gap."
        )

    pivot = pivot.dropna(
        subset=[
            "ne-sid",
            "ne-shd",
            "e-edgef1",
            "valid_dag_pct",
            "threshold_valid_dag_pct",
        ]
    )
    if pivot.empty:
        raise EmptyAnalysisDataError("No data remaining for metric-DAG accuracy plot.")

    # Compute the validity gap (threshold − sampled).
    pivot["dag_validity_gap"] = (
        pivot["threshold_valid_dag_pct"] - pivot["valid_dag_pct"]
    )

    _METRIC_PANELS = [
        ("ne-sid", r"ne-SID $\downarrow$"),
        ("ne-shd", r"ne-SHD $\downarrow$"),
        ("e-edgef1", r"Edge F1 $\uparrow$"),
    ]

    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)

    for col_idx, (metric_col, metric_label) in enumerate(_METRIC_PANELS):
        ax = axes[0, col_idx]
        for cat in sorted(pivot["AxisCategory"].unique()):
            cdf = pivot[pivot["AxisCategory"] == cat]
            ax.scatter(
                cdf["dag_validity_gap"],
                cdf[metric_col],
                c=_AXIS_COLORS.get(cat, "#aaaaaa"),
                marker="o",
                label=(_AXIS_DISPLAY.get(cat, cat) if col_idx == 0 else None),
                s=40,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.3,
            )

        ax.set_xlabel(
            "DAG Validity Gap (Threshold − Sampled) %", fontsize=_LABEL_FONTSIZE
        )
        ax.set_ylabel(metric_label, fontsize=_LABEL_FONTSIZE)
        ax.set_title(
            f"Validity Gap vs. {metric_label}",
            fontsize=_PANEL_TITLE_FONTSIZE,
        )
        ax.grid(True, linestyle="--", alpha=0.4)

    # Compact legend: one entry per shift category.
    cat_handles = []
    for cat in sorted(pivot["AxisCategory"].unique()):
        cat_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=_AXIS_COLORS.get(cat, "#aaaaaa"),
                linestyle="None",
                markersize=7,
                label=_AXIS_DISPLAY.get(cat, cat),
            )
        )
    fig.legend(
        handles=cat_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(len(cat_handles), 10),
        fontsize=_LEGEND_FONTSIZE,
        frameon=False,
    )

    fig.suptitle(
        "AviCi: DAG Validity Gap vs. Structural Metrics",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, pivot)
    log.info("Saved metric-DAG accuracy plot to %s", output_path)
    return pivot


# =====================================================================
# 8. FP/FN Ratio Dynamics (RQ2)
# =====================================================================


def generate_fp_fn_ratio_dynamics(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Line plot of mean FP/(FP+FN) ratio per model across shift axes.

    A ratio near 1 means the model is false-positive-dominated
    (over-connecting); near 0 means false-negative-dominated
    (under-connecting).  Changes across axes reveal whether a model's
    error profile shifts qualitatively under different kinds of
    distribution shift.
    """
    needed = {"fp_count", "fn_count"}
    subset = raw_df[raw_df["Metric"].isin(needed)].copy()
    if subset.empty:
        raise EmptyAnalysisDataError("No FP/FN data for ratio dynamics.")

    # Per task, pivot FP and FN.
    task_agg = (
        subset.groupby(
            ["Model", "DatasetKey", "AxisCategory", "TaskIdx", "Metric"],
            dropna=False,
        )["Value"]
        .mean()
        .reset_index()
    )
    task_pivot = task_agg.pivot_table(
        index=["Model", "DatasetKey", "AxisCategory", "TaskIdx"],
        columns="Metric",
        values="Value",
    ).reset_index()
    task_pivot.columns.name = None

    if "fp_count" not in task_pivot.columns or "fn_count" not in task_pivot.columns:
        raise EmptyAnalysisDataError("Missing FP/FN columns for ratio dynamics.")

    task_pivot["total_errors"] = task_pivot["fp_count"] + task_pivot["fn_count"]
    task_pivot["fp_ratio"] = np.where(
        task_pivot["total_errors"] > 0,
        task_pivot["fp_count"] / task_pivot["total_errors"],
        np.nan,
    )

    # Aggregate per model × axis.
    agg = (
        task_pivot.groupby(["Model", "AxisCategory"], dropna=False)["fp_ratio"]
        .agg(Mean="mean", SEM=metric_sem)
        .reset_index()
    )

    axes_order = ["id", "graph", "mechanism", "noise", "compound", "nodes", "samples"]
    axes_present = [a for a in axes_order if a in agg["AxisCategory"].unique()]
    models = [m for m in _canonical_model_order() if m in agg["Model"].unique()]

    if not models or not axes_present:
        raise EmptyAnalysisDataError("Insufficient data for FP/FN ratio dynamics.")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x_positions = np.arange(len(axes_present))
    for model in models:
        mdf = agg[agg["Model"] == model]
        means = []
        sems = []
        for axis in axes_present:
            row = mdf[mdf["AxisCategory"] == axis]
            if row.empty:
                means.append(np.nan)
                sems.append(0.0)
            else:
                means.append(float(row.iloc[0]["Mean"]))
                sems.append(float(row.iloc[0]["SEM"]))
        ax.errorbar(
            x_positions,
            means,
            yerr=sems,
            marker=_model_marker(model),
            color=_model_color(model),
            label=model,
            linewidth=2,
            markersize=8,
            capsize=4,
            alpha=0.85,
        )

    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.annotate(
        "balanced",
        xy=(len(axes_present) - 0.5, 0.5),
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=8,
        color="black",
        alpha=0.6,
        ha="right",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([_AXIS_DISPLAY.get(a, a) for a in axes_present], fontsize=10)
    ax.set_ylabel("FP / (FP + FN) Ratio", fontsize=_LABEL_FONTSIZE)
    ax.set_title(
        "Error Profile Dynamics Across Shift Axes",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=_LEGEND_FONTSIZE, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, agg)
    log.info("Saved FP/FN ratio dynamics to %s", output_path)
    return agg


# =====================================================================
# 9. Per-Mechanism Model Ranking Heatmap (RQ1 / RQ2)
# =====================================================================


def generate_mechanism_heatmap(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Heatmap of mean ne-SID per model per mechanism type.

    Rows are mechanism types, columns are models.  The cell with the
    lowest ne-SID in each row is highlighted.  This answers the
    practical question: for a given functional class, which model
    should one prefer?
    """
    subset = raw_df[raw_df["Metric"].eq("ne-sid")].copy()
    subset = subset.dropna(subset=["MechType"])
    if subset.empty:
        raise EmptyAnalysisDataError("No ne-SID data with MechType for heatmap.")

    agg = (
        subset.groupby(["Model", "MechType"], dropna=False)["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Value": "MeanSID"})
    )

    models = [m for m in _canonical_model_order() if m in agg["Model"].unique()]
    mechs = sorted(agg["MechType"].unique())

    if not models or not mechs:
        raise EmptyAnalysisDataError("Insufficient data for mechanism heatmap.")

    # Pivot into matrix.
    matrix = agg.pivot_table(index="MechType", columns="Model", values="MeanSID")
    matrix = matrix.reindex(index=mechs, columns=models)

    # Row-wise best (lowest ne-SID).
    row_best = matrix.idxmin(axis=1)

    fig, ax = plt.subplots(
        figsize=(max(7, 1.2 * len(models)), max(4, 0.8 * len(mechs)))
    )

    import matplotlib.colors as mcolors

    # Use a diverging colormap centered on the median.
    vmin = float(matrix.min().min())
    vmax = float(matrix.max().max())
    cmap = plt.cm.RdYlGn_r  # Red = bad (high SID), Green = good (low SID)

    im = ax.imshow(
        matrix.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    # Annotate cells.
    for i, mech in enumerate(mechs):
        for j, model in enumerate(models):
            val = matrix.iloc[i, j]
            if np.isnan(val):
                continue
            is_best = row_best.get(mech) == model
            weight = "bold" if is_best else "normal"
            text_color = "white" if val > (vmin + vmax) / 2 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight=weight,
                color=text_color,
            )

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(mechs)))
    mech_labels = {
        "gp": "RFF",
        "linear": "Linear",
        "logistic": "Logistic",
        "mlp": "MLP",
        "periodic": "Periodic",
        "pnl": "PNL",
        "square": "Quadratic",
    }
    ax.set_yticklabels([mech_labels.get(m, m) for m in mechs], fontsize=9)
    ax.set_title(
        rf"Mean Normalized ${_HAT_E}$-SID by Mechanism Type",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"ne-SID $\downarrow$", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, matrix)
    log.info("Saved mechanism heatmap to %s", output_path)
    return matrix


# =====================================================================
# 10. Orientation vs. Skeleton Gap (RQ1 / RQ2)
# =====================================================================


def generate_orientation_skeleton_gap(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Grouped bar of skeleton F1 and orientation accuracy per model per axis.

    Each model shows two bars per axis: skeleton F1 (undirected edge
    recovery) and orientation accuracy (fraction of correctly directed
    skeleton edges).  The gap between them isolates how much error
    comes from directionality alone.
    """
    needed = {"skeleton_f1", "orientation_accuracy"}
    subset = raw_df[raw_df["Metric"].isin(needed)].copy()
    subset = subset[is_fixed_size_task_frame(subset)]
    if subset.empty:
        raise EmptyAnalysisDataError("No data for orientation-skeleton gap.")

    agg = (
        subset.groupby(["Model", "AxisCategory", "Metric"], dropna=False)["Value"]
        .agg(Mean="mean", SEM=metric_sem)
        .reset_index()
    )

    axes_order = ["id", "graph", "mechanism", "noise", "compound"]
    axes_present = [a for a in axes_order if a in agg["AxisCategory"].unique()]
    models = [m for m in _canonical_model_order() if m in agg["Model"].unique()]

    if not models or not axes_present:
        raise EmptyAnalysisDataError("Insufficient data for orientation-skeleton gap.")

    n_models = len(models)
    fig, axes_arr = plt.subplots(
        1, n_models, figsize=(5 * n_models, 5), squeeze=False, sharey=True
    )

    metric_colors = {"skeleton_f1": "#1f77b4", "orientation_accuracy": "#ff7f0e"}
    metric_labels = {
        "skeleton_f1": "Skeleton F1",
        "orientation_accuracy": "Orientation Acc.",
    }

    for idx, model in enumerate(models):
        ax = axes_arr[0, idx]
        mdf = agg[agg["Model"] == model]

        x = np.arange(len(axes_present))
        w = 0.35

        for k, metric in enumerate(["skeleton_f1", "orientation_accuracy"]):
            means = []
            sems = []
            for axis in axes_present:
                row = mdf[(mdf["AxisCategory"] == axis) & (mdf["Metric"] == metric)]
                if row.empty:
                    means.append(0.0)
                    sems.append(0.0)
                else:
                    means.append(float(row.iloc[0]["Mean"]))
                    sems.append(float(row.iloc[0]["SEM"]))

            offset = (k - 0.5) * w
            ax.bar(
                x + offset,
                means,
                w,
                yerr=sems,
                label=metric_labels[metric] if idx == 0 else None,
                color=metric_colors[metric],
                edgecolor="white",
                linewidth=0.5,
                capsize=3,
                alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_AXIS_DISPLAY.get(a, a) for a in axes_present], fontsize=9, rotation=15
        )
        ax.set_title(model, fontsize=_PANEL_TITLE_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 1.0)

    axes_arr[0, 0].set_ylabel("Score", fontsize=_LABEL_FONTSIZE)

    # Shared legend.
    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=2,
            fontsize=_LEGEND_FONTSIZE,
            frameon=False,
        )

    fig.suptitle(
        "Skeleton Recovery vs. Edge Orientation Accuracy",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, agg)
    log.info("Saved orientation-skeleton gap to %s", output_path)
    return agg


# =====================================================================
# Public driver
# =====================================================================


def generate_all_deep_results(
    raw_df: pd.DataFrame,
    output_dir: Path,
    *,
    summary_df: pd.DataFrame | None = None,
    strict: bool = True,
) -> None:
    """Run all deep-dive analyses and write outputs to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses: list[tuple[str, object, str]] = [
        (
            "efficiency frontier",
            generate_efficiency_frontier,
            "efficiency_frontier.pdf",
        ),
        (
            "distance-degradation scatter",
            generate_distance_degradation_scatter,
            "distance_degradation_scatter.pdf",
        ),
        ("sparsity dynamics", generate_sparsity_dynamics, "sparsity_dynamics.pdf"),
        (
            "accuracy-calibration joint",
            generate_accuracy_calibration_joint,
            "accuracy_calibration_joint.pdf",
        ),
        ("win-count table", generate_win_count_table, "win_counts.tex"),
        (
            "metric disagreement",
            generate_metric_disagreement,
            "metric_disagreement.pdf",
        ),
        (
            "metric-DAG accuracy",
            generate_metric_dag_accuracy,
            "rq2_metric_dag_accuracy.pdf",
        ),
        (
            "FP/FN ratio dynamics",
            generate_fp_fn_ratio_dynamics,
            "fp_fn_ratio_dynamics.pdf",
        ),
        (
            "mechanism heatmap",
            generate_mechanism_heatmap,
            "mechanism_heatmap.pdf",
        ),
        (
            "orientation-skeleton gap",
            generate_orientation_skeleton_gap,
            "orientation_skeleton_gap.pdf",
        ),
    ]

    for label, func, filename in analyses:
        try:
            func(raw_df, output_dir / filename)
        except Exception:
            if strict:
                raise
            log.warning("Deep-result '%s' failed.", label, exc_info=True)

    # AUC-F1 divergence needs summary_df for the AUC metric.
    try:
        generate_auc_f1_divergence(
            raw_df, output_dir / "auc_f1_divergence.pdf", summary_df=summary_df
        )
    except Exception:
        if strict:
            raise
        log.warning("Deep-result 'AUC-F1 divergence' failed.", label, exc_info=True)
