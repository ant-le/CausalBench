from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from causal_meta.analysis.common.plot_style import (
    HAT_E,
    LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    SUBTITLE_FONTSIZE,
    TITLE_FONTSIZE,
    TOP_LEGEND_ANCHOR,
    TOP_LEGEND_RECT,
)
from causal_meta.analysis.diagnostics.failure_modes import ood_category as _ood_category
from causal_meta.analysis.plots.utils import draw_point_plot
from causal_meta.analysis.utils import MODEL_COLORS, MODEL_MARKERS

log = logging.getLogger(__name__)

_HAT_E_TEXT = rf"${HAT_E}$"
_TITLE_FONTSIZE = TITLE_FONTSIZE
_PANEL_TITLE_FONTSIZE = SUBTITLE_FONTSIZE
_LABEL_FONTSIZE = LABEL_FONTSIZE
_LEGEND_FONTSIZE = LEGEND_FONTSIZE


def _model_color(model: str) -> str:
    """Return the canonical colour for *model*, falling back to grey."""
    return MODEL_COLORS.get(model, "#555555")


def generate_structural_figure(df: pd.DataFrame, output_path: Path) -> None:
    # ---------------------------------------------------------
    # 1) Structural Metrics (ne-shd, ne-sid) - 1 Row, 2 Columns
    # ---------------------------------------------------------
    structural_metrics = [
        (
            "ne-shd",
            rf"Normalized {_HAT_E_TEXT}-SHD",
            "Level 1: Graph Structure (normalized SHD) ↓",
        ),
        (
            "ne-sid",
            rf"Normalized {_HAT_E_TEXT}-SID",
            "Level 2: Interventional Accuracy (normalized SID) ↓",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

    axes_flat = axes.flatten()

    for idx, (metric_id, ylabel, title) in enumerate(structural_metrics):
        draw_point_plot(
            axes_flat[idx],
            df,
            metric_id,
            ylabel,
            title,
            log_scale=False,  # Use linear scale as requested
            show_legend=False,  # We will add a shared legend
        )

    # Shared legend for Figure 1 - Top Center, No Border
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Model",
            loc="center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(labels),
            fontsize=12,
            frameon=False,  # Remove border
        )

    # Set axis limits per metric independently (SHD and SID are on different scales).
    for idx, (metric_id, _, _) in enumerate(structural_metrics):
        metric_df = df[df["Metric"] == metric_id]
        if metric_df.empty:
            continue

        means = pd.to_numeric(metric_df["Mean"], errors="coerce")
        if "SEM" in metric_df.columns:
            sems = pd.to_numeric(metric_df["SEM"], errors="coerce").fillna(0.0)
        else:
            sems = pd.Series(0.0, index=metric_df.index)
        lower = (means - sems).dropna()
        upper = (means + sems).dropna()
        if lower.empty or upper.empty:
            continue

        min_val = float(lower.min())
        max_val = float(upper.max())
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            continue

        if np.isclose(min_val, max_val):
            pad = max(1.0, abs(max_val) * 0.1)
            axes_flat[idx].set_ylim(min_val - pad, max_val + pad)
            continue

        span = max_val - min_val
        axes_flat[idx].set_ylim(min_val - 0.1 * span, max_val + 0.1 * span)

    # Reserve top 10% for legend
    plt.tight_layout(rect=(0, 0, 1, 0.9))

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    log.info("Saved %s", output_path)


def generate_performance_figure(df: pd.DataFrame, output_path: Path) -> None:
    # ---------------------------------------------------------
    # 2) Other Metrics - Grid Layout (2 Rows, 3 Columns)
    # ---------------------------------------------------------
    other_metrics = [
        ("auc", "AUC", "Causal Discovery AUC ↑"),
        (
            "graph_nll_per_edge",
            "Graph NLL / edge",
            "Graph Negative Log-Likelihood per edge ↓",
        ),
        ("ancestor_f1", "Ancestor F1", "Ancestor F1 Score ↑"),
        ("e-edgef1", "Edge F1", "Expected Edge F1 Score ↑"),
        ("edge_entropy", "Entropy", "Edge Entropy ↓"),
        (
            "valid_dag_pct",
            "Valid DAG Samples (%)",
            "Valid DAG Samples (%) ↑",
        ),
        (
            "threshold_valid_dag_pct",
            "Thresholded DAG (%)",
            "Thresholded Mean-Graph DAG (%) ↑",
        ),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes_flat = axes.flatten()

    for idx, (metric_id, ylabel, title) in enumerate(other_metrics):
        log_scale = metric_id == "graph_nll_per_edge"
        draw_point_plot(
            axes_flat[idx],
            df,
            metric_id,
            ylabel,
            title,
            log_scale=log_scale,
            show_legend=False,
        )

    # Shared legend for the full figure.
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Model",
            loc="center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=len(labels),
            fontsize=12,
            frameon=False,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    log.info("Saved %s", output_path)


# ── Helper: pivot long-format DF to wide per-dataset rows ──────────────


def _pivot_metrics(
    df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Pivot the long-format DataFrame so each metric becomes its own column.

    Returns one row per (RunID, Model, DatasetKey) with columns for each
    metric's Mean value plus the enrichment columns (GraphType, SpectralDist, etc.).
    """
    if df.empty or "Metric" not in df.columns:
        return pd.DataFrame()
    subset = df[df["Metric"].isin(metrics)].copy()
    if subset.empty:
        return pd.DataFrame()

    # Enrich columns that are constant per (RunID, DatasetKey)
    enrich_cols = [
        c
        for c in (
            "RunID",
            "RunName",
            "Model",
            "ModelKey",
            "Dataset",
            "DatasetKey",
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

    pivoted = subset.pivot_table(
        index=enrich_cols,
        columns="Metric",
        values="Mean",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None
    return pivoted


# ── E.2  Entropy-vs-accuracy scatter (S2: Posterior Calibration) ────────


def generate_calibration_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot of edge_entropy (x) vs ne-shd (y) per model, coloured by OOD category.

    An ideal calibrated model shows high entropy → high normalized SHD and low
    entropy → low normalized SHD. Overconfident OOD predictions cluster at
    low-entropy / high-error regions.
    """
    wide = _pivot_metrics(df, ["edge_entropy", "ne-shd"])
    if wide.empty or "edge_entropy" not in wide.columns or "ne-shd" not in wide.columns:
        log.warning("Insufficient data for calibration scatter; skipping.")
        return

    wide["OODCategory"] = wide["DatasetKey"].apply(_ood_category)

    models = sorted(wide["Model"].unique())
    n_models = len(models)
    if n_models == 0:
        return

    category_colors = {
        "ID": "#2ca02c",
        "OOD-Graph": "#d62728",
        "OOD-Mech": "#ff7f0e",
        "OOD-Both": "#9467bd",
        "OOD": "#8c564b",
    }

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
    axes_flat: list[Axes] = list(axes.flatten())

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        mdf = wide[wide["Model"] == model]
        for cat, colour in category_colors.items():
            cdf = mdf[mdf["OODCategory"] == cat]
            if cdf.empty:
                continue
            ax.scatter(
                cdf["edge_entropy"],
                cdf["ne-shd"],
                label=cat,
                color=colour,
                s=60,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.3,
            )
        ax.set_xlabel("Edge Entropy", fontsize=11)
        ax.set_ylabel(rf"Normalized {_HAT_E_TEXT}-SHD ↓", fontsize=_LABEL_FONTSIZE)
        ax.set_title(model, fontsize=_PANEL_TITLE_FONTSIZE, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=TOP_LEGEND_ANCHOR,
            ncol=len(labels),
            fontsize=10,
            frameon=False,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved calibration scatter to %s", output_path)


# ── E.4  Distance-vs-degradation scatter (S5) ──────────────────────────


def generate_distance_degradation_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter of spectral distance (x) vs normalized E-SID degradation (y).

    One series per model with a linear trend line.  This is the key figure that
    upgrades the analysis from categorical to quantitative.
    """
    wide = _pivot_metrics(df, ["ne-sid"])
    if wide.empty or "ne-sid" not in wide.columns:
        log.warning("Insufficient data for distance-degradation scatter; skipping.")
        return

    if "SpectralDist" not in wide.columns:
        log.warning("No SpectralDist column; skipping distance-degradation scatter.")
        return

    wide["OODCategory"] = wide["DatasetKey"].apply(_ood_category)

    group_cols = [c for c in ("RunID", "Model") if c in wide.columns]
    if not group_cols:
        group_cols = ["Model"]

    # Compute per-run/per-model ID baseline (mean normalized E-SID across ID datasets)
    id_means = (
        wide[wide["OODCategory"] == "ID"]
        .groupby(group_cols)["ne-sid"]
        .mean()
        .rename("id_baseline")
    )
    wide = wide.merge(id_means, on=group_cols, how="left")
    wide["degradation"] = wide["ne-sid"] - wide["id_baseline"]

    # OOD-only fit/scatter (ID rows define the baseline and are not fit points).
    plot_df = wide[wide["OODCategory"] != "ID"].dropna(
        subset=["SpectralDist", "degradation"]
    )
    if plot_df.empty:
        log.warning("No valid rows for distance-degradation scatter; skipping.")
        return

    if "RunID" in group_cols:
        series_keys: list[tuple[str, str]] = sorted(
            {
                (str(rid), str(model))
                for rid, model in plot_df[["RunID", "Model"]].values
            }
        )
    else:
        series_keys = [("", str(model)) for model in sorted(plot_df["Model"].unique())]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (run_id, model) in enumerate(series_keys):
        if "RunID" in group_cols:
            mdf = plot_df[(plot_df["RunID"] == run_id) & (plot_df["Model"] == model)]
            label = f"{model} [{run_id}]"
        else:
            mdf = plot_df[plot_df["Model"] == model]
            label = model

        if mdf.empty:
            continue

        colour = _model_color(model)
        ax.scatter(
            mdf["SpectralDist"],
            mdf["degradation"],
            label=label,
            color=colour,
            s=60,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
        )
        # Trend line
        x = mdf["SpectralDist"].to_numpy(dtype=float)
        y = mdf["degradation"].to_numpy(dtype=float)
        if len(x) >= 2:
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(float(x.min()), float(x.max()), 50)
            ax.plot(
                x_line,
                np.polyval(coeffs, x_line),
                color=colour,
                linestyle="--",
                alpha=0.6,
            )

    ax.set_xlabel("Spectral Distance from Training Distribution", fontsize=12)
    ax.set_ylabel(
        rf"Normalized {_HAT_E_TEXT}-SID Degradation (vs. ID baseline) ↑ = worse",
        fontsize=_LABEL_FONTSIZE,
    )
    ax.set_title(
        "Shift Distance vs. OOD Degradation",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    legend_title = "Run/Model" if "RunID" in group_cols else "Model"
    ax.legend(title=legend_title, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved distance-degradation scatter to %s", output_path)


# ── E.6  Density-stratified normalized E-SID plot (S6) ─────────────────


def generate_density_stratified_figure(df: pd.DataFrame, output_path: Path) -> None:
    """Plot normalized E-SID vs sparsity level for each model.

    Uses the ER families that vary sparsity (ER20/40/60) at fixed n_nodes=20.
    One line per model, x-axis = SparsityParam.
    """
    wide = _pivot_metrics(df, ["ne-sid"])
    if wide.empty or "ne-sid" not in wide.columns:
        log.warning("Insufficient data for density-stratified plot; skipping.")
        return

    if "SparsityParam" not in wide.columns:
        log.warning("No SparsityParam column; skipping density-stratified plot.")
        return

    # Filter to ER families only (those with non-null sparsity and graph_type == "er")
    er_df = wide[
        wide["SparsityParam"].notna() & (wide["GraphType"].str.lower() == "er")
    ].copy()

    if er_df.empty:
        log.warning("No ER families with sparsity data; skipping density plot.")
        return

    models = sorted(er_df["Model"].unique())

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, model in enumerate(models):
        mdf = er_df[er_df["Model"] == model].sort_values("SparsityParam")
        # Group by sparsity level in case there are multiple mech types
        grouped = (
            mdf.groupby("SparsityParam")["ne-sid"].agg(["mean", "sem"]).reset_index()
        )
        colour = _model_color(model)
        marker = MODEL_MARKERS.get(model, "o")
        ax.errorbar(
            grouped["SparsityParam"],
            grouped["mean"],
            yerr=grouped["sem"],
            label=model,
            color=colour,
            marker=marker,
            capsize=3,
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel(
        "Sparsity Parameter (edge probability / expected edges)",
        fontsize=_LABEL_FONTSIZE,
    )
    ax.set_ylabel(rf"Normalized {_HAT_E_TEXT}-SID ↓", fontsize=_LABEL_FONTSIZE)
    ax.set_title(
        "Performance vs. Graph Density (ER families)",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Model", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved density-stratified figure to %s", output_path)


# ── E.1  Failure mode stacked bar chart (S1) ───────────────────────────


def generate_failure_mode_bar(
    fractions_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grouped stacked bar chart: x = OOD condition, colour = failure mode.

    Args:
        fractions_df: Output of
            :func:`~causal_meta.analysis.failure_modes.failure_mode_fractions`
            with columns ``Model``, ``DatasetKey``, and one column per failure
            mode category (values in [0, 1]).
        output_path: Where to save the figure.
    """
    from causal_meta.analysis.diagnostics.failure_modes import (
        FAILURE_MODE_CATEGORIES,
        FAILURE_MODE_COLORS,
        ood_category,
    )

    if fractions_df.empty:
        log.warning("Empty fractions DataFrame; skipping failure-mode bar chart.")
        return

    if "DatasetKey" not in fractions_df.columns or "Model" not in fractions_df.columns:
        log.warning("Missing DatasetKey/Model columns; skipping failure-mode bar.")
        return

    # Ensure all category columns exist
    for cat in FAILURE_MODE_CATEGORIES:
        if cat not in fractions_df.columns:
            fractions_df[cat] = 0.0

    fractions_df = fractions_df.copy()
    fractions_df["OODCategory"] = fractions_df["DatasetKey"].apply(ood_category)

    # Aggregate: mean fractions over datasets in the same OOD category per model
    agg_cols = FAILURE_MODE_CATEGORIES
    grouped = (
        fractions_df.groupby(["Model", "OODCategory"])[agg_cols].mean().reset_index()
    )

    models = sorted(grouped["Model"].unique())
    categories = sorted(grouped["OODCategory"].unique())
    n_models = len(models)
    n_categories = len(categories)

    if n_models == 0 or n_categories == 0:
        log.warning("No data for failure-mode bar chart; skipping.")
        return

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_categories * n_models), 6))

    bar_width = 0.8 / n_models
    x_base = np.arange(n_categories)

    for model_idx, model in enumerate(models):
        mdf = grouped[grouped["Model"] == model].set_index("OODCategory")
        # Reindex to ensure all categories are present
        mdf = mdf.reindex(categories).fillna(0.0)

        bottoms = np.zeros(n_categories)
        x_pos = x_base + model_idx * bar_width

        for cat in FAILURE_MODE_CATEGORIES:
            heights = mdf[cat].to_numpy(dtype=float)
            ax.bar(
                x_pos,
                heights,
                bar_width,
                bottom=bottoms,
                label=f"{model} - {cat}" if model_idx == 0 else cat,
                color=FAILURE_MODE_COLORS.get(cat, "#999999"),
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += heights

    # X-axis: category labels centered under group
    ax.set_xticks(x_base + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Fraction of Test Tasks", fontsize=12)
    ax.set_title(
        "Failure Mode Distribution by OOD Condition",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Build legend: model patches (gray tones) + failure-mode patches (colors)
    from matplotlib.patches import Patch

    legend_handles = []
    # Failure mode colors
    for cat in FAILURE_MODE_CATEGORIES:
        legend_handles.append(
            Patch(facecolor=FAILURE_MODE_COLORS.get(cat, "#999"), label=cat)
        )
    # Model labels as text annotations in the bar groups
    if n_models > 1:
        for i, model in enumerate(models):
            legend_handles.append(
                Patch(
                    facecolor="white",
                    edgecolor="black",
                    label=f"Group {i + 1}: {model}",
                )
            )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved failure-mode bar chart to %s", output_path)


# ── Greyscale per-model failure mode bar ───────────────────────────────

# Greyscale palette for failure mode categories (ordered lightest→darkest).
_FAILURE_MODE_GREYS: dict[str, str] = {
    "reasonable": "#d9d9d9",
    "sparse": "#a6a6a6",
    "reversed": "#737373",
    "dense": "#404040",
    "empty": "#1a1a1a",
}


def generate_per_model_failure_mode_bar(
    fractions_df: pd.DataFrame,
    *,
    model: str,
    output_path: Path,
) -> None:
    """Greyscale stacked bar chart for a single model's failure modes.

    Produces one figure per amortised model.  The legend is placed above the
    plot and font sizes are increased for thesis readability.

    Args:
        fractions_df: Output of
            :func:`~causal_meta.analysis.failure_modes.failure_mode_fractions`
            with columns ``Model``, ``DatasetKey``, and one column per failure
            mode category (values in [0, 1]).
        model: The display name of the model to plot (e.g. ``"AviCi"``).
        output_path: Where to save the figure.
    """
    from matplotlib.patches import Patch

    from causal_meta.analysis.diagnostics.failure_modes import (
        FAILURE_MODE_CATEGORIES,
        ood_category,
    )

    if fractions_df.empty:
        log.warning("Empty fractions DataFrame; skipping per-model failure bar.")
        return

    model_df = fractions_df[fractions_df["Model"] == model].copy()
    if model_df.empty:
        log.warning("No failure-mode data for model '%s'; skipping.", model)
        return

    for cat in FAILURE_MODE_CATEGORIES:
        if cat not in model_df.columns:
            model_df[cat] = 0.0

    model_df["OODCategory"] = model_df["DatasetKey"].apply(ood_category)

    agg = model_df.groupby("OODCategory")[FAILURE_MODE_CATEGORIES].mean().reset_index()
    categories = sorted(agg["OODCategory"].unique())
    n_categories = len(categories)

    if n_categories == 0:
        log.warning("No OOD categories for model '%s'; skipping.", model)
        return

    fig, ax = plt.subplots(figsize=(max(7, 2 * n_categories), 5))

    x = np.arange(n_categories)
    bar_width = 0.55
    agg_indexed = agg.set_index("OODCategory").reindex(categories).fillna(0.0)

    bottoms = np.zeros(n_categories)
    for cat in FAILURE_MODE_CATEGORIES:
        heights = agg_indexed[cat].to_numpy(dtype=float)
        ax.bar(
            x,
            heights,
            bar_width,
            bottom=bottoms,
            color=_FAILURE_MODE_GREYS.get(cat, "#999999"),
            edgecolor="white",
            linewidth=0.6,
        )
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Fraction of Test Tasks", fontsize=_LABEL_FONTSIZE)
    ax.set_title(model, fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Legend above the axes
    handles = [
        Patch(facecolor=_FAILURE_MODE_GREYS.get(cat, "#999"), label=cat)
        for cat in FAILURE_MODE_CATEGORIES
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=TOP_LEGEND_ANCHOR,
        ncol=len(FAILURE_MODE_CATEGORIES),
        fontsize=11,
        frameon=False,
    )

    fig.tight_layout(rect=TOP_LEGEND_RECT)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved per-model failure-mode bar for '%s' to %s", model, output_path)


# ── E.3a  Entropy histogram: ID vs OOD (S4) ────────────────────────────


def generate_entropy_histogram(
    raw_df: pd.DataFrame,
    output_path: Path,
    score_metric: str = "edge_entropy",
) -> None:
    """Overlaid histograms of per-task entropy for ID vs OOD tasks, per model.

    Args:
        raw_df: Long-format per-task DataFrame from ``load_raw_task_dataframe``.
        output_path: Where to save the figure.
        score_metric: The uncertainty metric to histogram.
    """
    if raw_df.empty or "Metric" not in raw_df.columns:
        log.warning("Empty raw DataFrame; skipping entropy histogram.")
        return

    score_rows = raw_df[raw_df["Metric"] == score_metric].copy()
    if score_rows.empty:
        log.warning("No '%s' data for histogram; skipping.", score_metric)
        return

    if "DatasetKey" not in score_rows.columns:
        return

    score_rows["is_ood"] = score_rows["DatasetKey"].apply(
        lambda dk: (
            "OOD"
            if not (dk.lower().startswith("id_") or dk.lower() == "id_test")
            else "ID"
        )
    )

    models = sorted(score_rows["Model"].unique())
    n_models = len(models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)
    axes_flat = list(axes.flatten())

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        mdf = score_rows[score_rows["Model"] == model]

        id_vals = mdf[mdf["is_ood"] == "ID"]["Value"].dropna().to_numpy(dtype=float)
        ood_vals = mdf[mdf["is_ood"] == "OOD"]["Value"].dropna().to_numpy(dtype=float)

        bins = 30
        if len(id_vals) > 0:
            ax.hist(
                id_vals, bins=bins, alpha=0.6, label="ID", color="#2ca02c", density=True
            )
        if len(ood_vals) > 0:
            ax.hist(
                ood_vals,
                bins=bins,
                alpha=0.6,
                label="OOD",
                color="#d62728",
                density=True,
            )

        ax.set_xlabel(score_metric.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(model, fontsize=_PANEL_TITLE_FONTSIZE, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Per-Task {score_metric.replace('_', ' ').title()} Distribution: ID vs OOD",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved entropy histogram to %s", output_path)


# ── E.3b  Selective prediction Pareto curve (S4) ───────────────────────


def generate_selective_prediction_pareto(
    pareto_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot accuracy vs coverage for each model under selective prediction.

    Args:
        pareto_df: Output of
            :func:`~causal_meta.analysis.ood_detection.compute_selective_prediction`
            with columns ``Model``, ``Coverage``, ``MeanValue`` and
            ``AccuracyMetric`` (plus ``RunID`` when available).
        output_path: Where to save the figure.
    """
    if pareto_df.empty:
        log.warning("Empty Pareto DataFrame; skipping selective prediction plot.")
        return

    if "Coverage" not in pareto_df.columns:
        log.warning("Missing Coverage column; skipping Pareto plot.")
        return

    value_col = "MeanValue" if "MeanValue" in pareto_df.columns else "MeanAccuracy"
    if value_col not in pareto_df.columns:
        log.warning("Missing MeanValue/MeanAccuracy columns; skipping Pareto plot.")
        return

    df = pareto_df.copy()
    if "AccuracyMetric" not in df.columns:
        df["AccuracyMetric"] = "ne-shd"

    metric_order = sorted(df["AccuracyMetric"].dropna().unique())
    if not metric_order:
        log.warning("No accuracy metrics found for Pareto plot; skipping.")
        return

    series_cols = [c for c in ("RunID", "Model") if c in df.columns]
    if not series_cols:
        series_cols = ["Model"]

    def _series_label(group_key: object) -> str:
        if not isinstance(group_key, tuple):
            if len(series_cols) == 1 and series_cols[0] == "Model":
                return str(group_key)
            return str(group_key)
        key_map = {k: v for k, v in zip(series_cols, group_key)}
        model = str(key_map.get("Model", "unknown"))
        run_id = key_map.get("RunID")
        return f"{model} [{run_id}]" if run_id is not None else model

    all_series_keys = list(df.groupby(series_cols, dropna=False).groups.keys())
    all_labels = [_series_label(k) for k in all_series_keys]

    # Build colour map from MODEL_COLORS, falling back for unknown labels
    def _label_to_model(label: str) -> str:
        for model_name in MODEL_COLORS:
            if label == model_name or label.startswith(f"{model_name} ["):
                return model_name
        return label

    color_map = {
        label: _model_color(_label_to_model(label)) for label in sorted(set(all_labels))
    }

    fig, axes = plt.subplots(
        1,
        len(metric_order),
        figsize=(8 * len(metric_order), 6),
        squeeze=False,
    )

    for axis_idx, metric in enumerate(metric_order):
        ax = axes[0, axis_idx]
        metric_df = df[df["AccuracyMetric"] == metric]
        if metric_df.empty:
            ax.set_visible(False)
            continue

        for group_key, series_df in metric_df.groupby(series_cols, dropna=False):
            label = _series_label(group_key)
            mdf = series_df.sort_values("Coverage")
            ax.plot(
                mdf["Coverage"],
                mdf[value_col],
                label=label,
                color=color_map[label],
                linewidth=2,
                marker=".",
                markersize=4,
            )

        metric_label = metric.replace("-", "-").upper()
        if metric == "ne-shd":
            y_label = rf"Mean normalized {_HAT_E_TEXT}-SHD of accepted predictions ↓"
        elif metric == "ne-sid":
            y_label = rf"Mean normalized {_HAT_E_TEXT}-SID of accepted predictions ↓"
        elif metric == "e-shd":
            y_label = rf"Mean {_HAT_E_TEXT}-SHD of accepted predictions ↓"
        elif metric == "e-sid":
            y_label = rf"Mean {_HAT_E_TEXT}-SID of accepted predictions ↓"
        else:
            y_label = f"Mean {metric} of accepted predictions"

        ax.set_xlabel("Coverage (fraction of predictions accepted)", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(metric_label, fontsize=_PANEL_TITLE_FONTSIZE, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Shared legend on top.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Model",
            loc="upper center",
            bbox_to_anchor=TOP_LEGEND_ANCHOR,
            ncol=len(labels),
            fontsize=10,
            frameon=False,
        )

    fig.suptitle(
        "Selective Prediction: Accuracy vs Coverage",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
        y=1.06,
    )

    fig.tight_layout(rect=TOP_LEGEND_RECT)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved selective prediction Pareto to %s", output_path)


# ── Posterior Failure Diagnostics plots (Section 1) ─────────────────────


def generate_event_probability_bar(
    diagnostics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grouped bar chart of posterior event probabilities by OOD condition.

    One group per OOD category, one bar per event type, faceted by model.

    Args:
        diagnostics_df: Output of
            :func:`~causal_meta.analysis.posterior_diagnostics.run_posterior_diagnostics`
            with columns ``Model``, ``DatasetKey``, ``p_empty``, ``p_dense``,
            ``p_skeleton_correct_orient_wrong``, ``p_fragmented``.
        output_path: Where to save the figure.
    """
    event_cols = [
        "p_empty",
        "p_dense",
        "p_skeleton_correct_orient_wrong",
        "p_fragmented",
    ]
    event_labels = {
        "p_empty": "P(empty)",
        "p_dense": "P(dense)",
        "p_skeleton_correct_orient_wrong": "P(skel ok, orient wrong)",
        "p_fragmented": "P(fragmented | truth connected)",
    }
    event_colors = {
        "p_empty": "#1f77b4",
        "p_dense": "#d62728",
        "p_skeleton_correct_orient_wrong": "#ff7f0e",
        "p_fragmented": "#9467bd",
    }

    if diagnostics_df.empty:
        log.warning("Empty diagnostics DataFrame; skipping event probability bar.")
        return

    for col in event_cols:
        if col not in diagnostics_df.columns:
            log.warning("Missing column %s; skipping event probability bar.", col)
            return

    df = diagnostics_df.copy()
    df["OODCategory"] = df["DatasetKey"].apply(_ood_category)

    # Average event probabilities per (Model, OODCategory)
    grouped = df.groupby(["Model", "OODCategory"])[event_cols].mean().reset_index()

    if "TruthConnected" in df.columns:
        connected_stats = (
            df.groupby(["Model", "OODCategory"])["TruthConnected"]
            .agg(
                ConnectedTasks=lambda s: int(pd.to_numeric(s, errors="coerce").sum()),
                TotalTasks="count",
            )
            .reset_index()
        )
        grouped = grouped.merge(
            connected_stats, on=["Model", "OODCategory"], how="left"
        )

    models = sorted(grouped["Model"].unique())
    categories = sorted(grouped["OODCategory"].unique())
    n_models = len(models)
    n_categories = len(categories)
    n_events = len(event_cols)

    if n_models == 0 or n_categories == 0:
        log.warning("No data for event probability bar chart; skipping.")
        return

    fig, axes = plt.subplots(
        1, n_models, figsize=(max(6, 3 * n_categories) * n_models, 6), squeeze=False
    )

    for m_idx, model in enumerate(models):
        ax = axes[0, m_idx]
        mdf = grouped[grouped["Model"] == model].set_index("OODCategory")
        mdf = mdf.reindex(categories)

        x = np.arange(n_categories)
        bar_width = 0.8 / n_events

        connected_counts = None
        if "ConnectedTasks" in mdf.columns:
            connected_counts = (
                pd.to_numeric(mdf["ConnectedTasks"], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=int)
            )

        for e_idx, event in enumerate(event_cols):
            heights = mdf[event].to_numpy(dtype=float)
            for c_idx, h in enumerate(heights):
                xpos = x[c_idx] + e_idx * bar_width
                if np.isnan(h):
                    ax.text(
                        xpos,
                        0.02,
                        "NA",
                        color=event_colors[event],
                        fontsize=7,
                        rotation=90,
                        ha="center",
                        va="bottom",
                    )
                    continue

                ax.bar(
                    xpos,
                    float(h),
                    bar_width,
                    color=event_colors[event],
                    edgecolor="white",
                    linewidth=0.5,
                )

        ax.set_xticks(x + bar_width * (n_events - 1) / 2)
        if connected_counts is not None:
            xticklabels = [
                f"{cat}\n(n_conn={connected_counts[i]})"
                for i, cat in enumerate(categories)
            ]
        else:
            xticklabels = categories
        ax.set_xticklabels(xticklabels, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel("Mean Posterior Probability", fontsize=11)
        ax.set_title(model, fontsize=_PANEL_TITLE_FONTSIZE, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        if m_idx == 0:
            from matplotlib.patches import Patch

            legend_handles = [
                Patch(facecolor=event_colors[event], label=event_labels[event])
                for event in event_cols
            ]
            ax.legend(
                handles=legend_handles,
                fontsize=9,
                loc="upper right",
                framealpha=0.9,
            )

    fig.suptitle(
        "Posterior Event Probabilities by OOD Condition",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved event probability bar chart to %s", output_path)


def generate_posterior_diagnostic_violins(
    diagnostics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Violin/box plots of posterior diagnostic summaries by OOD condition.

    Shows density_ratio and orientation_accuracy distributions across tasks,
    faceted by model.

    Args:
        diagnostics_df: Output of
            :func:`~causal_meta.analysis.posterior_diagnostics.run_posterior_diagnostics`
            with columns ``Model``, ``DatasetKey``, ``density_ratio_mean``,
            ``orientation_accuracy_mean``.
        output_path: Where to save the figure.
    """
    diag_cols = [
        ("density_ratio_mean", "Density Ratio (posterior mean)"),
        ("orientation_accuracy_mean", "Orientation Accuracy (posterior mean)"),
    ]

    if diagnostics_df.empty:
        log.warning("Empty diagnostics DataFrame; skipping violin plots.")
        return

    df = diagnostics_df.copy()
    df["OODCategory"] = df["DatasetKey"].apply(_ood_category)

    models = sorted(df["Model"].unique())
    categories = sorted(df["OODCategory"].unique())
    n_diags = len(diag_cols)

    if not models or not categories:
        log.warning("No data for posterior diagnostic violins; skipping.")
        return

    fig, axes = plt.subplots(
        n_diags,
        len(models),
        figsize=(max(6, 3 * len(categories)) * len(models), 5 * n_diags),
        squeeze=False,
    )

    cmap_cat = plt.get_cmap("tab10")
    cat_colors = {cat: cmap_cat(i) for i, cat in enumerate(categories)}

    for m_idx, model in enumerate(models):
        mdf = df[df["Model"] == model]

        for d_idx, (col, label) in enumerate(diag_cols):
            ax = axes[d_idx, m_idx]

            if col not in mdf.columns:
                ax.set_visible(False)
                continue

            data_per_cat = []
            cat_labels = []
            for cat in categories:
                vals = mdf.loc[mdf["OODCategory"] == cat, col].dropna().values
                if len(vals) > 0:
                    data_per_cat.append(vals)
                    cat_labels.append(cat)

            if not data_per_cat:
                ax.set_visible(False)
                continue

            bp = ax.boxplot(
                data_per_cat,
                tick_labels=cat_labels,
                patch_artist=True,
                widths=0.6,
                showfliers=True,
                flierprops={"markersize": 3, "alpha": 0.5},
            )

            for patch, cat in zip(bp["boxes"], cat_labels):
                patch.set_facecolor(cat_colors.get(cat, "#cccccc"))
                patch.set_alpha(0.7)

            ax.set_ylabel(label, fontsize=10)
            ax.set_title(
                f"{model}" if d_idx == 0 else "", fontsize=12, fontweight="bold"
            )
            ax.tick_params(axis="x", rotation=15)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

            # Add reference line for density ratio = 1
            if "density_ratio" in col:
                ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    fig.suptitle(
        "Posterior Diagnostic Distributions by OOD Condition",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved posterior diagnostic violin plots to %s", output_path)
