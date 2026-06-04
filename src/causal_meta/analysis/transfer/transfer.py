from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_meta.analysis.common.plot_style import (HAT_E, LABEL_FONTSIZE,
                                                    LEGEND_FONTSIZE,
                                                    SUBTITLE_FONTSIZE,
                                                    TITLE_FONTSIZE,
                                                    row_figsize)
from causal_meta.analysis.common.thesis import (TRANSFER_ANCHOR_LABELS,
                                                graph_code_of, id_mechanism_of,
                                                metric_sem, transfer_anchor)
from causal_meta.analysis.utils import (ERROR_SPECS, MODEL_COLORS,
                                        MODEL_MARKERS, PAPER_MODEL_LABELS,
                                        save_figure_data)

_HAT_E = HAT_E
_TITLE_FONTSIZE = TITLE_FONTSIZE
_PANEL_TITLE_FONTSIZE = SUBTITLE_FONTSIZE
_LABEL_FONTSIZE = LABEL_FONTSIZE
_LEGEND_FONTSIZE = LEGEND_FONTSIZE

_THREE_ROW_HEIGHT_RATIOS = [1.05, 1.9, 1.25]


def _three_row_figsize(
    n_cols: int, *, match_width_cols: int | None = None
) -> tuple[float, float]:
    """Return consistent Results sizing for DAG/metric/error transfer figures."""
    width_cols = max(match_width_cols or n_cols, 1)
    return row_figsize(width_cols, _THREE_ROW_HEIGHT_RATIOS)


log = logging.getLogger(__name__)


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#555555")


def _dag_ylabel() -> str:
    return r"Valid DAG (%) $\uparrow$"


def _error_count_ylabel() -> str:
    return r"Error count $\downarrow$"


def _model_legend_ncol(n_labels: int) -> int:
    """Return a compact legend column count for model legends."""
    if n_labels <= 2:
        return n_labels
    if n_labels <= 4:
        return 2
    if n_labels <= 6:
        return 3
    return 4


def _model_legend_fontsize(n_labels: int) -> int:
    """Return a readable model-legend font size for current label count."""
    if n_labels <= 4:
        return _LEGEND_FONTSIZE
    return max(_LEGEND_FONTSIZE - 1, 7)


# ID anchor values for the transfer axes.
_ID_NODE_COUNT = 20
_ID_SAMPLE_COUNT = 500

# Full training-support node counts used during DG pre-training.
_TRAINING_NODE_COUNTS: set[int] = {10, 20, 30, 40}

# Ordered list of transfer ladder anchors: (mechanism, graph_code).
_TRANSFER_ANCHORS: list[tuple[str, str]] = [
    ("linear", "er20"),
    ("neuralnet", "sf2"),
]


# =====================================================================
# RQ2-specific 2×3 transfer figures
# =====================================================================

_ERROR_METRICS = ERROR_SPECS


def generate_rq2_transfer_figure(
    raw_df: pd.DataFrame,
    *,
    axis: str,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a 3-row × N-col transfer figure for the RQ2 thesis section.

    Layout: 3 rows × N columns (one per transfer anchor):
      row 0 (top)    – DAG validity % (AviCi sampled/thresholded, DiBS sampled)
      row 1 (middle) – ne-SID (lower is better, all models)
      row 2 (bottom) – Error decomposition (stacked FP / FN / Reversed)

    Args:
        raw_df: Long-format raw task DataFrame.
        axis: ``"nodes"`` or ``"samples"``.
        output_path: Path for the output PDF figure.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    if axis == "nodes":
        axis_categories = {"id", "nodes"}
        x_col = "NNodes"
        xlabel = "Target node count"
        suptitle = "Node-Count Target Environments"
        id_value = _ID_NODE_COUNT
    else:
        axis_categories = {"id", "samples"}
        x_col = "SamplesPerTask"
        xlabel = "Observational samples per task"
        suptitle = "Sample-Size Target Environments"
        id_value = _ID_SAMPLE_COUNT

    # Metrics needed: ne-sid for metric row, valid_dag_pct + threshold for
    # DAG row, fp/fn/reversed for error row.
    line_metrics = {"ne-sid", "valid_dag_pct", "threshold_valid_dag_pct"}
    error_metrics = {m for m, _, _ in _ERROR_METRICS}
    all_needed = line_metrics | error_metrics

    subset = raw_df[
        raw_df["AxisCategory"].isin(axis_categories) & raw_df["Metric"].isin(all_needed)
    ].copy()
    subset = subset[subset[x_col].notna()]

    if subset.empty:
        log.warning("No RQ2 transfer data for axis=%s; skipping.", axis)
        return pd.DataFrame()

    # Tag each row with its transfer anchor.
    subset["_anchor"] = subset["DatasetKey"].map(transfer_anchor)
    id_mask = subset["AxisCategory"] == "id"
    subset.loc[id_mask, "_anchor"] = subset.loc[id_mask, "DatasetKey"].map(
        lambda dk: (
            (id_mechanism_of(dk), graph_code_of(dk))
            if id_mechanism_of(dk) is not None
            else None
        )
    )

    present_anchors = [
        a for a in _TRANSFER_ANCHORS if a in set(subset["_anchor"].dropna().tolist())
    ]
    if not present_anchors:
        log.warning(
            "No recognised transfer anchors for axis=%s; skipping RQ2 figure.", axis
        )
        return pd.DataFrame()

    n_cols = len(present_anchors)
    height_ratios = _THREE_ROW_HEIGHT_RATIOS  # DAG, metric, error
    row_names = ["dag", "metric", "error"]
    n_rows = 3

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=_three_row_figsize(n_cols, match_width_cols=3),
        sharex="col",
        sharey=False,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    dag_row = 0
    metric_row = 1
    err_row = 2

    _DAG_MODELS = ["AviCi", "DiBS"]
    _DAG_METRICS = [
        ("threshold_valid_dag_pct", "Thresholded", 0.55),
        ("valid_dag_pct", "Sampled", 0.85),
    ]

    all_agg: list[pd.DataFrame] = []

    for col_idx, anchor in enumerate(present_anchors):
        anchor_data = subset[subset["_anchor"] == anchor]
        if anchor_data.empty:
            for r in range(n_rows):
                axes[r, col_idx].set_visible(False)
            continue

        agg = (
            anchor_data.groupby(["Model", x_col, "Metric"], dropna=False)["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg[x_col] = agg[x_col].astype(int)
        all_agg.append(agg)

        x_values = sorted(agg[x_col].unique())
        n_x = len(x_values)
        x_to_idx = {v: i for i, v in enumerate(x_values)}

        models = [m for m in PAPER_MODEL_LABELS.values() if m in agg["Model"].unique()]
        n_models = len(models)
        width = 0.6
        offset_step = width / max(n_models, 1)

        anchor_label = TRANSFER_ANCHOR_LABELS.get(
            anchor, f"{anchor[1]} \u00d7 {anchor[0]}"
        )

        # Column header on topmost row.
        axes[0, col_idx].set_title(anchor_label, fontsize=_PANEL_TITLE_FONTSIZE)

        # ── Row 0: DAG validity (AviCi + DiBS) ──────────────────────
        dag_ax = axes[dag_row, col_idx]
        dag_model_names = [m for m in _DAG_MODELS if m in agg["Model"].unique()]
        n_dag_models = len(dag_model_names)
        bar_width = 0.15
        for dm_idx, dag_model in enumerate(dag_model_names):
            for m_idx, (m_name, m_label, m_alpha) in enumerate(_DAG_METRICS):
                m_data = agg[(agg["Model"] == dag_model) & (agg["Metric"] == m_name)]
                if m_data.empty:
                    continue
                dag_xs: list[float] = []
                dag_means: list[float] = []
                dag_sems: list[float] = []
                for v in x_values:
                    row = m_data[m_data[x_col] == v]
                    if row.empty:
                        continue
                    # Offset: model group first, then sampled/thresholded within
                    group_offset = (dm_idx - n_dag_models / 2 + 0.5) * bar_width * 2.2
                    inner_offset = (m_idx - 0.5) * bar_width
                    dag_xs.append(float(x_to_idx[v]) + group_offset + inner_offset)
                    dag_means.append(float(row.iloc[0]["Mean"]))
                    dag_sems.append(float(row.iloc[0]["SEM"]))
                if dag_xs:
                    lbl = f"{dag_model} {m_label}" if col_idx == 0 else None
                    dag_ax.bar(
                        dag_xs,
                        dag_means,
                        yerr=dag_sems,
                        width=bar_width,
                        color=_model_color(dag_model),
                        alpha=m_alpha,
                        capsize=2,
                        label=lbl,
                        edgecolor="white",
                        linewidth=0.3,
                    )

        dag_ax.set_ylim(0, 105)
        dag_ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if col_idx == 0:
            dag_ax.set_ylabel(_dag_ylabel(), fontsize=_LABEL_FONTSIZE)
            dag_handles, dag_labels = dag_ax.get_legend_handles_labels()
            if dag_handles:
                dag_ax.legend(
                    dag_handles,
                    dag_labels,
                    fontsize=_LEGEND_FONTSIZE,
                    loc="lower left",
                    frameon=True,
                    framealpha=0.8,
                )

        # ── Row 1: ne-SID (all models) ──────────────────────────────
        _plot_transfer_line(
            axes[metric_row, col_idx],
            agg,
            metric_name="ne-sid",
            x_col=x_col,
            x_values=x_values,
            x_to_idx=x_to_idx,
            models=models,
            offset_step=offset_step,
            n_models=n_models,
            ylabel=rf"Normalized ${_HAT_E}$-SID $\downarrow$",
            add_legend=(col_idx == 0),
        )

        # ── Row 2: Error decomposition ──────────────────────────────
        _plot_error_decomposition(
            axes[err_row, col_idx],
            agg,
            x_col=x_col,
            x_values=x_values,
            models=models,
            add_legend=(col_idx == 0),
        )

        # Shade training-support / ID region for all rows.
        for r in range(n_rows):
            ax = axes[r, col_idx]
            _shade_support_region(ax, axis, x_to_idx, n_x, id_value)

        # X-tick labels on bottom row only.
        bottom_ax = axes[n_rows - 1, col_idx]
        bottom_ax.set_xticks(np.arange(n_x))
        bottom_ax.set_xticklabels([str(v) for v in x_values], fontsize=10)
        bottom_ax.set_xlabel(xlabel, fontsize=_LABEL_FONTSIZE)

        for r in range(n_rows):
            axes[r, col_idx].grid(True, axis="y", linestyle="--", alpha=0.4)

    # Shared legend on top — models only (from the metric row).
    handles, labels = axes[metric_row, 0].get_legend_handles_labels()
    if handles:
        axes[metric_row, 0].legend(
            handles,
            labels,
            title="Model",
            loc="best",
            ncol=_model_legend_ncol(len(labels)),
            fontsize=_model_legend_fontsize(len(labels)),
            frameon=True,
            framealpha=0.8,
        )

    fig.suptitle(suptitle, fontsize=_TITLE_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    combined = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    save_figure_data(output_path, combined)
    return combined


def _plot_transfer_line(
    ax: plt.Axes,
    agg: pd.DataFrame,
    *,
    metric_name: str,
    x_col: str,
    x_values: list[int],
    x_to_idx: dict[int, int],
    models: list[str],
    offset_step: float,
    n_models: int,
    ylabel: str,
    add_legend: bool,
) -> None:
    """Plot a single metric as a line-with-errorbars panel."""
    metric_df = agg[agg["Metric"] == metric_name]
    for model_idx, model in enumerate(models):
        model_df = metric_df[metric_df["Model"] == model].sort_values(x_col)
        if model_df.empty:
            continue
        xs: list[float] = []
        means: list[float] = []
        sems: list[float] = []
        for _, row in model_df.iterrows():
            base_x = x_to_idx[int(row[x_col])]
            offset = (model_idx - n_models / 2 + 0.5) * offset_step
            xs.append(float(base_x) + offset)
            means.append(float(row["Mean"]))
            sems.append(float(row["SEM"]))
        ax.errorbar(
            xs,
            means,
            yerr=sems,
            fmt=MODEL_MARKERS.get(model, "o"),
            label=model if add_legend else None,
            color=_model_color(model),
            capsize=3,
            markersize=7,
            alpha=0.9,
        )
    if add_legend:
        ax.set_ylabel(ylabel, fontsize=_LABEL_FONTSIZE)


def _plot_error_decomposition(
    ax: plt.Axes,
    agg: pd.DataFrame,
    *,
    x_col: str,
    x_values: list[int],
    models: list[str],
    add_legend: bool,
) -> None:
    """Plot stacked FP/FN/reversed bars per model at each x position.

    Each model gets a thin stacked bar; the three error types are stacked
    vertically with distinct colours.
    """
    n_x = len(x_values)
    n_models = len(models)
    bar_width = 0.15

    for model_idx, model in enumerate(models):
        offset = (model_idx - n_models / 2 + 0.5) * bar_width
        bottoms = np.zeros(n_x)
        for metric_key, metric_label, colour in _ERROR_METRICS:
            metric_df = agg[
                (agg["Metric"] == metric_key) & (agg["Model"] == model)
            ].set_index(x_col)
            heights = np.array(
                [
                    float(metric_df.loc[v, "Mean"]) if v in metric_df.index else 0.0
                    for v in x_values
                ]
            )
            bar_label = metric_label if model_idx == 0 and add_legend else None
            ax.bar(
                np.arange(n_x) + offset,
                heights,
                bar_width,
                bottom=bottoms,
                color=colour,
                alpha=0.75,
                label=bar_label,
                edgecolor="white",
                linewidth=0.3,
            )
            bottoms += heights

    if add_legend:
        ax.set_ylabel(_error_count_ylabel(), fontsize=_LABEL_FONTSIZE)
        err_handles, err_labels = ax.get_legend_handles_labels()
        if err_handles:
            ax.legend(
                err_handles,
                err_labels,
                fontsize=_LEGEND_FONTSIZE,
                loc="upper right",
                frameon=True,
                framealpha=0.8,
            )


def _shade_support_region(
    ax: plt.Axes,
    axis: str,
    x_to_idx: dict[int, int],
    n_x: int,
    id_value: int,
) -> None:
    """Shade the in-training-support region(s) on an axis.

    For the *nodes* axis the entire interpolation range (min to max of
    ``_TRAINING_NODE_COUNTS``) is shaded as a single contiguous grey band
    to clearly mark the ID regime.  For *samples* only the single ID
    tick is highlighted.
    """
    if axis == "nodes":
        support_indices = sorted(
            x_to_idx[v] for v in _TRAINING_NODE_COUNTS if v in x_to_idx
        )
        if support_indices:
            lo = min(support_indices)
            hi = max(support_indices)
            ax.axvspan(lo - 0.5, hi + 0.5, color="#d9d9d9", alpha=0.40, zorder=0)
            if hi < n_x - 1:
                ax.axvline(hi + 0.5, color="#999999", linestyle=":", linewidth=1.0)
    elif id_value in x_to_idx:
        id_idx = x_to_idx[id_value]
        ax.axvspan(id_idx - 0.5, id_idx + 0.5, color="#d9d9d9", alpha=0.40, zorder=0)
        if id_idx < n_x - 1:
            ax.axvline(id_idx + 0.5, color="#999999", linestyle=":", linewidth=1.0)
