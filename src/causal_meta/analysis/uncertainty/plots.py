from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_meta.analysis.common.plot_style import (HAT_E, LABEL_FONTSIZE,
                                                    LEGEND_FONTSIZE,
                                                    SUBTITLE_FONTSIZE,
                                                    TITLE_FONTSIZE,
                                                    grid_figsize)
from causal_meta.analysis.common.thesis import (format_value, graph_code_of,
                                                id_mechanism_of)
from causal_meta.analysis.diagnostics.failure_modes import ood_category
from causal_meta.analysis.uncertainty.ood_detection import \
    compute_ood_detection_metrics
from causal_meta.analysis.utils import (MODEL_COLORS, MODEL_MARKERS,
                                        PAPER_MODEL_LABELS,
                                        EmptyAnalysisDataError,
                                        save_figure_data)

_HAT_E = HAT_E
_TITLE_FONTSIZE = TITLE_FONTSIZE
_PANEL_TITLE_FONTSIZE = SUBTITLE_FONTSIZE
_LABEL_FONTSIZE = LABEL_FONTSIZE
_LEGEND_FONTSIZE = LEGEND_FONTSIZE
_BOTTOM_LEGEND_ANCHOR = (0.5, 0.01)
_BOTTOM_LEGEND_RECT = [0, 0.08, 1, 1]


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#555555")


def _bold_if_best(value: str, *, is_best: bool) -> str:
    return r"\textbf{" + value + "}" if is_best else value


# ── Amortised models used in the thesis uncertainty scatters ────────
_AMORTISED_MODELS_ORDERED: tuple[str, ...] = ("AviCi", "BCNP")

_SCM_ID_ANCHORS: set[tuple[str, str]] = {
    # Graph-shift representative: linear mechanism over the ID graph families.
    ("linear", "er20"),
    ("linear", "er40"),
    ("linear", "er60"),
    ("linear", "sf1"),
    ("linear", "sf2"),
    ("linear", "sf3"),
    # Mechanism-shift representative: ER20 graph over ID mechanisms.
    ("neuralnet", "er20"),
    ("gpcde", "er20"),
    # Transfer/noise/compound anchors used elsewhere in the Results figures.
    ("neuralnet", "sf2"),
    ("gpcde", "er60"),
}

_TRANSFER_ID_ANCHORS: set[tuple[str, str]] = {
    ("linear", "er20"),
    ("neuralnet", "sf2"),
}


def _dataset_anchor(dataset_key: str) -> tuple[str, str] | None:
    mech = id_mechanism_of(dataset_key)
    graph = graph_code_of(dataset_key)
    if mech is None or graph is None:
        return None
    return (mech, graph)


_CATEGORY_COLORS: dict[str, str] = {
    "ID": "#2ca02c",
    "OOD-Graph": "#d62728",
    "OOD-Mech": "#9467bd",
    "OOD-Noise": "#8c564b",
    "OOD-Both": "#e377c2",
    "OOD-Nodes": "#ff7f0e",
    "OOD-Samples": "#1f77b4",
    "OOD": "#17becf",
}


def _add_ideal_tracking_line(
    ax: plt.Axes,
    x_vals: pd.Series,
    y_vals: pd.Series,
    *,
    linear: bool = False,
) -> None:
    """Draw a reference curve showing the ideal tracking direction.

    When *linear* is ``False`` (default, for entropy vs SID), the reference
    is a convex power-law ``y = a * x^p`` (p=2), reflecting the causal-
    cascade amplification between entropy and SID.

    When *linear* is ``True`` (for GraphNLL vs SID), the reference is a
    straight line ``y = a * x``, because NLL divergence scales approximately
    linearly with structural error.
    """
    x_clean = x_vals.dropna()
    y_clean = y_vals.dropna()
    if x_clean.empty or y_clean.empty:
        return
    x_hi = float(np.percentile(x_clean, 90))
    y_hi = float(np.percentile(y_clean, 90))
    if x_hi <= 0 or y_hi <= 0:
        return

    x_max = float(x_clean.max()) * 1.05
    xs = np.linspace(0, x_max, 100)

    if linear:
        a = y_hi / x_hi
        ys = a * xs
    else:
        p = 2.0
        a = y_hi / (x_hi**p)
        ys = a * xs**p

    ax.plot(
        xs,
        ys,
        color="#888888",
        linestyle=":",
        linewidth=1.3,
        alpha=0.7,
        zorder=0,
        label="_ideal",  # hidden from legend
    )


def _prepare_edge_entropy_pivot(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-family edge entropy and ne-SID means for scatter plots."""
    needed = {"ne-sid", "edge_entropy"}
    index_cols = [
        "Model",
        "DatasetKey",
        "Dataset",
        "AxisCategory",
        "NNodes",
        "SamplesPerTask",
    ]
    subset = raw_df[raw_df["Metric"].isin(needed)].copy()
    agg = (
        subset.groupby([*index_cols, "Metric"], dropna=False)["Value"]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=index_cols,
        columns="Metric",
        values="Value",
    ).reset_index()
    pivot.columns.name = None
    for col in needed:
        if col not in pivot.columns:
            raise EmptyAnalysisDataError(
                f"Missing required column for edge-entropy scatter: {col}."
            )
    pivot = pivot.dropna(subset=["ne-sid", "edge_entropy"])
    if pivot.empty:
        raise EmptyAnalysisDataError("No edge-entropy scatter data available.")

    pivot["edge_entropy"] = pivot["edge_entropy"] / np.log(2.0)
    pivot["OODCategory"] = pivot["DatasetKey"].map(
        lambda k: ood_category(k, binary=False)
    )
    return pivot


def _plot_edge_entropy_scatter(
    pivot: pd.DataFrame,
    *,
    output_path: Path,
    label_col: str,
    legend_title: str,
    title: str,
    colors: dict[str, str],
    label_order: list[str],
    marker_col: str | None = None,
    markers: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Plot amortised-model edge entropy against structural error."""
    from scipy.stats import spearmanr

    pivot = pivot[pivot["Model"].isin(_AMORTISED_MODELS_ORDERED)].copy()
    models = [m for m in _AMORTISED_MODELS_ORDERED if m in pivot["Model"].unique()]
    if not models:
        raise EmptyAnalysisDataError(
            "No amortised models found for edge-entropy scatter."
        )

    n_cols = len(models)
    n_rows = 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=grid_figsize(n_cols, n_rows, panel_width=5.2, panel_height=3.8),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    score_metric = "edge_entropy"
    flat_axes = axes.ravel()
    for model_idx, model in enumerate(models):
        ax = flat_axes[model_idx]
        row_idx, col_idx = divmod(model_idx, n_cols)
        model_df = pivot[pivot["Model"] == model]

        for label in label_order:
            label_df = model_df[model_df[label_col] == label]
            if label_df.empty:
                continue

            if marker_col and markers:
                for m_val, m_char in markers.items():
                    m_df = label_df[label_df[marker_col] == m_val]
                    if m_df.empty:
                        continue
                    is_first_marker = m_val == list(markers.keys())[0]
                    ax.scatter(
                        m_df[score_metric],
                        m_df["ne-sid"],
                        color=colors.get(label, "#aaaaaa"),
                        marker=m_char,
                        label=(
                            label
                            if model_idx == 0 and is_first_marker
                            else "_nolegend_"
                        ),
                        s=34,
                        alpha=0.85,
                        edgecolors="white",
                        linewidths=0.4,
                    )
            else:
                ax.scatter(
                    label_df[score_metric],
                    label_df["ne-sid"],
                    color=colors.get(label, "#aaaaaa"),
                    label=label if model_idx == 0 else "_nolegend_",
                    s=34,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.4,
                )

        if marker_col and markers and model_idx == n_cols - 1:
            for m_val, m_char in markers.items():
                ax.scatter([], [], color="gray", marker=m_char, label=m_val)

        # Spearman annotation.
        if len(model_df) >= 3:
            rho, p_val = spearmanr(model_df[score_metric], model_df["ne-sid"])
            if np.isfinite(rho):
                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                ax.annotate(
                    f"$\\rho$={rho:.2f} ({p_str})",
                    xy=(0.03, 0.97),
                    xycoords="axes fraction",
                    fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8),
                )

        if row_idx == n_rows - 1:
            ax.set_xlabel("Edge Entropy", fontsize=_LABEL_FONTSIZE)
        if col_idx == 0:
            ax.set_ylabel(
                rf"Normalized ${_HAT_E}$-SID",
                fontsize=_LABEL_FONTSIZE,
            )
        ax.set_title(model, fontsize=_PANEL_TITLE_FONTSIZE)
        ax.grid(True, linestyle="--", alpha=0.35)

    for ax in flat_axes[len(models) :]:
        ax.set_visible(False)

    # Shared legend below the main title.
    handles, labels = flat_axes[0].get_legend_handles_labels()
    # Filter out internal labels.
    keep = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    if keep:
        legend_ncol = len(keep)
        legend_fontsize = (
            _LEGEND_FONTSIZE if len(keep) <= 7 else max(_LEGEND_FONTSIZE - 1, 7)
        )
        fig.legend(
            [h for h, _ in keep],
            [l for _, l in keep],
            loc="lower center",
            bbox_to_anchor=_BOTTOM_LEGEND_ANCHOR,
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            frameon=True,
            framealpha=0.85,
            columnspacing=0.9,
            handletextpad=0.35,
        )

    fig.suptitle(title, fontsize=_TITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, pivot)
    return pivot


def _sequential_colors(labels: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("viridis", max(len(labels), 2))
    return {label: cmap(i) for i, label in enumerate(labels)}


def generate_edge_entropy_scm_scatter(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    """Scatter for fixed-size SCM component shifts, limited to amortised models."""
    pivot = _prepare_edge_entropy_pivot(raw_df)
    pivot = pivot[
        pivot["AxisCategory"].isin(["id", "graph", "mechanism", "noise", "compound"])
    ].copy()
    id_mask = pivot["AxisCategory"] == "id"
    keep_id = id_mask & pivot["DatasetKey"].map(
        lambda dk: _dataset_anchor(str(dk)) in _SCM_ID_ANCHORS
    )
    pivot = pivot[~id_mask | keep_id].copy()
    if pivot.empty:
        raise EmptyAnalysisDataError("No SCM-shift data for edge-entropy scatter.")

    label_order = [
        label
        for label in _CATEGORY_COLORS
        if label in set(pivot["OODCategory"].dropna().tolist())
    ]
    return _plot_edge_entropy_scatter(
        pivot,
        output_path=output_path,
        label_col="OODCategory",
        legend_title="Target Environments",
        title="SCM Shifts: Edge Entropy vs. Structural Error",
        colors=_CATEGORY_COLORS,
        label_order=label_order,
    )


def _plot_edge_entropy_transfer(
    pivot: pd.DataFrame,
    *,
    output_path: Path,
    x_col: str,
    xlabel: str,
    legend_title: str,
    title: str,
    id_value: int,
) -> pd.DataFrame:
    """Plot amortised-model edge entropy against transfer shift (line plot)."""
    pivot = pivot[pivot["Model"].isin(_AMORTISED_MODELS_ORDERED)].copy()
    models = [m for m in _AMORTISED_MODELS_ORDERED if m in pivot["Model"].unique()]
    if not models:
        raise EmptyAnalysisDataError(
            "No amortised models found for edge-entropy transfer plot."
        )

    n_cols = len(models)
    n_rows = 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=grid_figsize(n_cols, n_rows, panel_width=5.2, panel_height=3.8),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    score_metric = "edge_entropy"
    anchors = sorted(pivot["Anchor"].unique())
    markers = {"ER20 × Linear": "o", "SF2 × MLP": "X"}

    flat_axes = axes.ravel()
    for model_idx, model in enumerate(models):
        ax = flat_axes[model_idx]
        model_df = pivot[pivot["Model"] == model]

        for anchor in anchors:
            adf = model_df[model_df["Anchor"] == anchor].sort_values(x_col)
            if adf.empty:
                continue

            agg = adf.groupby(x_col)[score_metric].agg(["mean", "sem"]).reset_index()

            x_vals = agg[x_col].values
            y_vals = agg["mean"].values
            y_errs = agg["sem"].fillna(0.0).values

            marker = markers.get(anchor, "o")
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_errs,
                marker=marker,
                label=anchor if model_idx == 0 else "_nolegend_",
                capsize=4,
                markersize=8,
                alpha=0.9,
                color=MODEL_COLORS.get(model, "#555555"),
            )
            ax.plot(
                x_vals,
                y_vals,
                color=MODEL_COLORS.get(model, "#555555"),
                linestyle="--",
                alpha=0.5,
            )

        ax.set_title(model, fontsize=_PANEL_TITLE_FONTSIZE)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlabel(xlabel, fontsize=_LABEL_FONTSIZE)
        if model_idx == 0:
            ax.set_ylabel("Edge Entropy", fontsize=_LABEL_FONTSIZE)

        # Draw ID value reference
        ax.axvline(
            id_value,
            color="gray",
            linestyle=":",
            label=f"ID {xlabel}" if model_idx == 0 else "_nolegend_",
        )

    # Shared legend below the main title.
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=_BOTTOM_LEGEND_ANCHOR,
            ncol=len(labels),
            fontsize=_LEGEND_FONTSIZE,
            frameon=True,
            framealpha=0.85,
        )

    fig.suptitle(title, fontsize=_TITLE_FONTSIZE)
    fig.tight_layout(rect=_BOTTOM_LEGEND_RECT)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, pivot)
    return pivot


def generate_edge_entropy_node_scatter(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    """Scatter for node-count shifts, limited to amortised models."""
    pivot = _prepare_edge_entropy_pivot(raw_df)
    pivot = pivot[pivot["AxisCategory"].isin(["id", "nodes"])].copy()
    id_mask = pivot["AxisCategory"] == "id"
    keep_id = id_mask & pivot["DatasetKey"].map(
        lambda dk: _dataset_anchor(str(dk)) in _TRANSFER_ID_ANCHORS
    )
    pivot = pivot[~id_mask | keep_id].copy()
    if pivot.empty:
        raise EmptyAnalysisDataError("No node-shift data for edge-entropy scatter.")
    from causal_meta.analysis.common.thesis import TRANSFER_ANCHOR_LABELS

    pivot["NodeCountLabel"] = pivot.apply(
        lambda row: (
            f"d={int(row['NNodes'])}" + (" (ID)" if row["AxisCategory"] == "id" else "")
        ),
        axis=1,
    )
    pivot["Anchor"] = pivot["DatasetKey"].map(
        lambda dk: TRANSFER_ANCHOR_LABELS.get(
            _dataset_anchor(str(dk)) or ("", ""), "Unknown Anchor"
        )
    )
    label_order = sorted(
        pivot["NodeCountLabel"].unique(),
        key=lambda label: ("(ID)" not in label, int(label.split("=")[1].split()[0])),
    )
    return _plot_edge_entropy_scatter(
        pivot,
        output_path=output_path,
        label_col="NodeCountLabel",
        legend_title="Node Count",
        title="Node-Count Shifts: Edge Entropy vs. Structural Error",
        colors=_sequential_colors(label_order),
        label_order=label_order,
        marker_col="Anchor",
        markers={"ER20 × Linear": "o", "SF2 × MLP": "X"},
    )


def _plot_sample_size_uncertainty(
    pivot: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
) -> pd.DataFrame:
    """Plot sample size vs edge entropy in subplots."""
    pivot = pivot[pivot["Model"].isin(_AMORTISED_MODELS_ORDERED)].copy()
    models = [m for m in _AMORTISED_MODELS_ORDERED if m in pivot["Model"].unique()]
    if not models:
        raise EmptyAnalysisDataError(
            "No amortised models found for edge-entropy sample scatter."
        )

    anchors = sorted(pivot["Anchor"].unique())
    n_cols = len(anchors)
    n_rows = len(models)

    anchor_colors = {"ER20 × Linear": "#2ca02c", "SF2 × MLP": "#9467bd"}

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=grid_figsize(n_cols, n_rows, panel_width=4.5, panel_height=3.5),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row_idx, model in enumerate(models):
        for col_idx, anchor in enumerate(anchors):
            ax = axes[row_idx, col_idx]
            df = pivot[(pivot["Model"] == model) & (pivot["Anchor"] == anchor)]

            if df.empty:
                continue

            ax.scatter(
                df["SamplesPerTask"],
                df["edge_entropy"],
                color=anchor_colors.get(anchor, "#1f77b4"),
                alpha=0.7,
                s=34,
                label=anchor if row_idx == 0 else "_nolegend_",
                edgecolors="white",
                linewidths=0.4,
            )

            if row_idx == 0:
                ax.set_title(anchor, fontsize=_PANEL_TITLE_FONTSIZE)

            if col_idx == 0:
                ax.annotate(
                    model,
                    xy=(-0.35, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    va="center",
                    ha="center",
                    fontweight="bold",
                    fontsize=_PANEL_TITLE_FONTSIZE,
                )
                ax.set_ylabel("Edge Entropy", fontsize=_LABEL_FONTSIZE)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Sample Size", fontsize=_LABEL_FONTSIZE)
                ax.set_xscale("log")
                ax.set_xticks([50, 100, 200, 500, 1000, 2000])
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            # Draw ID value reference
            ax.axvline(500, color="gray", linestyle=":", zorder=0)
            ax.grid(True, linestyle="--", alpha=0.35)

    # Shared legend below the main title.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Target Environments",
            loc="lower center",
            bbox_to_anchor=_BOTTOM_LEGEND_ANCHOR,
            ncol=len(labels),
            fontsize=_LEGEND_FONTSIZE,
            frameon=True,
            framealpha=0.85,
        )

    fig.suptitle(title, fontsize=_TITLE_FONTSIZE)
    fig.tight_layout(rect=_BOTTOM_LEGEND_RECT)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return pivot


def generate_edge_entropy_sample_scatter(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    """Scatter for sample-size shifts, limited to amortised models."""
    pivot = _prepare_edge_entropy_pivot(raw_df)
    pivot = pivot[pivot["AxisCategory"].isin(["id", "samples"])].copy()
    id_mask = pivot["AxisCategory"] == "id"
    keep_id = id_mask & pivot["DatasetKey"].map(
        lambda dk: _dataset_anchor(str(dk)) in _TRANSFER_ID_ANCHORS
    )
    pivot = pivot[~id_mask | keep_id].copy()
    if pivot.empty:
        raise EmptyAnalysisDataError("No sample-shift data for edge-entropy scatter.")

    from causal_meta.analysis.common.thesis import TRANSFER_ANCHOR_LABELS

    pivot["Anchor"] = pivot["DatasetKey"].map(
        lambda dk: TRANSFER_ANCHOR_LABELS.get(
            _dataset_anchor(str(dk)) or ("", ""), "Unknown Anchor"
        )
    )

    return _plot_sample_size_uncertainty(
        pivot,
        output_path=output_path,
        title="Sample-Size Shifts: Edge Entropy",
    )


def generate_ece_summary_table(raw_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    subset = raw_df[raw_df["Metric"].eq("ece")].copy()
    if subset.empty:
        raise EmptyAnalysisDataError("No ECE data available for calibration summary.")
    subset["Split"] = np.where(subset["AxisCategory"].eq("id"), "ID", "OOD")
    family_means = (
        subset.groupby(["Model", "Split", "DatasetKey"], dropna=False)["Value"]
        .mean()
        .reset_index()
    )
    split_agg = (
        family_means.groupby(["Model", "Split"], dropna=False)["Value"]
        .agg(Mean="mean", SD=lambda values: float(values.std(ddof=1)))
        .reset_index()
    )
    split_agg["SD"] = split_agg["SD"].fillna(0.0)
    overall_family_means = (
        subset.groupby(["Model", "DatasetKey"], dropna=False)["Value"]
        .mean()
        .reset_index()
    )
    overall_agg = (
        overall_family_means.groupby(["Model"], dropna=False)["Value"]
        .agg(Mean="mean", SD=lambda values: float(values.std(ddof=1)))
        .reset_index()
    )
    overall_agg["SD"] = overall_agg["SD"].fillna(0.0)
    overall_agg["Split"] = "Overall"
    combined = pd.concat([split_agg, overall_agg], ignore_index=True)
    split_order = ["ID", "OOD", "Overall"]
    lines = [
        r"\footnotesize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{ID ECE $\downarrow$} & \textbf{OOD ECE $\downarrow$} & \textbf{Overall ECE $\downarrow$} \\",
        r"\midrule",
    ]
    best_by_split = {
        split: float(combined[combined["Split"] == split]["Mean"].dropna().min())
        for split in split_order
    }
    for model in list(PAPER_MODEL_LABELS.values()):
        model_rows = combined[combined["Model"] == model]
        cells: list[str] = []
        for split in split_order:
            row = model_rows[model_rows["Split"] == split]
            if row.empty:
                cells.append("-")
                continue
            mean = float(row.iloc[0]["Mean"])
            sd = float(row.iloc[0]["SD"])
            cell = format_value(mean, sd)
            if abs(mean - best_by_split[split]) < 1e-6:
                cell = _bold_if_best(cell, is_best=True)
            cells.append(cell)
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n")
    return combined


def generate_ood_detection_summary_table(
    raw_df: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    edge_entropy_df = compute_ood_detection_metrics(raw_df, score_metric="edge_entropy")
    graph_nll_df = compute_ood_detection_metrics(
        raw_df, score_metric="graph_nll_per_edge"
    )
    frames: list[pd.DataFrame] = []
    for score_name, detection_df in (
        ("edge_entropy", edge_entropy_df),
        ("graph_nll_per_edge", graph_nll_df),
    ):
        if detection_df.empty:
            continue
        renamed = detection_df.rename(
            columns={
                "AUROC": f"{score_name}_AUROC",
                "AUPRC": f"{score_name}_AUPRC",
                "N_ID": f"{score_name}_N_ID",
                "N_OOD": f"{score_name}_N_OOD",
            }
        )
        frames.append(renamed.drop(columns=["ScoreMetric"], errors="ignore"))
    if not frames:
        raise EmptyAnalysisDataError("No OOD detection metrics could be computed.")
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on=["RunID", "Model"], how="outer")
    combined = combined.sort_values("Model")
    lines = [
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Entropy AUROC} & \textbf{Entropy AUPRC} & \textbf{Graph NLL / edge AUROC} & \textbf{Graph NLL / edge AUPRC} \\",
        r"\midrule",
    ]
    metric_cols = [
        "edge_entropy_AUROC",
        "edge_entropy_AUPRC",
        "graph_nll_per_edge_AUROC",
        "graph_nll_per_edge_AUPRC",
    ]
    col_best = {
        col: float(combined[col].dropna().max())
        for col in metric_cols
        if col in combined.columns
    }

    def _fmt_cell(value: float, col: str) -> str:
        cell = f"{value:.3f}"
        return (
            r"\textbf{" + cell + "}"
            if np.isfinite(value)
            and abs(value - col_best.get(col, float("-inf"))) < 1e-6
            else cell
        )

    for _, row in combined.iterrows():
        lines.append(
            f"{row['Model']} & {_fmt_cell(float(row.get('edge_entropy_AUROC', float('nan'))), 'edge_entropy_AUROC')} & {_fmt_cell(float(row.get('edge_entropy_AUPRC', float('nan'))), 'edge_entropy_AUPRC')} & {_fmt_cell(float(row.get('graph_nll_per_edge_AUROC', float('nan'))), 'graph_nll_per_edge_AUROC')} & {_fmt_cell(float(row.get('graph_nll_per_edge_AUPRC', float('nan'))), 'graph_nll_per_edge_AUPRC')}"
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    output_path.write_text("\n".join(lines) + "\n")
    return combined
