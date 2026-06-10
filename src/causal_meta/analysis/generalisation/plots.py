from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_meta.analysis.common.plot_style import (ANNOTATION_FONTSIZE, HAT_E,
                                                    HEATMAP_FIGSIZE,
                                                    LABEL_FONTSIZE,
                                                    LEGEND_FONTSIZE,
                                                    SUBTITLE_FONTSIZE,
                                                    TABLE_CELL_FONTSIZE,
                                                    TICK_FONTSIZE,
                                                    TITLE_FONTSIZE,
                                                    TOP_LEGEND_ANCHOR,
                                                    TOP_LEGEND_RECT,
                                                    WIDE_TWO_ROW_FIGSIZE,
                                                    grid_figsize, metric_label,
                                                    row_figsize)
from causal_meta.analysis.common.thesis import (GRAPH_ANCHOR_LABELS,
                                                ID_MECHANISM_LABELS,
                                                TRANSFER_ANCHOR_LABELS,
                                                axis_category, format_value,
                                                graph_code_of, graph_family_of,
                                                id_mechanism_of,
                                                is_fixed_size_task_frame,
                                                mech_shift_graph_anchor,
                                                metric_sem, noise_shift_anchor,
                                                paper_model_label,
                                                thesis_dataset_label,
                                                transfer_anchor)
from causal_meta.analysis.utils import (ERROR_SPECS, GRAPH_DESCRIPTION_MAP,
                                        MECH_DESCRIPTION_MAP, MODEL_COLORS,
                                        MODEL_MARKERS, PAPER_MODEL_LABELS,
                                        EmptyAnalysisDataError,
                                        load_runs_dataframe, save_figure_data)

log = logging.getLogger(__name__)

# Local aliases keep older plotting helpers readable while all values are now
# sourced from the shared thesis plot-style module.
_HAT_E = HAT_E
_TITLE_FONTSIZE = TITLE_FONTSIZE
_PANEL_TITLE_FONTSIZE = SUBTITLE_FONTSIZE
_LABEL_FONTSIZE = LABEL_FONTSIZE
_TICK_FONTSIZE = TICK_FONTSIZE
_LEGEND_FONTSIZE = LEGEND_FONTSIZE

_THREE_ROW_HEIGHT_RATIOS = [1.05, 1.9, 1.25]
_SHIFT_XTICK_FONTSIZE = 10
_GRAPH_SHIFT_CODE_ORDER: tuple[str, ...] = (
    "er20",
    "er40",
    "er60",
    "sf1",
    "sf2",
    "sf3",
    "grg",
    "sbm",
    "ws",
)
_GRAPH_SHIFT_ORDER = {code: idx for idx, code in enumerate(_GRAPH_SHIFT_CODE_ORDER)}
_GRAPH_SHIFT_LABELS = {
    code: GRAPH_ANCHOR_LABELS.get(code, code.upper())
    for code in _GRAPH_SHIFT_CODE_ORDER
}


def _three_row_figsize(
    n_panels: int, *, match_width_cols: int | None = None
) -> tuple[float, float]:
    """Return consistent Results sizing for DAG/metric/error row figures."""
    width_panels = max(match_width_cols or n_panels, 1)
    return row_figsize(width_panels, _THREE_ROW_HEIGHT_RATIOS)


def _sid_ylabel(metric_name: str) -> str:
    return metric_label(metric_name, down=True)


def _dag_ylabel() -> str:
    return r"Valid DAG (%) $\uparrow$"


def _error_count_ylabel(*, log_scale: bool = False) -> str:
    if log_scale:
        return r"Mean error count (log-scale) $\downarrow$"
    return r"Error count $\downarrow$"


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#555555")


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


def _graph_shift_sort_value(dataset_key: str) -> int:
    graph_code = graph_code_of(dataset_key)
    if graph_code is None:
        return len(_GRAPH_SHIFT_ORDER)
    return _GRAPH_SHIFT_ORDER.get(graph_code, len(_GRAPH_SHIFT_ORDER))


def _resolve_models(model_filter: Sequence[str] | None = None) -> list[str]:
    """Return the ordered model list, optionally restricted to *model_filter*."""
    if model_filter is not None:
        filter_set = set(model_filter)
        return [m for m in PAPER_MODEL_LABELS.values() if m in filter_set]
    return list(PAPER_MODEL_LABELS.values())


def _bold_if_best(value: str, *, is_best: bool) -> str:
    return r"\textbf{" + value + "}" if is_best else value


_SHIFT_AXIS_SPECS: dict[str, tuple[str, str]] = {
    "graph": ("graph", "Graph Target Environments"),
    "mechanism": ("mechanism", "Mechanism Target Environments"),
    "noise": ("noise", "Noise Target Environments"),
    "compound": ("compound", "Graph-Mechanism Target Environments"),
}

_COMPOUND_ID_REPRESENTATIVES: tuple[tuple[str, str], ...] = (
    ("linear", "er20"),
    ("neuralnet", "sf2"),
    ("gpcde", "er60"),
)

_COMPOUND_OOD_GRAPHS: list[str] = ["sbm", "ws", "grg"]

_COMPOUND_OOD_MECH_ORDER: dict[str, int] = {
    "periodic": 0,
    "logistic_map": 1,
    "pnl_tanh": 2,
    "square": 3,
}

_MECH_COMBO_LABELS: dict[str, str] = {
    "linear": "Linear",
    "neuralnet": "MLP",
    "gpcde": "RFF",
    "pnl_tanh": "PNL",
    "logistic_map": "Logistic",
    "periodic": "Periodic",
    "square": "Square",
}


def _mech_combo_label(mech_key: str) -> str:
    return _MECH_COMBO_LABELS.get(
        mech_key, MECH_DESCRIPTION_MAP.get(mech_key, mech_key)
    )


def _graph_mech_combo_label(graph_code: str, mech_key: str) -> str:
    graph_label = GRAPH_ANCHOR_LABELS.get(graph_code, graph_code.upper())
    return f"{graph_label}+{_mech_combo_label(mech_key)}"


def _ood_mech_code(dataset_key: str) -> str | None:
    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)
    if body.startswith("ood_mech_"):
        remainder = body[len("ood_mech_") :]
    elif body.startswith("ood_both_"):
        remainder = body[len("ood_both_") :]
    else:
        return None

    graph_code = graph_code_of(dk)
    if graph_code is None:
        return None

    prefix = f"{graph_code}_"
    if remainder.startswith(prefix):
        return remainder[len(prefix) :]
    suffix = f"_{graph_code}"
    if remainder.endswith(suffix):
        return remainder[: -len(suffix)]
    return None


# ── Graph-shift multi-panel helpers ────────────────────────────────────


def _graph_shift_label(dataset_key: str) -> str:
    """Produce a short graph-topology label for graph shift panels.

    OOD-graph datasets return the OOD graph name (e.g. ``"SBM"``).
    ID datasets return the ID graph spec without suffix.
    """
    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)

    if dk.startswith("ood_graph_"):
        first_token = body[len("ood_graph_") :].split("_")[0]
        return GRAPH_DESCRIPTION_MAP.get(first_token, first_token.upper())

    # ID datasets — show full graph spec
    for code in sorted(GRAPH_DESCRIPTION_MAP.keys(), key=len, reverse=True):
        if f"_{code}" in body:
            return GRAPH_DESCRIPTION_MAP[code]

    return dataset_key


# ── Shared row helpers for 3-row shift figures ────────────────────────


def _plot_dag_row_panel(
    dag_ax: plt.Axes,
    dag_panel_data: pd.DataFrame | None,
    datasets: list[str],
    x_base: np.ndarray,
    label_fn,
    *,
    panel_idx: int = 0,
    id_count: int = 0,
    stress_count: int = 0,
) -> None:
    """Plot AviCi DAG validity (sampled + thresholded) in a single panel axis."""
    if dag_panel_data is None or dag_panel_data.empty:
        return

    _DAG_METRICS = [
        ("threshold_valid_dag_pct", "Thresholded", 0.55),
        ("valid_dag_pct", "Sampled", 0.85),
    ]
    bar_width = 0.20
    for m_idx, (m_name, m_label, m_alpha) in enumerate(_DAG_METRICS):
        m_data = dag_panel_data[dag_panel_data["Metric"] == m_name]
        if m_data.empty:
            continue
        dag_agg = (
            m_data.groupby(["DatasetKey", "Dataset", "AxisCategory"], dropna=False)[
                "Value"
            ]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        dag_agg["DatasetLabel"] = dag_agg["DatasetKey"].map(label_fn)

        dag_xs: list[float] = []
        dag_means: list[float] = []
        dag_sems: list[float] = []
        for i, ds in enumerate(datasets):
            row = dag_agg[dag_agg["DatasetLabel"] == ds]
            if row.empty:
                continue
            offset = (m_idx - 0.5) * bar_width
            dag_xs.append(float(x_base[i]) + offset)
            dag_means.append(float(row.iloc[0]["Mean"]))
            dag_sems.append(float(row.iloc[0]["SEM"]))

        if dag_xs:
            dag_ax.bar(
                dag_xs,
                dag_means,
                yerr=dag_sems,
                width=bar_width,
                color=_model_color("AviCi"),
                alpha=m_alpha,
                capsize=2,
                label=m_label if panel_idx == 0 else None,
            )

    dag_ax.set_ylim(0, 105)
    dag_ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if panel_idx == 0:
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

    if 0 < id_count < len(datasets):
        dag_ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
        dag_ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    if stress_count > 0:
        n_ds = len(datasets)
        stress_start = n_ds - stress_count
        dag_ax.axvspan(
            stress_start - 0.5, n_ds - 0.5, color="#ffe0e0", alpha=0.35, zorder=0
        )
        dag_ax.axvline(
            stress_start - 0.5, color="#cc4444", linestyle=":", linewidth=1.0
        )


def _plot_error_row_panel(
    err_ax: plt.Axes,
    error_panel_data: pd.DataFrame | None,
    datasets: list[str],
    x_base: np.ndarray,
    models: list[str],
    label_fn,
    *,
    panel_idx: int = 0,
    id_count: int = 0,
    stress_count: int = 0,
) -> None:
    """Plot stacked error decomposition (FP/FN/Reversed) in a single panel axis."""
    if error_panel_data is None or error_panel_data.empty:
        return

    n_models = len(models)
    bar_width = 0.20

    error_agg = (
        error_panel_data.groupby(["Model", "DatasetKey", "Metric"], dropna=False)[
            "Value"
        ]
        .mean()
        .reset_index()
        .rename(columns={"Value": "Mean"})
    )
    error_agg["DatasetLabel"] = error_agg["DatasetKey"].map(label_fn)

    for model_idx, model in enumerate(models):
        model_err = error_agg[error_agg["Model"] == model]
        offset = (model_idx - n_models / 2 + 0.5) * bar_width
        bottoms = np.zeros(len(datasets))
        for err_key, err_label, err_color in ERROR_SPECS:
            heights = np.zeros(len(datasets))
            for ds_idx, ds in enumerate(datasets):
                row = model_err[
                    (model_err["DatasetLabel"] == ds) & (model_err["Metric"] == err_key)
                ]
                if not row.empty:
                    heights[ds_idx] = float(row.iloc[0]["Mean"])
            label = err_label if model_idx == 0 and panel_idx == 0 else None
            err_ax.bar(
                x_base + offset,
                heights,
                bar_width,
                bottom=bottoms,
                color=err_color,
                alpha=0.75,
                label=label,
                edgecolor="white",
                linewidth=0.3,
            )
            bottoms += heights

    err_ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    if panel_idx == 0:
        err_ax.set_ylabel(_error_count_ylabel(), fontsize=_LABEL_FONTSIZE)
        err_handles, err_labels = err_ax.get_legend_handles_labels()
        if err_handles:
            err_ax.legend(
                err_handles,
                err_labels,
                fontsize=_LEGEND_FONTSIZE,
                loc="upper right",
                frameon=True,
                framealpha=0.8,
            )

    if 0 < id_count < len(datasets):
        err_ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
        err_ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    if stress_count > 0:
        n_ds = len(datasets)
        stress_start = n_ds - stress_count
        err_ax.axvspan(
            stress_start - 0.5, n_ds - 0.5, color="#ffe0e0", alpha=0.35, zorder=0
        )
        err_ax.axvline(
            stress_start - 0.5, color="#cc4444", linestyle=":", linewidth=1.0
        )


def _generate_graph_shift_panels(
    subset: pd.DataFrame,
    metric_name: str,
    axis_title: str,
    output_path: Path,
    *,
    model_filter: Sequence[str] | None = None,
    avici_dag_df: pd.DataFrame | None = None,
    error_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate a multi-panel graph shift figure, one column per ID mechanism family.

    Three-row layout when auxiliary data is supplied:
      Row 0 (top):    AviCi DAG validity (sampled + thresholded).
      Row 1 (middle): Primary metric (E-SID / NE-SID).
      Row 2 (bottom): Error decomposition (FP / FN / Reversed).
    """
    subset = subset.copy()
    subset["MechFamily"] = subset["DatasetKey"].map(id_mechanism_of)
    subset = subset[subset["MechFamily"].notna()]

    mech_families = [
        (mk, ml)
        for mk, ml in ID_MECHANISM_LABELS.items()
        if mk in subset["MechFamily"].unique()
    ]
    if not mech_families:
        return pd.DataFrame()

    n_panels = len(mech_families)
    models = _resolve_models(model_filter)
    n_models = len(models)

    has_dag = avici_dag_df is not None and not avici_dag_df.empty
    has_err = error_df is not None and not error_df.empty

    # Tag auxiliary DataFrames with MechFamily the same way.
    dag_sub: pd.DataFrame | None = None
    if has_dag:
        dag_sub = avici_dag_df.copy()
        dag_sub["MechFamily"] = dag_sub["DatasetKey"].map(id_mechanism_of)
        dag_sub = dag_sub[dag_sub["MechFamily"].notna()]
    err_sub: pd.DataFrame | None = None
    if has_err:
        err_sub = error_df.copy()
        err_sub["MechFamily"] = err_sub["DatasetKey"].map(id_mechanism_of)
        err_sub = err_sub[err_sub["MechFamily"].notna()]

    # Build row structure: DAG (optional) | metric | error (optional).
    height_ratios: list[float] = []
    row_names: list[str] = []
    if has_dag:
        height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[0])
        row_names.append("dag")
    height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[1])
    row_names.append("metric")
    if has_err:
        height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[2])
        row_names.append("error")
    n_rows = len(row_names)

    fig, axes = plt.subplots(
        n_rows,
        n_panels,
        figsize=_three_row_figsize(n_panels),
        sharex="col",
        sharey=False,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios, "wspace": 0.16},
    )

    metric_row = row_names.index("metric")
    dag_row = row_names.index("dag") if has_dag else None
    err_row = row_names.index("error") if has_err else None

    all_agg: list[pd.DataFrame] = []

    for panel_idx, (mech_key, mech_label) in enumerate(mech_families):
        ax = axes[metric_row, panel_idx]
        panel_data = subset[subset["MechFamily"] == mech_key]

        agg = (
            panel_data.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["DatasetLabel"] = agg["DatasetKey"].map(_graph_shift_label)
        agg["_graph_order"] = agg["DatasetKey"].map(_graph_shift_sort_value)
        agg["OrderX"] = agg["_graph_order"]
        agg = agg.sort_values(["_graph_order", "DatasetLabel"]).drop(
            columns=["_graph_order"]
        )
        all_agg.append(agg)

        datasets = list(agg["DatasetLabel"].unique())
        axis_lookup = (
            agg[["DatasetLabel", "AxisCategory"]]
            .drop_duplicates()
            .set_index("DatasetLabel")["AxisCategory"]
            .to_dict()
        )
        n_datasets = len(datasets)
        width = 0.6
        offset_step = width / max(n_models, 1)
        x_base = np.arange(n_datasets)
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")

        # ── Metric row ────────────────────────────────────────────────
        for model_idx, model in enumerate(models):
            model_agg = agg[agg["Model"] == model]
            xs: list[float] = []
            means: list[float] = []
            sems: list[float] = []
            for i, ds in enumerate(datasets):
                row = model_agg[model_agg["DatasetLabel"] == ds]
                if row.empty:
                    continue
                offset = (model_idx - n_models / 2 + 0.5) * offset_step
                xs.append(float(x_base[i]) + offset)
                means.append(float(row.iloc[0]["Mean"]))
                sems.append(float(row.iloc[0]["SEM"]))
            if xs:
                ax.errorbar(
                    xs,
                    means,
                    yerr=sems,
                    fmt=MODEL_MARKERS.get(model, "o"),
                    label=model if panel_idx == 0 else None,
                    color=_model_color(model),
                    capsize=3,
                    markersize=7,
                    alpha=0.9,
                )

        # Column header on the topmost row (dag if present, else metric).
        axes[0, panel_idx].set_title(mech_label, fontsize=_PANEL_TITLE_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if panel_idx == 0:
            ylabel = _sid_ylabel(metric_name)
            ax.set_ylabel(ylabel, fontsize=_LABEL_FONTSIZE)

        # Grey ID region on metric row
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

        # ── DAG row (top) ─────────────────────────────────────────────
        if dag_row is not None:
            dag_panel = (
                dag_sub[dag_sub["MechFamily"] == mech_key]
                if dag_sub is not None
                else pd.DataFrame()
            )
            _plot_dag_row_panel(
                axes[dag_row, panel_idx],
                dag_panel,
                datasets,
                x_base,
                _graph_shift_label,
                panel_idx=panel_idx,
                id_count=id_count,
            )

        # ── Error row (bottom) ────────────────────────────────────────
        if err_row is not None:
            err_panel = (
                err_sub[err_sub["MechFamily"] == mech_key]
                if err_sub is not None
                else pd.DataFrame()
            )
            _plot_error_row_panel(
                axes[err_row, panel_idx],
                err_panel,
                datasets,
                x_base,
                models,
                _graph_shift_label,
                panel_idx=panel_idx,
                id_count=id_count,
            )

    # X tick labels on the bottom row only (sharex handles the rest).
    bottom_row = n_rows - 1
    for pi in range(n_panels):
        if pi < len(all_agg) and not all_agg[pi].empty:
            panel_datasets = list(all_agg[pi]["DatasetLabel"].unique())
        else:
            panel_datasets = []
        axes[bottom_row, pi].set_xticks(np.arange(len(panel_datasets)))
        axes[bottom_row, pi].set_xticklabels(
            panel_datasets,
            rotation=30,
            ha="right",
            fontsize=_SHIFT_XTICK_FONTSIZE,
        )

    # Shared legend on top
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

    fig.suptitle(axis_title, fontsize=_TITLE_FONTSIZE)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.12, wspace=0.16)
    fig.savefig(output_path)
    plt.close(fig)

    result = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    save_figure_data(output_path, result)
    return result


# ── ID baseline restriction helpers ────────────────────────────────────


def _restrict_id_to_anchor(
    subset: pd.DataFrame, *, mech_key: str | None, graph_code: str | None
) -> pd.DataFrame:
    """Keep only the ID dataset matching the given mechanism and graph anchor.

    For mechanism shift panels, ``mech_key`` is ``None`` (mechanism is OOD),
    so we match only on graph.  For noise shift, both are set.
    """
    id_mask = subset["AxisCategory"] == "id"

    def _keep(dk: str) -> bool:
        if graph_code is not None and graph_code_of(dk) != graph_code:
            return False
        if mech_key is not None and id_mechanism_of(dk) != mech_key:
            return False
        return True

    keep_id = id_mask & subset["DatasetKey"].map(_keep)
    return subset[keep_id | ~id_mask].copy()


def _restrict_to_graph_anchor(subset: pd.DataFrame, graph_code: str) -> pd.DataFrame:
    """Keep only rows whose graph code matches *graph_code*.

    Unlike :func:`_restrict_id_to_anchor`, this filters **all** rows (ID and
    OOD alike), ensuring mechanism-shift families with different graph anchors
    are not mixed in summary panels that use a single representative anchor.
    """
    return subset[
        subset["DatasetKey"].map(lambda dk: graph_code_of(dk) == graph_code)
    ].copy()


# ── Mechanism-shift multi-panel helpers (one panel per graph anchor) ───

_MECH_SHIFT_GRAPH_ANCHORS: list[str] = ["er20", "er60", "sf2"]
"""Graph anchors used for mechanism-shift panels."""


def _mech_shift_label(dataset_key: str) -> str:
    """Short mechanism label for mechanism-shift panels.

    OOD-mechanism datasets return the OOD mechanism name (e.g. ``"Periodic"``).
    ID datasets return the ID mechanism without suffix.
    """
    from causal_meta.analysis.utils import MECH_DESCRIPTION_MAP

    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)

    if dk.startswith("ood_mech_"):
        # Strip prefix and graph anchor to get mechanism code
        remainder = body[len("ood_mech_") :]
        # Remove trailing graph anchor
        for code in sorted(GRAPH_ANCHOR_LABELS.keys(), key=len, reverse=True):
            if remainder.endswith(f"_{code}"):
                remainder = remainder[: -len(code) - 1]
                break
        return MECH_DESCRIPTION_MAP.get(remainder, remainder.replace("_", " ").title())

    # ID datasets — show mechanism name
    for mech_key in ("neuralnet", "gpcde", "linear"):
        if f"_{mech_key}" in body:
            return MECH_DESCRIPTION_MAP.get(mech_key, mech_key)

    return dataset_key


def _generate_mech_shift_panels(
    subset: pd.DataFrame,
    metric_name: str,
    axis_title: str,
    output_path: Path,
    *,
    model_filter: Sequence[str] | None = None,
    avici_dag_df: pd.DataFrame | None = None,
    error_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate a multi-panel mechanism shift figure, one column per graph anchor.

    Three-row layout when auxiliary data is supplied:
      Row 0 (top):    AviCi DAG validity (sampled + thresholded).
      Row 1 (middle): Primary metric (E-SID / NE-SID).
      Row 2 (bottom): Error decomposition (FP / FN / Reversed).
    """
    subset = subset.copy()
    subset["_graph_anchor"] = subset["DatasetKey"].map(mech_shift_graph_anchor)
    subset.loc[subset["AxisCategory"] == "id", "_graph_anchor"] = subset.loc[
        subset["AxisCategory"] == "id", "DatasetKey"
    ].map(graph_code_of)

    anchors = [
        a for a in _MECH_SHIFT_GRAPH_ANCHORS if a in subset["_graph_anchor"].unique()
    ]
    if not anchors:
        return pd.DataFrame()

    has_dag = avici_dag_df is not None and not avici_dag_df.empty
    has_err = error_df is not None and not error_df.empty

    n_panels = len(anchors)
    models = _resolve_models(model_filter)
    n_models = len(models)

    # Tag auxiliary DataFrames with the same anchor.
    dag_sub: pd.DataFrame | None = None
    if has_dag:
        dag_sub = avici_dag_df.copy()
        dag_sub["_graph_anchor"] = dag_sub["DatasetKey"].map(mech_shift_graph_anchor)
        dag_sub.loc[dag_sub["AxisCategory"] == "id", "_graph_anchor"] = dag_sub.loc[
            dag_sub["AxisCategory"] == "id", "DatasetKey"
        ].map(graph_code_of)
    err_sub: pd.DataFrame | None = None
    if has_err:
        err_sub = error_df.copy()
        err_sub["_graph_anchor"] = err_sub["DatasetKey"].map(mech_shift_graph_anchor)
        err_sub.loc[err_sub["AxisCategory"] == "id", "_graph_anchor"] = err_sub.loc[
            err_sub["AxisCategory"] == "id", "DatasetKey"
        ].map(graph_code_of)

    # Build row structure: DAG (optional) | metric | error (optional).
    height_ratios: list[float] = []
    row_names: list[str] = []
    if has_dag:
        height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[0])
        row_names.append("dag")
    height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[1])
    row_names.append("metric")
    if has_err:
        height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[2])
        row_names.append("error")
    n_rows = len(row_names)

    fig, axes = plt.subplots(
        n_rows,
        n_panels,
        figsize=_three_row_figsize(n_panels),
        sharex="col",
        sharey=False,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios, "wspace": 0.16},
    )

    metric_row = row_names.index("metric")
    dag_row = row_names.index("dag") if has_dag else None
    err_row = row_names.index("error") if has_err else None

    all_agg: list[pd.DataFrame] = []

    for panel_idx, anchor in enumerate(anchors):
        ax = axes[metric_row, panel_idx]
        ood_mask = (subset["AxisCategory"] == "mechanism") & (
            subset["_graph_anchor"] == anchor
        )
        id_mask = (subset["AxisCategory"] == "id") & (subset["_graph_anchor"] == anchor)
        panel_data = subset[ood_mask | id_mask]

        if panel_data.empty:
            for r in range(n_rows):
                axes[r, panel_idx].set_visible(False)
            continue

        agg = (
            panel_data.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["DatasetLabel"] = agg["DatasetKey"].map(_mech_shift_label)
        agg["_sort"] = agg["AxisCategory"].map({"id": 0}).fillna(1)
        agg = agg.sort_values(["_sort", "DatasetLabel"]).drop(columns=["_sort"])
        all_agg.append(agg)

        datasets = list(agg["DatasetLabel"].unique())
        axis_lookup = (
            agg[["DatasetLabel", "AxisCategory"]]
            .drop_duplicates()
            .set_index("DatasetLabel")["AxisCategory"]
            .to_dict()
        )
        n_datasets = len(datasets)
        width = 0.6
        offset_step = width / max(n_models, 1)
        x_base = np.arange(n_datasets)
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")

        # ── Metric row ────────────────────────────────────────────────
        for model_idx, model in enumerate(models):
            model_agg = agg[agg["Model"] == model]
            xs: list[float] = []
            means: list[float] = []
            sems: list[float] = []
            for i, ds in enumerate(datasets):
                row = model_agg[model_agg["DatasetLabel"] == ds]
                if row.empty:
                    continue
                offset = (model_idx - n_models / 2 + 0.5) * offset_step
                xs.append(float(x_base[i]) + offset)
                means.append(float(row.iloc[0]["Mean"]))
                sems.append(float(row.iloc[0]["SEM"]))
            if xs:
                ax.errorbar(
                    xs,
                    means,
                    yerr=sems,
                    fmt=MODEL_MARKERS.get(model, "o"),
                    label=model if panel_idx == 0 else None,
                    color=_model_color(model),
                    capsize=3,
                    markersize=7,
                    alpha=0.9,
                )

        anchor_label = GRAPH_ANCHOR_LABELS.get(anchor, anchor.upper())
        # Column header on the topmost row (dag if present, else metric).
        axes[0, panel_idx].set_title(
            f"Graph: {anchor_label}", fontsize=_PANEL_TITLE_FONTSIZE
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if panel_idx == 0:
            ylabel = _sid_ylabel(metric_name)
            ax.set_ylabel(ylabel, fontsize=_LABEL_FONTSIZE)

        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

        # ── DAG row (top) ─────────────────────────────────────────────
        if dag_row is not None:
            ood_dag = (dag_sub["AxisCategory"] == "mechanism") & (
                dag_sub["_graph_anchor"] == anchor
            )
            id_dag = (dag_sub["AxisCategory"] == "id") & (
                dag_sub["_graph_anchor"] == anchor
            )
            dag_panel = (
                dag_sub[ood_dag | id_dag] if dag_sub is not None else pd.DataFrame()
            )
            _plot_dag_row_panel(
                axes[dag_row, panel_idx],
                dag_panel,
                datasets,
                x_base,
                _mech_shift_label,
                panel_idx=panel_idx,
                id_count=id_count,
            )

        # ── Error row (bottom) ────────────────────────────────────────
        if err_row is not None:
            ood_err = (err_sub["AxisCategory"] == "mechanism") & (
                err_sub["_graph_anchor"] == anchor
            )
            id_err = (err_sub["AxisCategory"] == "id") & (
                err_sub["_graph_anchor"] == anchor
            )
            err_panel = (
                err_sub[ood_err | id_err] if err_sub is not None else pd.DataFrame()
            )
            _plot_error_row_panel(
                axes[err_row, panel_idx],
                err_panel,
                datasets,
                x_base,
                models,
                _mech_shift_label,
                panel_idx=panel_idx,
                id_count=id_count,
            )

    # X tick labels on the bottom row only.
    bottom_row = n_rows - 1
    for pi in range(n_panels):
        if pi < len(all_agg) and not all_agg[pi].empty:
            panel_datasets = list(all_agg[pi]["DatasetLabel"].unique())
        else:
            panel_datasets = []
        axes[bottom_row, pi].set_xticks(np.arange(len(panel_datasets)))
        axes[bottom_row, pi].set_xticklabels(
            panel_datasets,
            rotation=30,
            ha="right",
            fontsize=_SHIFT_XTICK_FONTSIZE,
        )

    # Shared legend on top
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

    fig.suptitle(axis_title, fontsize=_TITLE_FONTSIZE)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.12, wspace=0.16)
    fig.savefig(output_path)
    plt.close(fig)

    result = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    save_figure_data(output_path, result)
    return result


# ── Noise-shift multi-panel helpers (one panel per anchor) ─────────────

_NOISE_SHIFT_ANCHORS: list[tuple[str, str]] = [
    ("linear", "er20"),
    ("neuralnet", "sf2"),
]
"""(mechanism, graph_code) anchors used for noise-shift panels."""


def _noise_shift_label(dataset_key: str) -> str:
    """Short noise label for noise-shift panels.

    Returns ``"Laplace"``, ``"Uniform"``, or ``"Gaussian"``.
    """
    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)

    if dk.startswith("ood_noise_"):
        remainder = body[len("ood_noise_") :]
        # First token is the noise type
        noise_type = remainder.split("_")[0]
        return noise_type.title()

    # ID baseline
    return "Gaussian"


def _generate_noise_shift_panels(
    subset: pd.DataFrame,
    metric_name: str,
    axis_title: str,
    output_path: Path,
    *,
    model_filter: Sequence[str] | None = None,
    avici_dag_df: pd.DataFrame | None = None,
    error_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate a multi-panel noise shift figure, one column per anchor.

    Three-row layout when auxiliary data is supplied:
      Row 0 (top):    AviCi DAG validity (sampled + thresholded).
      Row 1 (middle): Primary metric (E-SID / NE-SID).
      Row 2 (bottom): Error decomposition (FP / FN / Reversed).
    """
    subset = subset.copy()
    subset["_noise_anchor"] = subset["DatasetKey"].map(noise_shift_anchor)
    id_mask = subset["AxisCategory"] == "id"
    subset.loc[id_mask, "_noise_anchor"] = subset.loc[id_mask, "DatasetKey"].map(
        lambda dk: (
            (id_mechanism_of(dk), graph_code_of(dk))
            if id_mechanism_of(dk) is not None
            else None
        )
    )

    present_anchors = [
        a
        for a in _NOISE_SHIFT_ANCHORS
        if a in set(subset["_noise_anchor"].dropna().tolist())
    ]
    if not present_anchors:
        return pd.DataFrame()

    has_dag = avici_dag_df is not None and not avici_dag_df.empty
    has_err = error_df is not None and not error_df.empty

    n_panels = len(present_anchors)
    models = _resolve_models(model_filter)
    n_models = len(models)

    def _tag_noise_anchor(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_noise_anchor"] = df["DatasetKey"].map(noise_shift_anchor)
        _id = df["AxisCategory"] == "id"
        df.loc[_id, "_noise_anchor"] = df.loc[_id, "DatasetKey"].map(
            lambda dk: (
                (id_mechanism_of(dk), graph_code_of(dk))
                if id_mechanism_of(dk) is not None
                else None
            )
        )
        return df

    dag_sub = _tag_noise_anchor(avici_dag_df) if has_dag else None
    err_sub = _tag_noise_anchor(error_df) if has_err else None

    # Build row structure: DAG (optional) | metric | error (optional).
    height_ratios: list[float] = []
    row_names: list[str] = []
    if has_dag:
        height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[0])
        row_names.append("dag")
    height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[1])
    row_names.append("metric")
    if has_err:
        height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[2])
        row_names.append("error")
    n_rows = len(row_names)

    fig, axes = plt.subplots(
        n_rows,
        n_panels,
        figsize=_three_row_figsize(n_panels, match_width_cols=3),
        sharex="col",
        sharey=False,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    metric_row = row_names.index("metric")
    dag_row = row_names.index("dag") if has_dag else None
    err_row = row_names.index("error") if has_err else None

    all_agg: list[pd.DataFrame] = []

    for panel_idx, anchor in enumerate(present_anchors):
        ax = axes[metric_row, panel_idx]
        panel_data = subset[subset["_noise_anchor"] == anchor]

        if panel_data.empty:
            for r in range(n_rows):
                axes[r, panel_idx].set_visible(False)
            continue

        agg = (
            panel_data.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["DatasetLabel"] = agg["DatasetKey"].map(_noise_shift_label)
        agg["_sort"] = agg["AxisCategory"].map({"id": 0}).fillna(1)
        agg = agg.sort_values(["_sort", "DatasetLabel"]).drop(columns=["_sort"])
        all_agg.append(agg)

        datasets = list(agg["DatasetLabel"].unique())
        axis_lookup = (
            agg[["DatasetLabel", "AxisCategory"]]
            .drop_duplicates()
            .set_index("DatasetLabel")["AxisCategory"]
            .to_dict()
        )
        n_datasets = len(datasets)
        width = 0.6
        offset_step = width / max(n_models, 1)
        x_base = np.arange(n_datasets)
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")

        # ── Metric row ────────────────────────────────────────────────
        for model_idx, model in enumerate(models):
            model_agg = agg[agg["Model"] == model]
            xs: list[float] = []
            means: list[float] = []
            sems: list[float] = []
            for i, ds in enumerate(datasets):
                row = model_agg[model_agg["DatasetLabel"] == ds]
                if row.empty:
                    continue
                offset = (model_idx - n_models / 2 + 0.5) * offset_step
                xs.append(float(x_base[i]) + offset)
                means.append(float(row.iloc[0]["Mean"]))
                sems.append(float(row.iloc[0]["SEM"]))
            if xs:
                ax.errorbar(
                    xs,
                    means,
                    yerr=sems,
                    fmt=MODEL_MARKERS.get(model, "o"),
                    label=model if panel_idx == 0 else None,
                    color=_model_color(model),
                    capsize=3,
                    markersize=7,
                    alpha=0.9,
                )

        mech_label = ID_MECHANISM_LABELS.get(anchor[0], anchor[0])
        graph_label = GRAPH_ANCHOR_LABELS.get(anchor[1], anchor[1].upper())
        # Column header on the topmost row (dag if present, else metric).
        axes[0, panel_idx].set_title(
            f"{mech_label} / {graph_label}",
            fontsize=_PANEL_TITLE_FONTSIZE,
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if panel_idx == 0:
            ylabel = _sid_ylabel(metric_name)
            ax.set_ylabel(ylabel, fontsize=_LABEL_FONTSIZE)

        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

        # ── DAG row (top) ─────────────────────────────────────────────
        if dag_row is not None:
            dag_panel = (
                dag_sub[dag_sub["_noise_anchor"] == anchor]
                if dag_sub is not None
                else pd.DataFrame()
            )
            _plot_dag_row_panel(
                axes[dag_row, panel_idx],
                dag_panel,
                datasets,
                x_base,
                _noise_shift_label,
                panel_idx=panel_idx,
                id_count=id_count,
            )

        # ── Error row (bottom) ────────────────────────────────────────
        if err_row is not None:
            err_panel = (
                err_sub[err_sub["_noise_anchor"] == anchor]
                if err_sub is not None
                else pd.DataFrame()
            )
            _plot_error_row_panel(
                axes[err_row, panel_idx],
                err_panel,
                datasets,
                x_base,
                models,
                _noise_shift_label,
                panel_idx=panel_idx,
                id_count=id_count,
            )

    # X tick labels on the bottom row only.
    bottom_row = n_rows - 1
    for pi in range(n_panels):
        if pi < len(all_agg) and not all_agg[pi].empty:
            panel_datasets = list(all_agg[pi]["DatasetLabel"].unique())
        else:
            panel_datasets = []
        axes[bottom_row, pi].set_xticks(np.arange(len(panel_datasets)))
        axes[bottom_row, pi].set_xticklabels(
            panel_datasets,
            rotation=30,
            ha="right",
            fontsize=_SHIFT_XTICK_FONTSIZE,
        )

    # Shared legend on top
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

    fig.suptitle(axis_title, fontsize=_TITLE_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    result = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    save_figure_data(output_path, result)
    return result


# ── Compound-shift multi-panel helpers ─────────────────────────────────


def _compound_shift_label(dataset_key: str) -> str:
    """Short label for compound-shift panels.

    OOD-compound datasets show only the OOD mechanism label because the OOD graph
    is fixed by the panel title. ID representatives show their graph/mechanism
    anchor without suffix. Extreme stress-test families get their
    special multi-line labels from :data:`_EXTREME_LABELS`.
    """
    # Stress-test families have dedicated rich labels.
    if dataset_key in _EXTREME_LABELS:
        return _EXTREME_LABELS[dataset_key]

    dk = dataset_key.lower()

    if dk.startswith("ood_both_"):
        mech_code = _ood_mech_code(dk)
        if mech_code is not None:
            graph_code = graph_code_of(dk)
            if graph_code is not None:
                return _graph_mech_combo_label(
                    graph_code=graph_code, mech_key=mech_code
                )

        # Fallback
        body = re.sub(r"_d\d+_n\d+$", "", dk)
        remainder = body[len("ood_both_") :]
        graph_code = graph_code_of(dk)
        if graph_code is not None and remainder.startswith(f"{graph_code}_"):
            remainder = remainder[len(graph_code) + 1 :]
        return MECH_DESCRIPTION_MAP.get(remainder, remainder.replace("_", " ").title())

    mech_key = id_mechanism_of(dk)
    graph_code = graph_code_of(dk)
    if mech_key is None or graph_code is None:
        return dataset_key
    return _graph_mech_combo_label(graph_code=graph_code, mech_key=mech_key)


def _compound_shift_sort_key(dataset_key: str) -> tuple[int, int, str]:
    """Sort ID representatives first, then OOD mechanisms in a stable order."""
    dk = dataset_key.lower()
    mech_key = id_mechanism_of(dk)
    graph_code = graph_code_of(dk)
    anchor = (mech_key, graph_code)

    if dk.startswith("id_") and anchor in _COMPOUND_ID_REPRESENTATIVES:
        return (0, _COMPOUND_ID_REPRESENTATIVES.index(anchor), dk)

    if dk.startswith("ood_both_"):
        body = re.sub(r"_d\d+_n\d+$", "", dk)
        remainder = body[len("ood_both_") :]
        ood_graph = graph_code_of(dk)
        if ood_graph is not None and remainder.startswith(f"{ood_graph}_"):
            remainder = remainder[len(ood_graph) + 1 :]
        return (1, _COMPOUND_OOD_MECH_ORDER.get(remainder, 999), dk)

    return (2, 999, dk)


def _generate_compound_shift_panels(
    subset: pd.DataFrame,
    metric_name: str,
    axis_title: str,
    output_path: Path,
    *,
    stress_df: pd.DataFrame | None = None,
    model_filter: Sequence[str] | None = None,
    avici_dag_df: pd.DataFrame | None = None,
    error_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate a multi-panel compound-shift figure, one column per OOD graph.

    Three-row layout when auxiliary data is supplied:
      Row 0 (top):    AviCi DAG validity (sampled + thresholded).
      Row 1 (middle): Primary metric (NE-SID).
      Row 2 (bottom): Error decomposition (FP / FN / Reversed).

    When *stress_df* is provided the extreme stress-test families are appended
    after a second vertical divider, showing the ID -> compound -> stress
    progression in a single figure.
    """
    subset = subset.copy()
    id_mask = subset["AxisCategory"] == "id"
    keep_id = id_mask & subset["DatasetKey"].map(
        lambda dk: (
            (id_mechanism_of(dk), graph_code_of(dk)) in _COMPOUND_ID_REPRESENTATIVES
        )
    )
    subset = subset[keep_id | ~id_mask].copy()

    present_graphs = [
        graph_code
        for graph_code in _COMPOUND_OOD_GRAPHS
        if (
            (subset["AxisCategory"] == "compound")
            & subset["DatasetKey"].map(lambda dk: graph_code_of(dk) == graph_code)
        ).any()
    ]
    if not present_graphs:
        return pd.DataFrame()

    # Build a lookup from graph_code -> stress-test family key
    stress_by_graph: dict[str, pd.DataFrame] = {}
    if stress_df is not None and not stress_df.empty:
        for fam_key in _EXTREME_FAMILIES:
            gc = graph_code_of(fam_key)
            if gc is not None:
                fam_rows = stress_df[stress_df["DatasetKey"] == fam_key]
                if not fam_rows.empty:
                    stress_by_graph[gc] = fam_rows

    has_dag = avici_dag_df is not None and not avici_dag_df.empty
    has_err = error_df is not None and not error_df.empty

    n_panels = len(present_graphs)
    models = _resolve_models(model_filter)
    n_models = len(models)

    # Prepare auxiliary DataFrames with the same ID-representative filter.
    def _filter_compound_id(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        _id = df["AxisCategory"] == "id"
        _keep = _id & df["DatasetKey"].map(
            lambda dk: (
                (id_mechanism_of(dk), graph_code_of(dk)) in _COMPOUND_ID_REPRESENTATIVES
            )
        )
        return df[_keep | ~_id].copy()

    dag_sub = _filter_compound_id(avici_dag_df) if has_dag else None
    err_sub = _filter_compound_id(error_df) if has_err else None

    # Build row structure: DAG (optional) | metric | error (optional).
    height_ratios: list[float] = []
    row_names: list[str] = []
    if has_dag:
        height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[0])
        row_names.append("dag")
    height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[1])
    row_names.append("metric")
    if has_err:
        height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[2])
        row_names.append("error")
    n_rows = len(row_names)

    fig, axes = plt.subplots(
        n_rows,
        n_panels,
        figsize=_three_row_figsize(n_panels),
        sharex="col",
        sharey=False,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    metric_row = row_names.index("metric")
    dag_row = row_names.index("dag") if has_dag else None
    err_row = row_names.index("error") if has_err else None

    all_agg: list[pd.DataFrame] = []

    for panel_idx, graph_code in enumerate(present_graphs):
        ax = axes[metric_row, panel_idx]
        ood_mask = (subset["AxisCategory"] == "compound") & subset["DatasetKey"].map(
            lambda dk: graph_code_of(dk) == graph_code
        )
        panel_data = subset[ood_mask | (subset["AxisCategory"] == "id")].copy()

        if panel_data.empty:
            for r in range(n_rows):
                axes[r, panel_idx].set_visible(False)
            continue

        agg = (
            panel_data.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["DatasetLabel"] = agg["DatasetKey"].map(_compound_shift_label)
        agg["FixedConcept"] = GRAPH_ANCHOR_LABELS.get(graph_code, graph_code.upper())
        agg["SortKey"] = agg["DatasetKey"].map(_compound_shift_sort_key)
        agg = agg.sort_values("SortKey").drop(columns=["SortKey"])

        # ── Append stress-test rows if present for this graph ─────────
        stress_agg: pd.DataFrame | None = None
        if graph_code in stress_by_graph:
            stress_panel = stress_by_graph[graph_code]
            stress_agg = (
                stress_panel.groupby(["Model", "DatasetKey"], dropna=False)["Value"]
                .agg(Mean="mean", SEM=metric_sem)
                .reset_index()
            )
            stress_agg["AxisCategory"] = "stress"
            stress_agg["Dataset"] = stress_agg["DatasetKey"]
            stress_agg["DatasetLabel"] = stress_agg["DatasetKey"].map(
                lambda dk: _EXTREME_LABELS.get(dk, dk)
            )
            stress_agg["FixedConcept"] = GRAPH_ANCHOR_LABELS.get(
                graph_code, graph_code.upper()
            )
            agg = pd.concat([agg, stress_agg], ignore_index=True)

        all_agg.append(agg)

        datasets = list(agg["DatasetLabel"].unique())
        axis_lookup = (
            agg[["DatasetLabel", "AxisCategory"]]
            .drop_duplicates()
            .set_index("DatasetLabel")["AxisCategory"]
            .to_dict()
        )
        n_datasets = len(datasets)
        width = 0.6
        offset_step = width / max(n_models, 1)
        x_base = np.arange(n_datasets)
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")

        # ── Metric row ────────────────────────────────────────────────
        for model_idx, model in enumerate(models):
            model_agg = agg[agg["Model"] == model]
            xs: list[float] = []
            means: list[float] = []
            sems: list[float] = []
            for i, ds in enumerate(datasets):
                row = model_agg[model_agg["DatasetLabel"] == ds]
                if row.empty:
                    continue
                offset = (model_idx - n_models / 2 + 0.5) * offset_step
                xs.append(float(x_base[i]) + offset)
                means.append(float(row.iloc[0]["Mean"]))
                sems.append(float(row.iloc[0]["SEM"]))
            if xs:
                ax.errorbar(
                    xs,
                    means,
                    yerr=sems,
                    fmt=MODEL_MARKERS.get(model, "o"),
                    label=model if panel_idx == 0 else None,
                    color=_model_color(model),
                    capsize=3,
                    markersize=7,
                    alpha=0.9,
                )

        # Column header on the topmost row (dag if present, else metric).
        axes[0, panel_idx].set_title(
            f"Graph: {GRAPH_ANCHOR_LABELS.get(graph_code, graph_code.upper())}",
            fontsize=_PANEL_TITLE_FONTSIZE,
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if panel_idx == 0:
            ylabel = _sid_ylabel(metric_name)
            ax.set_ylabel(ylabel, fontsize=_LABEL_FONTSIZE)

        # Grey ID region
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

        # Stress-test region (light red tint + second divider)
        stress_count = 0
        if stress_agg is not None and not stress_agg.empty:
            stress_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "stress")
            if stress_count > 0:
                stress_start = n_datasets - stress_count
                ax.axvspan(
                    stress_start - 0.5,
                    n_datasets - 0.5,
                    color="#ffe0e0",
                    alpha=0.35,
                    zorder=0,
                )
                ax.axvline(
                    stress_start - 0.5,
                    color="#cc4444",
                    linestyle=":",
                    linewidth=1.0,
                )

        # Stress-test dataset keys for this graph (used in DAG/error rows too).
        stress_keys = (
            set(stress_by_graph[graph_code]["DatasetKey"].unique())
            if graph_code in stress_by_graph
            else set()
        )

        # ── DAG row (top) ─────────────────────────────────────────────
        if dag_row is not None:
            dag_ood = (dag_sub["AxisCategory"] == "compound") & dag_sub[
                "DatasetKey"
            ].map(lambda dk: graph_code_of(dk) == graph_code)
            dag_id = dag_sub["AxisCategory"] == "id"
            # Include stress-test families for this graph in the DAG row.
            dag_stress = dag_sub["DatasetKey"].isin(stress_keys)
            dag_panel = (
                dag_sub[dag_ood | dag_id | dag_stress]
                if dag_sub is not None
                else pd.DataFrame()
            )
            _plot_dag_row_panel(
                axes[dag_row, panel_idx],
                dag_panel,
                datasets,
                x_base,
                _compound_shift_label,
                panel_idx=panel_idx,
                id_count=id_count,
                stress_count=stress_count,
            )

        # ── Error row (bottom) ────────────────────────────────────────
        if err_row is not None:
            err_ood = (err_sub["AxisCategory"] == "compound") & err_sub[
                "DatasetKey"
            ].map(lambda dk: graph_code_of(dk) == graph_code)
            err_id = err_sub["AxisCategory"] == "id"
            # Include stress-test families for this graph in the error row.
            err_stress = err_sub["DatasetKey"].isin(stress_keys)
            err_panel = (
                err_sub[err_ood | err_id | err_stress]
                if err_sub is not None
                else pd.DataFrame()
            )
            _plot_error_row_panel(
                axes[err_row, panel_idx],
                err_panel,
                datasets,
                x_base,
                models,
                _compound_shift_label,
                panel_idx=panel_idx,
                id_count=id_count,
                stress_count=stress_count,
            )

    # X tick labels on the bottom row only.
    bottom_row = n_rows - 1
    for pi in range(n_panels):
        if pi < len(all_agg) and not all_agg[pi].empty:
            panel_datasets = list(all_agg[pi]["DatasetLabel"].unique())
        else:
            panel_datasets = []
        axes[bottom_row, pi].set_xticks(np.arange(len(panel_datasets)))
        axes[bottom_row, pi].set_xticklabels(
            panel_datasets,
            rotation=30,
            ha="right",
            fontsize=_SHIFT_XTICK_FONTSIZE,
        )

    # Shared legend on top
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

    fig.suptitle(axis_title, fontsize=_TITLE_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    result = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    save_figure_data(output_path, result)
    return result


# ── RQ2: Degradation heatmap ──────────────────────────────────────────

_DEGRADATION_SHIFT_AXES: dict[str, str] = {
    "graph": "Graph",
    "mechanism": "Mechanism",
    "noise": "Noise",
    "compound": "Graph-Mechanism",
}
"""Shift axes included in the RQ2 degradation heatmap."""


def _degradation_id_subset(
    full_df: pd.DataFrame,
    shift_key: str,
) -> pd.DataFrame:
    """Return the ID rows that form the correct baseline for *shift_key*.

    Each shift axis uses a different anchored ID subset so that the
    degradation ratio isolates only the varied dimension:

    * **graph** -- ID families with Linear mechanism (graph is the only
      variable in the graph-shift figure).
    * **mechanism** -- ID families on ER-20 (mechanism is the only variable).
    * **noise** -- all fixed-size ID families (both anchors participate).
    * **compound** -- the three representative ID anchor families.
    * **nodes** -- fixed-size ID families on the two transfer anchors.
    * **samples** -- same as *nodes*.
    """
    id_data = full_df[full_df["AxisCategory"] == "id"].copy()

    if shift_key == "graph":
        return id_data[
            id_data["DatasetKey"].map(lambda dk: id_mechanism_of(dk) == "linear")
        ]
    if shift_key == "mechanism":
        return _restrict_to_graph_anchor(id_data, "er20")
    if shift_key == "compound":
        return id_data[
            id_data["DatasetKey"].map(
                lambda dk: (
                    (id_mechanism_of(dk), graph_code_of(dk))
                    in _COMPOUND_ID_REPRESENTATIVES
                )
            )
        ]
    if shift_key == "stress":
        # Stress test = compound shift + extreme sizes.
        # Use the same ID baseline as compound shift.
        return id_data[
            id_data["DatasetKey"].map(
                lambda dk: (
                    (id_mechanism_of(dk), graph_code_of(dk))
                    in _COMPOUND_ID_REPRESENTATIVES
                )
            )
        ]
    if shift_key in ("nodes", "samples"):
        # Transfer anchors: (linear, er20), (neuralnet, sf2)
        _TRANSFER_ANCHORS = {("linear", "er20"), ("neuralnet", "sf2")}
        return id_data[
            id_data["DatasetKey"].map(
                lambda dk: (id_mechanism_of(dk), graph_code_of(dk)) in _TRANSFER_ANCHORS
            )
        ]
    # noise and any other: all fixed-size ID
    return id_data


def _degradation_family_id_subset(
    full_df: pd.DataFrame,
    shift_key: str,
    dataset_key: str,
) -> pd.DataFrame:
    """Return ID rows for the anchor matching one OOD family."""
    id_data = full_df[full_df["AxisCategory"] == "id"].copy()

    if shift_key == "graph":
        mech_key = id_mechanism_of(dataset_key)
        if mech_key is not None:
            return id_data[
                id_data["DatasetKey"].map(lambda dk: id_mechanism_of(dk) == mech_key)
            ]
    elif shift_key == "mechanism":
        graph_code = mech_shift_graph_anchor(dataset_key)
        if graph_code is not None:
            return _restrict_to_graph_anchor(id_data, graph_code)
    elif shift_key == "noise":
        anchor = noise_shift_anchor(dataset_key)
        if anchor is not None:
            mech_key, graph_code = anchor
            return id_data[
                id_data["DatasetKey"].map(
                    lambda dk: (
                        id_mechanism_of(dk) == mech_key
                        and graph_code_of(dk) == graph_code
                    )
                )
            ]
    elif shift_key in ("nodes", "samples"):
        anchor = transfer_anchor(dataset_key)
        if anchor is not None:
            mech_key, graph_code = anchor
            return id_data[
                id_data["DatasetKey"].map(
                    lambda dk: (
                        id_mechanism_of(dk) == mech_key
                        and graph_code_of(dk) == graph_code
                    )
                )
            ]
    elif shift_key in ("compound", "stress"):
        graph_code = graph_code_of(dataset_key)
        graph_anchor = {
            ood_graph: id_graph
            for _, id_graph, ood_graph, _ in _COMPOUND_ISOLATION_TRIPLETS
        }.get(graph_code)
        if graph_anchor is not None:
            return id_data[
                id_data["DatasetKey"].map(
                    lambda dk: (
                        id_mechanism_of(dk) == "neuralnet"
                        and graph_code_of(dk) == graph_anchor
                    )
                )
            ]

    return _degradation_id_subset(full_df, shift_key)


def generate_degradation_heatmap(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate a heatmap showing per-model degradation ratio across shift axes.

    Each cell shows the mean per-family ``OOD ne-SID / ID ne-SID`` ratio for a
    given model and shift axis, with the standard deviation (SD) of per-family
    ratios shown below.  The ID baseline is matched to each OOD family so that
    each ratio isolates only the varied distributional dimension.

    A ratio of 1.0 means no degradation; higher values indicate worse OOD
    performance relative to the relevant ID baseline.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output PDF figure.
        model_filter: If given, only include these model display names.

    Returns:
        Pivot table (models x shift axes) of degradation ratios.
    """
    import matplotlib.colors as mcolors

    metric_name = "ne-sid"
    subset = raw_df[raw_df["Metric"].eq(metric_name)].copy()
    # All shift axes in the heatmap use fixed-size tasks.
    fixed_subset = subset[is_fixed_size_task_frame(subset)]
    if fixed_subset.empty:
        log.warning("No ne-sid data for degradation heatmap; skipping.")
        return pd.DataFrame()

    models = _resolve_models(model_filter)

    rows: list[dict[str, object]] = []
    for shift_key, shift_label in _DEGRADATION_SHIFT_AXES.items():
        pool = fixed_subset
        ood_data = pool[pool["AxisCategory"] == shift_key]
        family_ratios_by_model: dict[str, list[float]] = {model: [] for model in models}

        for dataset_key, family_df in ood_data.groupby("DatasetKey", dropna=False):
            id_data = _degradation_family_id_subset(
                fixed_subset, shift_key, str(dataset_key)
            )
            if id_data.empty:
                continue
            id_means = id_data.groupby("Model")["Value"].mean().to_dict()
            ood_means = family_df.groupby("Model")["Value"].mean().to_dict()
            for model in models:
                id_val = id_means.get(model)
                ood_val = ood_means.get(model)
                if id_val is None or ood_val is None or id_val <= 0:
                    continue
                family_ratios_by_model[model].append(float(ood_val) / float(id_val))

        for model in models:
            family_ratios = pd.Series(family_ratios_by_model[model], dtype="float64")
            if family_ratios.empty:
                ratio = float("nan")
                sd = float("nan")
            else:
                ratio = float(family_ratios.mean())
                sd = float(family_ratios.std(ddof=1)) if len(family_ratios) > 1 else 0.0

            rows.append(
                {
                    "Model": model,
                    "Shift": shift_label,
                    "Ratio": ratio,
                    "SD": sd,
                }
            )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        log.warning("No data for degradation heatmap; skipping.")
        return pd.DataFrame()

    pivot = result_df.pivot(index="Model", columns="Shift", values="Ratio")
    pivot_sd = result_df.pivot(index="Model", columns="Shift", values="SD")

    # Reorder rows and columns
    pivot = pivot.reindex(index=[m for m in models if m in pivot.index])
    pivot = pivot.reindex(
        columns=[v for v in _DEGRADATION_SHIFT_AXES.values() if v in pivot.columns]
    )
    pivot_sd = pivot_sd.reindex(index=pivot.index, columns=pivot.columns)

    fig, ax = plt.subplots(
        figsize=(HEATMAP_FIGSIZE[0] * 0.74, HEATMAP_FIGSIZE[1] * 0.74)
    )

    # Diverging colormap around 1.0 (no degradation)
    raw_vmin = max(0.5, float(pivot.min().min()) - 0.1) if not pivot.empty else 0.5
    raw_vmax = max(float(pivot.max().max()) + 0.1, 1.5) if not pivot.empty else 2.0
    # Keep color scale visually balanced around 1.0.
    span = max(1.0 - raw_vmin, raw_vmax - 1.0)
    vmin = max(0.5, 1.0 - span)
    vmax = 1.0 + span
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # red = worse (high ratio), green = good (low ratio)

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=4.5, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=4.5)

    # Annotate cells with ratio + SD spread.
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            sd_val = pivot_sd.values[i, j]
            if np.isnan(val):
                text = "--"
            else:
                text = f"{val:.2f}"
                if not np.isnan(sd_val):
                    text += f"\n({sd_val:.2f})"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=3.6,
                fontweight="normal",
                color="white" if val > (vmin + vmax) / 2 else "black",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.03, fraction=0.055, aspect=28)
    cbar.set_label("OOD / ID ratio", fontsize=4)
    cbar.ax.tick_params(labelsize=4)

    ax.set_title(
        rf"Normalized ${_HAT_E}$-SID Degradation Ratio (OOD / ID)",
        fontsize=6,
        pad=4,
    )
    fig.subplots_adjust(left=0.22, right=0.90, bottom=0.20, top=0.88)
    fig.savefig(output_path)
    plt.close(fig)

    save_figure_data(output_path, result_df)
    return pivot


# ── RQ2: Worst-task identification and comparison ─────────────────────


def identify_worst_ood_families(
    raw_df: pd.DataFrame,
    *,
    amortised_models: Sequence[str] | None = None,
    metric_name: str = "ne-sid",
    top_k: int = 10,
) -> pd.DataFrame:
    """Identify the OOD families where amortised models degrade most.

    For each OOD family the degradation ratio is computed as
    ``mean(OOD metric) / mean(ID metric)`` where the ID baseline is
    the anchored ID subset for the family's shift axis.  Families are
    ranked by the *maximum* degradation ratio across the amortised models
    and the top-*k* worst are returned.

    Args:
        raw_df: Long-format raw task DataFrame.
        amortised_models: Display names of amortised models.
            Defaults to :data:`AMORTISED_MODELS` from ``utils``.
        metric_name: Metric to use for degradation ranking.
        top_k: Number of worst families to return.

    Returns:
        DataFrame with columns ``DatasetKey``, ``AxisCategory``,
        ``Model``, ``OOD_Mean``, ``ID_Mean``, ``Ratio``, sorted by
        worst ``MaxRatio`` descending.
    """
    from causal_meta.analysis.utils import AMORTISED_MODELS as _AMORTISED

    if amortised_models is None:
        amortised_models = sorted(_AMORTISED)

    subset = raw_df[raw_df["Metric"].eq(metric_name)].copy()
    fixed = subset[is_fixed_size_task_frame(subset)]

    # Build per-axis ID baselines for amortised models
    id_means: dict[tuple[str, str], float] = {}  # (model, shift_key) -> mean
    for shift_key in _DEGRADATION_SHIFT_AXES:
        id_data = _degradation_id_subset(fixed, shift_key)
        for model in amortised_models:
            model_id = id_data[id_data["Model"] == model]["Value"]
            if not model_id.empty:
                id_means[(model, shift_key)] = float(model_id.mean())

    # Compute per-family degradation
    ood_cats = set(_DEGRADATION_SHIFT_AXES.keys()) - {"stress"}
    ood_data = subset[subset["AxisCategory"].isin(ood_cats)]
    # For fixed-size axes use the fixed subset; transfer axes keep all sizes
    fixed_ood = fixed[fixed["AxisCategory"].isin(ood_cats - {"nodes", "samples"})]
    transfer_ood = ood_data[ood_data["AxisCategory"].isin({"nodes", "samples"})]
    # Stress families are compound-category but identified by dataset key
    stress_ood = subset[subset["DatasetKey"].isin(_EXTREME_FAMILIES)]
    ood_pool = pd.concat([fixed_ood, transfer_ood, stress_ood], ignore_index=True)

    rows: list[dict[str, object]] = []
    for (model, dk), grp in ood_pool.groupby(["Model", "DatasetKey"]):
        model_str = str(model)
        if model_str not in amortised_models:
            continue
        # Stress families get their own axis key
        dk_str = str(dk)
        if dk_str in _EXTREME_FAMILIES:
            axis_cat = "stress"
        else:
            axis_cat = str(grp["AxisCategory"].iloc[0])
        ood_mean = float(grp["Value"].mean())
        id_val = id_means.get((model_str, axis_cat))
        if id_val is None or id_val <= 0:
            continue
        ratio = ood_mean / id_val
        rows.append(
            {
                "DatasetKey": str(dk),
                "AxisCategory": axis_cat,
                "Model": model_str,
                "OOD_Mean": ood_mean,
                "ID_Mean": id_val,
                "Ratio": ratio,
            }
        )

    if not rows:
        log.warning("No OOD degradation data for worst-task identification.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Rank by maximum ratio across amortised models per family
    max_ratio = (
        result.groupby("DatasetKey")["Ratio"].max().rename("MaxRatio").reset_index()
    )
    result = result.merge(max_ratio, on="DatasetKey")
    result = result.sort_values("MaxRatio", ascending=False)

    # Select top-K unique families
    top_families = list(result["DatasetKey"].unique()[:top_k])
    return result[result["DatasetKey"].isin(top_families)].copy()


def generate_rq2_worst_task_comparison(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    worst_families_df: pd.DataFrame | None = None,
    metric_name: str = "ne-sid",
    top_k: int = 10,
) -> pd.DataFrame:
    """Generate a three-row figure comparing all models on the worst OOD families.

    Top row: thresholded valid-DAG percentage for the same families.
    Middle row: grouped ne-SID comparison across families.
    Bottom row: stacked error decomposition (FP / FN / Reversed).

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output PDF figure.
        worst_families_df: Pre-computed worst families (optional).
        metric_name: Metric to plot in top row.
        top_k: Number of worst families to show.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    if worst_families_df is None or worst_families_df.empty:
        worst_families_df = identify_worst_ood_families(
            raw_df,
            metric_name=metric_name,
            top_k=top_k,
        )
    if worst_families_df.empty:
        log.warning("No worst families identified; skipping RQ2 comparison.")
        return pd.DataFrame()

    models = _resolve_models()

    def _families_with_all_models(family_keys: Sequence[str]) -> list[str]:
        metric_rows = raw_df[
            raw_df["Metric"].eq(metric_name) & raw_df["DatasetKey"].isin(family_keys)
        ]
        present = metric_rows.groupby("DatasetKey")["Model"].apply(lambda s: set(s))
        required = set(models)
        return [fk for fk in family_keys if required.issubset(present.get(fk, set()))]

    target_families = _families_with_all_models(
        list(worst_families_df["DatasetKey"].unique())
    )[:top_k]
    stress_families = _families_with_all_models(_EXTREME_FAMILIES)
    plot_families = target_families + stress_families
    if not plot_families:
        log.warning("No complete worst or stress families for RQ2 comparison.")
        return pd.DataFrame()

    # ── Top row data: primary metric ──────────────────────────────────
    subset = raw_df[
        raw_df["Metric"].eq(metric_name) & raw_df["DatasetKey"].isin(plot_families)
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    n_models = len(models)

    agg = (
        subset.groupby(
            ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
        )["Value"]
        .agg(Mean="mean", SEM=metric_sem)
        .reset_index()
    )
    agg["DatasetLabel"] = agg.apply(
        lambda row: (
            _compound_shift_label(str(row["DatasetKey"]))
            if str(row["DatasetKey"]).startswith("ood_both_")
            or str(row["DatasetKey"]) in _EXTREME_FAMILIES
            else thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"]))
        ),
        axis=1,
    )

    # Sort by shift-type group then by worst degradation ratio within group.
    _GROUP_ORDER = {
        "graph": 0,
        "mechanism": 1,
        "noise": 2,
        "compound": 3,
        "stress": 4,
        "nodes": 5,
        "samples": 6,
    }
    agg["_axis_group"] = agg["DatasetKey"].apply(
        lambda dk: "stress" if str(dk) in _EXTREME_FAMILIES else None
    )
    mask_no_group = agg["_axis_group"].isna()
    agg.loc[mask_no_group, "_axis_group"] = agg.loc[
        mask_no_group, "AxisCategory"
    ].astype(str)

    agg["_group_sort"] = agg["_axis_group"].map(_GROUP_ORDER).fillna(99)
    family_order = {fk: i for i, fk in enumerate(plot_families)}
    agg["_family_sort"] = agg["DatasetKey"].map(family_order).fillna(999)
    agg = agg.sort_values(["_group_sort", "_family_sort"]).drop(
        columns=["_group_sort", "_family_sort"]
    )

    width = 0.6
    offset_step = width / max(n_models, 1)

    # ── Top row data: valid-DAG percentage ────────────────────────────
    dag_subset = raw_df[
        raw_df["Metric"].isin(["valid_dag_pct", "threshold_valid_dag_pct"])
        & raw_df["DatasetKey"].isin(plot_families)
    ].copy()
    dag_agg = pd.DataFrame()
    if not dag_subset.empty:
        dag_agg = (
            dag_subset.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory", "Metric"],
                dropna=False,
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        dag_agg["DatasetLabel"] = dag_agg.apply(
            lambda row: (
                _compound_shift_label(str(row["DatasetKey"]))
                if str(row["DatasetKey"]).startswith("ood_both_")
                or str(row["DatasetKey"]) in _EXTREME_FAMILIES
                else thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"]))
            ),
            axis=1,
        )

    # ── Bottom row data: error decomposition ──────────────────────────
    error_metric_names = {"fp_count", "fn_count", "reversed_count"}
    error_subset = raw_df[
        raw_df["Metric"].isin(error_metric_names)
        & raw_df["DatasetKey"].isin(plot_families)
    ].copy()

    error_agg = pd.DataFrame()
    if not error_subset.empty:
        error_agg = (
            error_subset.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory", "Metric"],
                dropna=False,
            )["Value"]
            .mean()
            .reset_index()
            .rename(columns={"Value": "Mean"})
        )
        error_agg["DatasetLabel"] = error_agg.apply(
            lambda row: (
                _compound_shift_label(str(row["DatasetKey"]))
                if str(row["DatasetKey"]).startswith("ood_both_")
                or str(row["DatasetKey"]) in _EXTREME_FAMILIES
                else thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"]))
            ),
            axis=1,
        )

    _AXIS_GROUP_LABELS: dict[str, str] = {
        "graph": "Graph Target Environments",
        "mechanism": "Mechanism Target Environments",
        "noise": "Noise Target Environments",
        "compound": "Graph-Mechanism Target Environments",
        "stress": "Stress-Test Target Environments",
        "nodes": "Node-Count Target Environments",
        "samples": "Sample-Size Target Environments",
    }

    def _panel_environment_title(family_keys: Sequence[str], fallback: str) -> str:
        groups = {
            "stress" if str(key) in _EXTREME_FAMILIES else axis_category(str(key))
            for key in family_keys
        }
        groups.discard("other")
        if len(groups) == 1:
            group = next(iter(groups))
            return _AXIS_GROUP_LABELS.get(group, fallback)
        return fallback

    panels = [
        (
            _panel_environment_title(
                target_families, "Hardest OOD Target Environments"
            ),
            target_families,
        ),
        (
            _panel_environment_title(
                stress_families, "Stress-Test Target Environments"
            ),
            stress_families,
        ),
    ]
    panels = [(title, keys) for title, keys in panels if keys]
    n_panels = len(panels)
    max_panel_families = max(len(keys) for _, keys in panels)

    # ── Create metric/DAG/error rows with separate hardest-OOD and stress panels ──
    fig, axes = plt.subplots(
        3,
        n_panels,
        figsize=(
            max(WIDE_TWO_ROW_FIGSIZE[0], 0.95 * max_panel_families * n_panels),
            WIDE_TWO_ROW_FIGSIZE[1] * 1.28,
        ),
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": [0.9, 1, 1]},
    )

    # Axis background colours and group labels helper.
    _AXIS_BG: dict[str, str] = {
        "graph": "#e8f0fe",
        "mechanism": "#fef3e0",
        "noise": "#f0f8e8",
        "compound": "#fde8e8",
        "stress": "#fde8fe",
        "nodes": "#f0e8fe",
        "samples": "#e8fef0",
    }

    def _apply_axis_bg(
        ax: plt.Axes, datasets: list[str], axis_lookup: dict[str, str]
    ) -> None:
        for i, ds in enumerate(datasets):
            bg = _AXIS_BG.get(axis_lookup.get(ds, ""), None)
            if bg:
                ax.axvspan(i - 0.5, i + 0.5, color=bg, alpha=0.4, zorder=0)
        ordered_groups = [axis_lookup.get(ds, "") for ds in datasets]
        prev_group = None
        for i, grp in enumerate(ordered_groups):
            if grp != prev_group:
                if prev_group is not None:
                    ax.axvline(
                        i - 0.5, color="#888888", linestyle="-", linewidth=1.0, zorder=2
                    )
                prev_group = grp

    first_dag_handles: tuple[list, list] | None = None
    first_error_handles: tuple[list, list] | None = None
    bar_width = 0.6 / max(n_models, 1)
    dag_model_order = ["AviCi", "DiBS"]
    dag_metric_style: dict[str, tuple[str, float]] = {
        "threshold_valid_dag_pct": ("Thresholded", 0.55),
        "valid_dag_pct": ("Sampled", 0.85),
    }

    for panel_idx, (panel_title, panel_keys) in enumerate(panels):
        panel_agg = agg[agg["DatasetKey"].isin(panel_keys)].copy()
        panel_dag = dag_agg[dag_agg["DatasetKey"].isin(panel_keys)].copy()
        panel_error = error_agg[error_agg["DatasetKey"].isin(panel_keys)].copy()
        panel_order = {fk: i for i, fk in enumerate(panel_keys)}
        panel_agg["_panel_sort"] = panel_agg["DatasetKey"].map(panel_order)
        panel_agg = panel_agg.sort_values(["_panel_sort", "Model"])
        datasets = list(dict.fromkeys(panel_agg["DatasetLabel"]))
        n_datasets = len(datasets)
        x_base = np.arange(n_datasets)
        axis_lookup = (
            panel_agg[["DatasetLabel", "_axis_group"]]
            .drop_duplicates()
            .set_index("DatasetLabel")["_axis_group"]
            .to_dict()
        )

        ax_dag = axes[0, panel_idx]
        ax_top = axes[1, panel_idx]
        ax_bottom = axes[2, panel_idx]
        # No background shading for this comparison view.

        for model_idx, model in enumerate(models):
            model_agg = panel_agg[panel_agg["Model"] == model]
            xs: list[float] = []
            means: list[float] = []
            sems: list[float] = []
            for i, ds in enumerate(datasets):
                row = model_agg[model_agg["DatasetLabel"] == ds]
                if row.empty:
                    continue
                offset = (model_idx - n_models / 2 + 0.5) * offset_step
                xs.append(float(x_base[i]) + offset)
                means.append(float(row.iloc[0]["Mean"]))
                sems.append(float(row.iloc[0]["SEM"]))
            if xs:
                ax_top.errorbar(
                    xs,
                    means,
                    yerr=sems,
                    fmt=MODEL_MARKERS.get(model, "o"),
                    label=model,
                    color=_model_color(model),
                    capsize=2,
                    markersize=6,
                    alpha=0.9,
                )

        ax_top.set_ylabel(
            _sid_ylabel(metric_name) if panel_idx == 0 else "", fontsize=_LABEL_FONTSIZE
        )
        ax_top.grid(True, axis="y", linestyle="--", alpha=0.4)

        if not panel_dag.empty:
            dag_models = [
                model
                for model in dag_model_order
                if model in panel_dag["Model"].unique()
            ]
            n_dag_models = len(dag_models)
            dag_bar_width = 0.15
            for model_idx, model in enumerate(dag_models):
                group_offset = (
                    (model_idx - n_dag_models / 2 + 0.5) * dag_bar_width * 2.2
                )
                for metric_idx, (metric_key, (label_sfx, alpha)) in enumerate(
                    dag_metric_style.items()
                ):
                    metric_dag = panel_dag[panel_dag["Metric"] == metric_key]
                    if metric_dag.empty:
                        continue
                    model_dag = metric_dag[metric_dag["Model"] == model]
                    xs: list[float] = []
                    means: list[float] = []
                    sems: list[float] = []
                    for i, ds in enumerate(datasets):
                        row = model_dag[model_dag["DatasetLabel"] == ds]
                        if row.empty:
                            continue
                        inner_offset = (metric_idx - 0.5) * dag_bar_width
                        xs.append(float(x_base[i]) + group_offset + inner_offset)
                        means.append(float(row.iloc[0]["Mean"]))
                        sems.append(float(row.iloc[0]["SEM"]))
                    if xs:
                        label = f"{model} {label_sfx}" if panel_idx == 0 else None
                        ax_dag.bar(
                            xs,
                            means,
                            yerr=sems,
                            width=dag_bar_width,
                            label=label,
                            color=_model_color(model),
                            capsize=2,
                            alpha=alpha,
                            edgecolor="white",
                            linewidth=0.3,
                        )
        ax_dag.set_title(panel_title, fontsize=_TITLE_FONTSIZE)
        ax_dag.set_ylim(-5, 105)
        ax_dag.set_ylabel(
            r"Valid DAG (\%) $\uparrow$" if panel_idx == 0 else "",
            fontsize=_LABEL_FONTSIZE,
        )
        ax_dag.grid(True, axis="y", linestyle="--", alpha=0.4)
        if first_dag_handles is None:
            h, l = ax_dag.get_legend_handles_labels()
            first_dag_handles = (h, l)

        if not panel_error.empty:
            for model_idx, model in enumerate(models):
                model_err = panel_error[panel_error["Model"] == model]
                offset = (model_idx - n_models / 2 + 0.5) * bar_width
                bottoms = np.zeros(n_datasets)
                for err_key, err_label, err_color in ERROR_SPECS:
                    heights = np.zeros(n_datasets)
                    for ds_idx, ds in enumerate(datasets):
                        row = model_err[
                            (model_err["DatasetLabel"] == ds)
                            & (model_err["Metric"] == err_key)
                        ]
                        if not row.empty:
                            heights[ds_idx] = float(row.iloc[0]["Mean"])
                    ax_bottom.bar(
                        x_base + offset,
                        heights,
                        bar_width,
                        bottom=bottoms,
                        color=err_color,
                        alpha=0.78,
                        label=err_label if model_idx == 0 else None,
                        edgecolor="white",
                        linewidth=0.3,
                    )
                    bottoms += heights

        ax_bottom.set_ylabel(
            _error_count_ylabel(log_scale=True) if panel_idx == 0 else "",
            fontsize=_LABEL_FONTSIZE,
        )
        ax_bottom.set_yscale("symlog", linthresh=10)
        ax_bottom.set_xticks(x_base)
        ax_bottom.set_xticklabels(
            datasets, rotation=35, ha="right", fontsize=_TICK_FONTSIZE
        )
        ax_bottom.grid(True, axis="y", linestyle="--", alpha=0.4)
        if first_error_handles is None:
            h, l = ax_bottom.get_legend_handles_labels()
            first_error_handles = (h, l)

    if first_dag_handles is not None and first_dag_handles[0]:
        model_label_count = len(first_dag_handles[1])
        axes[0, 0].legend(
            first_dag_handles[0],
            first_dag_handles[1],
            title="Model",
            loc="best",
            ncol=_model_legend_ncol(model_label_count),
            fontsize=_model_legend_fontsize(model_label_count),
            frameon=True,
            framealpha=0.8,
        )

    if first_error_handles is not None and first_error_handles[0]:
        axes[2, -1].legend(
            first_error_handles[0],
            first_error_handles[1],
            loc="upper right",
            fontsize=_LEGEND_FONTSIZE,
            frameon=True,
            framealpha=0.9,
        )

    fig.suptitle(
        "All Models on Hardest OOD Families and Stress Tests",
        fontsize=_TITLE_FONTSIZE,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    result = agg.drop(columns=["_axis_group"], errors="ignore")
    save_figure_data(output_path, result)
    return result


def generate_shift_figure(
    raw_df: pd.DataFrame,
    *,
    shift_axis: str,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
    with_avici_dag_row: bool = False,
) -> pd.DataFrame:
    axis_cat, axis_title = _SHIFT_AXIS_SPECS[shift_axis]
    metric_name = "ne-sid" if shift_axis == "compound" else "e-sid"
    subset = raw_df[
        raw_df["Metric"].eq(metric_name) & raw_df["AxisCategory"].isin(["id", axis_cat])
    ].copy()
    subset = subset[is_fixed_size_task_frame(subset)]

    # Filter to target models when requested
    if model_filter is not None:
        subset = subset[subset["Model"].isin(model_filter)]

    # Prepare AviCi DAG validity data when requested.
    # Include both sampled (valid_dag_pct) and thresholded
    # (threshold_valid_dag_pct) DAG rates.
    avici_dag_df: pd.DataFrame | None = None
    error_df: pd.DataFrame | None = None
    if with_avici_dag_row:
        avici_dag_df = raw_df[
            raw_df["Metric"].isin(["valid_dag_pct", "threshold_valid_dag_pct"])
            & raw_df["AxisCategory"].isin(["id", axis_cat])
            & raw_df["Model"].eq("AviCi")
        ].copy()
        avici_dag_df = avici_dag_df[is_fixed_size_task_frame(avici_dag_df)]
        if avici_dag_df.empty:
            avici_dag_df = None

        # Error decomposition (FP / FN / Reversed) for all models in scope.
        _error_metrics = [k for k, _, _ in ERROR_SPECS]
        error_df = raw_df[
            raw_df["Metric"].isin(_error_metrics)
            & raw_df["AxisCategory"].isin(["id", axis_cat])
        ].copy()
        error_df = error_df[is_fixed_size_task_frame(error_df)]
        if model_filter is not None:
            error_df = error_df[error_df["Model"].isin(model_filter)]
        if error_df.empty:
            error_df = None

    # ── Graph shift: multi-panel by mechanism family ──────────────────
    if shift_axis == "graph":
        return _generate_graph_shift_panels(
            subset,
            metric_name,
            axis_title,
            output_path,
            model_filter=model_filter,
            avici_dag_df=avici_dag_df,
            error_df=error_df,
        )

    # ── Mechanism shift: multi-panel by graph anchor ──────────────────
    if shift_axis == "mechanism":
        return _generate_mech_shift_panels(
            subset,
            metric_name,
            axis_title,
            output_path,
            model_filter=model_filter,
            avici_dag_df=avici_dag_df,
            error_df=error_df,
        )

    # ── Noise shift: multi-panel by (mechanism, graph) anchor ─────────
    if shift_axis == "noise":
        return _generate_noise_shift_panels(
            subset,
            metric_name,
            axis_title,
            output_path,
            model_filter=model_filter,
            avici_dag_df=avici_dag_df,
            error_df=error_df,
        )

    # ── Compound shift: multi-panel by fixed OOD graph ────────────────
    if shift_axis == "compound":
        return _generate_compound_shift_panels(
            subset,
            metric_name,
            axis_title,
            output_path,
            model_filter=model_filter,
            avici_dag_df=avici_dag_df,
            error_df=error_df,
        )

    agg = (
        subset.groupby(
            ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
        )["Value"]
        .agg(Mean="mean", SEM=metric_sem)
        .reset_index()
    )
    agg["DatasetLabel"] = agg.apply(
        lambda row: thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"])),
        axis=1,
    )
    agg["_sort"] = agg["AxisCategory"].map({"id": 0}).fillna(1)
    agg = agg.sort_values(["_sort", "DatasetLabel"]).drop(columns=["_sort"])

    datasets = list(agg["DatasetLabel"].unique())
    models = _resolve_models(model_filter)
    axis_lookup = (
        agg[["DatasetLabel", "AxisCategory"]]
        .drop_duplicates()
        .set_index("DatasetLabel")["AxisCategory"]
        .to_dict()
    )
    n_datasets = len(datasets)
    n_models = len(models)
    width = 0.6
    offset_step = width / max(n_models, 1)
    x_base = np.arange(n_datasets)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * n_datasets), 5))

    for model_idx, model in enumerate(models):
        model_agg = agg[agg["Model"] == model]
        xs: list[float] = []
        means: list[float] = []
        sems: list[float] = []
        for i, ds in enumerate(datasets):
            row = model_agg[model_agg["DatasetLabel"] == ds]
            if row.empty:
                continue
            offset = (model_idx - n_models / 2 + 0.5) * offset_step
            xs.append(float(x_base[i]) + offset)
            means.append(float(row.iloc[0]["Mean"]))
            sems.append(float(row.iloc[0]["SEM"]))
        if xs:
            ax.errorbar(
                xs,
                means,
                yerr=sems,
                fmt=MODEL_MARKERS.get(model, "o"),
                label=model,
                color=_model_color(model),
                capsize=3,
                markersize=7,
                alpha=0.9,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
    ylabel = _sid_ylabel(metric_name)
    ax.set_ylabel(ylabel, fontsize=_LABEL_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
    if 0 < id_count < len(datasets):
        ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
        ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    # Shared legend on top.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            title="Model",
            loc="best",
            ncol=_model_legend_ncol(len(labels)),
            fontsize=_model_legend_fontsize(len(labels)),
            frameon=True,
            framealpha=0.8,
        )

    fig.suptitle(axis_title, fontsize=_TITLE_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, agg)
    return agg


def generate_graph_density_ablation_comparison(
    *,
    input_root: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Compare original AviCi graph shifts against the graph-ID ablation.

    The figure shows absolute ``e-sid`` values for the original run and the
    graph-ID ablation on the fixed-size graph-shift families. Each panel fixes a
    mechanism generator and compares ER anchors against the OOD graph families.
    """

    original_run_dir = input_root / "avici"
    ablation_run_dir = input_root / "avici_ablation"
    missing = [
        str(path)
        for path in (original_run_dir, ablation_run_dir)
        if not (path / "metrics.json").exists()
    ]
    if missing:
        raise EmptyAnalysisDataError(
            "Missing AviCi graph ablation runs: " + ", ".join(missing)
        )

    df = load_runs_dataframe(
        [original_run_dir, ablation_run_dir],
        translate_names=False,
    )
    if df.empty:
        raise EmptyAnalysisDataError(
            "No summary metrics found for AviCi ablation plot."
        )

    dataset_pattern = re.compile(
        r"^(id_(linear|neuralnet|gpcde)_(er20|er40|er60|sf1|sf2|sf3)_d20_n500|"
        r"ood_graph_(sbm|ws|grg)_(linear|neuralnet|gpcde)_d20_n500)$"
    )
    subset = df[
        df["ModelKey"].str.lower().eq("avici")
        & df["Metric"].eq("e-sid")
        & df["NNodes"].eq(20)
        & df["SamplesPerTask"].eq(500)
        & df["DatasetKey"].str.match(dataset_pattern)
    ].copy()
    if subset.empty:
        raise EmptyAnalysisDataError("No fixed-size AviCi graph-shift metrics found.")

    subset["MechanismKey"] = subset["DatasetKey"].map(id_mechanism_of)
    subset["GraphCode"] = subset["DatasetKey"].map(graph_code_of)
    subset["RunGroup"] = np.where(
        subset["RunDir"].eq(str(original_run_dir)),
        "Original graph shifts",
        "OOD graphs in training",
    )
    subset["IsAnchor"] = subset["DatasetKey"].str.startswith("id_")
    subset = subset.dropna(subset=["MechanismKey", "GraphCode"]).copy()
    if subset.empty:
        raise EmptyAnalysisDataError(
            "No matched graph-family rows available for AviCi ablation plot."
        )

    tick_codes = [
        graph_code
        for graph_code in _GRAPH_SHIFT_CODE_ORDER
        if not subset.loc[subset["GraphCode"].eq(graph_code)].empty
    ]
    tick_positions = list(range(len(tick_codes)))
    tick_position_map = {graph_code: idx for idx, graph_code in enumerate(tick_codes)}
    tick_labels = [_GRAPH_SHIFT_LABELS[graph_code] for graph_code in tick_codes]

    plot_rows: list[pd.DataFrame] = []

    fig, axes_arr = plt.subplots(
        1, 3, figsize=grid_figsize(3), sharey=True, squeeze=False
    )
    axes = axes_arr[0]
    mech_order = ["linear", "neuralnet", "gpcde"]
    series_styles = {
        "Original graph shifts": {
            "color": "#4d4d4d",
            "alpha": 0.9,
        },
        "Graph-ID ablation": {
            "color": "#5b4ea3",
            "alpha": 0.9,
        },
    }

    for ax, mech_key in zip(axes, mech_order):
        panel = subset[subset["MechanismKey"].eq(mech_key)].copy()
        if panel.empty:
            ax.set_visible(False)
            continue

        panel_plot = panel.copy()
        panel_plot["Series"] = np.where(
            panel_plot["RunDir"].eq(str(original_run_dir)),
            "Original graph shifts",
            "Graph-ID ablation",
        )
        panel_plot["PointLabel"] = panel_plot["GraphCode"].map(_GRAPH_SHIFT_LABELS)
        panel_plot["OrderX"] = panel_plot["GraphCode"].map(tick_position_map)
        plot_rows.append(panel_plot)

        for x_idx, graph_code in enumerate(tick_codes):
            if graph_code in {"er20", "er40", "er60", "sf1", "sf2", "sf3"}:
                ax.axvspan(
                    x_idx - 0.5, x_idx + 0.5, color="#e8e8e8", alpha=0.55, zorder=0
                )
        if len(tick_codes) > 6:
            ax.axvline(5.5, color="#888888", linestyle="-", linewidth=1.0, zorder=2)

        bar_width = 0.32
        series_order = ["Original graph shifts", "Graph-ID ablation"]
        for series_idx, series_name in enumerate(series_order):
            series = panel_plot[panel_plot["Series"].eq(series_name)].sort_values(
                "OrderX"
            )
            if series.empty:
                continue
            style = series_styles[series_name]
            xs = series["OrderX"].to_numpy() + (series_idx - 0.5) * bar_width
            ax.bar(
                xs,
                series["Mean"],
                yerr=series["SEM"],
                width=bar_width,
                label=series_name,
                color=style["color"],
                capsize=3,
                alpha=style["alpha"],
                edgecolor="white",
                linewidth=0.4,
            )

        ax.set_title(
            ID_MECHANISM_LABELS.get(mech_key, mech_key), fontsize=_PANEL_TITLE_FONTSIZE
        )
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.grid(True, axis="x", linestyle=":", alpha=0.2)
        ax.set_xlabel("Graph family", fontsize=_LABEL_FONTSIZE)

    axes[0].set_ylabel(metric_label("e-sid", down=True), fontsize=_LABEL_FONTSIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(
            handles,
            labels,
            loc="upper left",
            fontsize=_LEGEND_FONTSIZE,
            frameon=True,
            framealpha=0.8,
        )

    fig.suptitle(
        "AviCi Graph-ID Ablation: Absolute SID",
        fontsize=_TITLE_FONTSIZE,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    result = pd.concat(plot_rows, ignore_index=True)
    save_figure_data(output_path, result)
    log.info("Saved graph-density ablation comparison to %s", output_path)
    return result


def generate_graph_ablation_delta_figure(
    *,
    input_root: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Plot AviCi graph-ID ablation deltas in the standard 3x3 layout.

    The figure compares ``experiments/thesis_runs/avici`` against
    ``experiments/thesis_runs/avici_ablation`` on the fixed-size graph-shift
    families and plots ``ablation - original``. Negative SID/error deltas are
    therefore improvements, while positive DAG-validity deltas are improvements.
    """

    from causal_meta.analysis.common.thesis import prepare_raw_dataframe

    original_run_dir = input_root / "avici"
    ablation_run_dir = input_root / "avici_ablation"
    missing = [
        str(path)
        for path in (original_run_dir, ablation_run_dir)
        if not (path / "metrics.json").exists()
    ]
    if missing:
        raise EmptyAnalysisDataError(
            "Missing AviCi graph ablation runs: " + ", ".join(missing)
        )

    raw_df = prepare_raw_dataframe([original_run_dir, ablation_run_dir])
    if raw_df.empty:
        raise EmptyAnalysisDataError("No raw AviCi ablation metrics found.")

    graph_id_codes = _GRAPH_SHIFT_CODE_ORDER[:6]
    graph_ood_codes = _GRAPH_SHIFT_CODE_ORDER[6:]
    mech_order = ["linear", "neuralnet", "gpcde"]
    panel_keys = {
        mech_key: [
            *(f"id_{mech_key}_{graph_code}_d20_n500" for graph_code in graph_id_codes),
            *(
                f"ood_graph_{graph_code}_{mech_key}_d20_n500"
                for graph_code in graph_ood_codes
            ),
        ]
        for mech_key in mech_order
    }
    dataset_keys = [key for keys in panel_keys.values() for key in keys]
    metric_keys = [
        "threshold_valid_dag_pct",
        "valid_dag_pct",
        "e-sid",
        "fp_count",
        "fn_count",
        "reversed_count",
    ]
    run_map = {
        "avici": "original",
        "avici_ablation4": "ablation",
    }

    subset = raw_df[
        raw_df["ModelKey"].str.lower().eq("avici")
        & raw_df["DatasetKey"].isin(dataset_keys)
        & raw_df["Metric"].isin(metric_keys)
    ].copy()
    subset["Run"] = subset["RunID"].map(run_map)
    subset = subset.dropna(subset=["Run"])
    if subset.empty:
        raise EmptyAnalysisDataError("No fixed-size AviCi ablation deltas found.")

    paired = subset.pivot_table(
        index=["DatasetKey", "Metric", "TaskIdx"],
        columns="Run",
        values="Value",
        aggfunc="first",
    ).reset_index()
    for needed in ("original", "ablation"):
        if needed not in paired.columns:
            raise EmptyAnalysisDataError(
                f"Missing '{needed}' run in AviCi ablation delta figure."
            )
    paired = paired.dropna(subset=["original", "ablation"])
    paired["Delta"] = paired["ablation"] - paired["original"]

    pivot = (
        paired.groupby(["DatasetKey", "Metric"], dropna=False)["Delta"]
        .agg(Delta="mean", DeltaSEM=metric_sem)
        .reset_index()
    )
    pivot["MechanismKey"] = pivot["DatasetKey"].map(id_mechanism_of)
    pivot["DatasetLabel"] = pivot["DatasetKey"].map(_graph_shift_label)
    pivot["IsAnchor"] = pivot["DatasetKey"].str.startswith("id_")

    fig, axes = plt.subplots(
        3,
        3,
        figsize=_three_row_figsize(3, match_width_cols=3),
        sharex="col",
        sharey="row",
        squeeze=False,
        gridspec_kw={"height_ratios": _THREE_ROW_HEIGHT_RATIOS, "wspace": 0.16},
    )

    dag_metric_style = {
        "threshold_valid_dag_pct": ("Thresholded", 0.5),
        "valid_dag_pct": ("Sampled", 0.85),
    }
    dag_handles: tuple[list, list] | None = None
    err_handles: tuple[list, list] | None = None
    plot_rows: list[pd.DataFrame] = []

    dag_limit = float(
        max(
            1.0,
            np.nanmax(
                np.abs(
                    pivot[
                        pivot["Metric"].isin(
                            ["threshold_valid_dag_pct", "valid_dag_pct"]
                        )
                    ]["Delta"].to_numpy()
                )
                + pivot[
                    pivot["Metric"].isin(["threshold_valid_dag_pct", "valid_dag_pct"])
                ]["DeltaSEM"].to_numpy()
            ),
        )
    )
    sid_limit = float(
        max(
            0.01,
            np.nanmax(
                np.abs(pivot[pivot["Metric"].eq("e-sid")]["Delta"].to_numpy())
                + pivot[pivot["Metric"].eq("e-sid")]["DeltaSEM"].to_numpy()
            ),
        )
    )
    error_limit = float(
        max(
            1.0,
            np.nanmax(
                np.abs(
                    pivot[
                        pivot["Metric"].isin(["fp_count", "fn_count", "reversed_count"])
                    ]["Delta"].to_numpy()
                )
                + pivot[
                    pivot["Metric"].isin(["fp_count", "fn_count", "reversed_count"])
                ]["DeltaSEM"].to_numpy()
            ),
        )
    )

    for col_idx, mech_key in enumerate(mech_order):
        panel = pivot[pivot["MechanismKey"].eq(mech_key)].copy()
        panel_order = {key: i for i, key in enumerate(panel_keys[mech_key])}
        panel = panel[panel["DatasetKey"].isin(panel_order)].copy()
        panel["OrderX"] = panel["DatasetKey"].map(panel_order)
        panel = panel.sort_values(["Metric", "OrderX"])
        plot_rows.append(panel)

        x_base = np.arange(len(panel_keys[mech_key]))
        labels = [_graph_shift_label(key) for key in panel_keys[mech_key]]

        ax_dag = axes[0, col_idx]
        ax_sid = axes[1, col_idx]
        ax_err = axes[2, col_idx]

        for i, key in enumerate(panel_keys[mech_key]):
            if key.startswith("id_"):
                for ax in (ax_dag, ax_sid, ax_err):
                    ax.axvspan(i - 0.5, i + 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
        for ax in (ax_dag, ax_sid, ax_err):
            ax.axvline(5.5, color="#999999", linestyle=":", linewidth=1.0, zorder=2)

        bar_width = 0.28
        for metric_idx, (metric_key, (metric_label_sfx, alpha)) in enumerate(
            dag_metric_style.items()
        ):
            metric_panel = panel[panel["Metric"].eq(metric_key)].sort_values("OrderX")
            xs = x_base + (metric_idx - 0.5) * bar_width
            vals = metric_panel["Delta"].to_numpy()
            sems = metric_panel["DeltaSEM"].to_numpy()
            bars = ax_dag.bar(
                xs,
                vals,
                yerr=sems,
                width=bar_width,
                color=_model_color("AviCi"),
                alpha=alpha,
                edgecolor="white",
                linewidth=0.3,
                capsize=3,
                label=metric_label_sfx if col_idx == 0 else None,
            )
            if dag_handles is None and col_idx == 0:
                dag_handles = ([bars[0]], [metric_label_sfx])
            elif dag_handles is not None and col_idx == 0:
                dag_handles[0].append(bars[0])
                dag_handles[1].append(metric_label_sfx)

        sid_panel = panel[panel["Metric"].eq("e-sid")].sort_values("OrderX")
        ax_sid.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", zorder=1)
        ax_sid.errorbar(
            x_base,
            sid_panel["Delta"].to_numpy(),
            yerr=sid_panel["DeltaSEM"].to_numpy(),
            fmt=MODEL_MARKERS.get("AviCi", "o"),
            color=_model_color("AviCi"),
            alpha=0.9,
            capsize=3,
            markersize=7,
            zorder=3,
        )

        local_err_handles: list = []
        local_err_labels: list[str] = []
        for err_key, err_label, err_color in ERROR_SPECS:
            err_panel = panel[panel["Metric"].eq(err_key)].sort_values("OrderX")
            vals = err_panel["Delta"].to_numpy()
            xs = (
                x_base
                + ({"fp_count": -1, "fn_count": 0, "reversed_count": 1}[err_key]) * 0.26
            )
            h_pos = ax_err.bar(
                xs,
                vals,
                0.26,
                color=err_color,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.3,
                label=err_label if col_idx == 0 else None,
            )
            if col_idx == 0:
                local_err_handles.append(h_pos[0])
                local_err_labels.append(err_label)
        if err_handles is None and local_err_handles:
            err_handles = (local_err_handles, local_err_labels)

        ax_dag.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", zorder=1)
        ax_err.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", zorder=1)

        ax_dag.set_title(
            ID_MECHANISM_LABELS.get(mech_key, mech_key), fontsize=_PANEL_TITLE_FONTSIZE
        )
        ax_dag.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax_sid.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax_err.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax_dag.set_ylim(-dag_limit, dag_limit)
        ax_sid.set_ylim(-sid_limit, sid_limit)
        ax_err.set_ylim(-error_limit, error_limit)

        ax_err.set_xticks(x_base)
        ax_err.set_xticklabels(
            labels,
            rotation=30,
            ha="right",
            fontsize=_SHIFT_XTICK_FONTSIZE,
        )

        if col_idx == 0:
            ax_dag.set_ylabel(r"$\Delta$ Valid DAG (\%)", fontsize=_LABEL_FONTSIZE)
            ax_sid.set_ylabel(
                rf"$\Delta$ {metric_label('e-sid', down=True)}",
                fontsize=_LABEL_FONTSIZE,
            )
            ax_err.set_ylabel(r"$\Delta$ Error count", fontsize=_LABEL_FONTSIZE)

    if dag_handles is not None and dag_handles[0]:
        axes[0, 0].legend(
            dag_handles[0],
            dag_handles[1],
            loc="lower left",
            fontsize=_LEGEND_FONTSIZE,
            frameon=True,
            framealpha=0.8,
        )

    if err_handles is not None and err_handles[0]:
        axes[2, -1].legend(
            err_handles[0],
            err_handles[1],
            loc="upper right",
            fontsize=_LEGEND_FONTSIZE,
            frameon=True,
            framealpha=0.8,
        )

    fig.suptitle("AviCi Graph-ID Ablation Deltas", fontsize=_TITLE_FONTSIZE)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.12, wspace=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    result = pd.concat(plot_rows, ignore_index=True)
    save_figure_data(output_path, result)
    log.info("Saved graph ablation delta figure to %s", output_path)
    return result


# ── Compound isolation figure (RQ1) ───────────────────────────────────

_COMPOUND_ISOLATION_TRIPLETS: list[tuple[str, str, str, str]] = [
    # (id_mech, id_graph, ood_graph, ood_mech)
    ("neuralnet", "er20", "sbm", "pnl_tanh"),
    ("neuralnet", "er60", "grg", "periodic"),
    ("neuralnet", "sf2", "ws", "logistic_map"),
]
"""Sparsity-matched compound-isolation triplets.  Each row is
(id_mechanism, id_graph, ood_graph, ood_mechanism).  The ID mechanism is
always *neuralnet* (MLP); ID/OOD graph pairs share comparable edge density:
sparse (ER-20 ≈ SBM), medium (ER-60 ≈ GRG), dense (SF-2 ≈ WS)."""

_ISOLATION_STAGE_LABELS: list[str] = [
    "ID",
    "Mechanism",
    "Graph",
    "Graph-Mechanism",
]
"""X-axis labels for the four compound-isolation stages."""

_ISOLATION_STAGE_BG: list[str] = [
    "#f2f2f2",  # ID — grey
    "#e6f0ff",  # Mechanism target — blue
    "#e6f0ff",  # Graph target — blue
    "#fff3e6",  # Graph-mechanism target — orange
]
"""Per-stage background colour for visual grouping."""

_ISOLATION_SPARSITY_TIER: dict[str, str] = {
    "er20": "Sparse",
    "er60": "Medium",
    "sf2": "Dense",
}


def _isolation_dataset_keys(
    id_mech: str,
    id_graph: str,
    ood_graph: str,
    ood_mech: str,
) -> list[str]:
    """Build the four DatasetKey values for one compound-isolation triplet."""
    return [
        f"id_{id_mech}_{id_graph}_d20_n500",
        f"ood_mech_{ood_mech}_{id_graph}_d20_n500",
        f"ood_graph_{ood_graph}_{id_mech}_d20_n500",
        f"ood_both_{ood_graph}_{ood_mech}_d20_n500",
    ]


def _isolation_panel_title(
    id_mech: str,
    id_graph: str,
    ood_graph: str,
    ood_mech: str,
) -> str:
    """Build a compact panel title aligned with thesis notation."""
    start = _graph_mech_combo_label(graph_code=id_graph, mech_key=id_mech)
    end = _graph_mech_combo_label(graph_code=ood_graph, mech_key=ood_mech)
    return f"{start} -> {end}"


def _isolation_stage_combo_labels(
    id_mech: str,
    id_graph: str,
    ood_graph: str,
    ood_mech: str,
) -> list[str]:
    """Return the four stage labels using the combination convention."""
    return [
        _graph_mech_combo_label(graph_code=id_graph, mech_key=id_mech),
        _graph_mech_combo_label(graph_code=id_graph, mech_key=ood_mech),
        _graph_mech_combo_label(graph_code=ood_graph, mech_key=id_mech),
        _graph_mech_combo_label(graph_code=ood_graph, mech_key=ood_mech),
    ]


def generate_compound_isolation_figure(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
    with_avici_dag_row: bool = False,
) -> pd.DataFrame:
    """Generate a 3-panel compound-isolation figure.

    Each panel traces a four-step progression for a single
    (OOD-graph, OOD-mechanism) pair, anchored to a specific ID family:

        ID  →  Mechanism Target (only)  →  Graph Target (only)  →  Graph-Mechanism Target

    The three panels cover sparse, medium and dense sparsity tiers using
    sparsity-matched ID/OOD graph topologies, isolating compound
    interaction from density confounds.

    When *with_avici_dag_row* is ``True`` the figure gains a top row for
    AviCi DAG-validity rates and a bottom row for FP/FN/Reversed error
    decomposition (3×3 grid); otherwise a single-row 1×3 metric figure
    is produced.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output PDF figure.
        model_filter: If given, only include these model display names.
        with_avici_dag_row: If True, add DAG-validity top row and error
            decomposition bottom row.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    metric_name = "ne-sid"
    models = _resolve_models(model_filter)
    n_models = len(models)
    n_panels = len(_COMPOUND_ISOLATION_TRIPLETS)
    n_stages = len(_ISOLATION_STAGE_LABELS)

    # ── Build per-panel key lists and titles ───────────────────────────
    panel_keys: list[list[str]] = []
    panel_titles: list[str] = []
    panel_stage_labels_all: list[list[str]] = []
    for triplet in _COMPOUND_ISOLATION_TRIPLETS:
        panel_keys.append(_isolation_dataset_keys(*triplet))
        panel_titles.append(_isolation_panel_title(*triplet))
        panel_stage_labels_all.append(_isolation_stage_combo_labels(*triplet))

    all_keys = {k for keys in panel_keys for k in keys}

    # ── Filter metric data ────────────────────────────────────────────
    metric_df = raw_df[
        raw_df["Metric"].eq(metric_name) & raw_df["DatasetKey"].isin(all_keys)
    ].copy()
    if model_filter is not None:
        metric_df = metric_df[metric_df["Model"].isin(model_filter)]

    if metric_df.empty:
        log.warning("No data for compound isolation figure.")
        return pd.DataFrame()

    present_keys = set(metric_df["DatasetKey"].unique())
    missing = all_keys - present_keys
    if missing:
        log.warning(
            "Compound isolation: %d/%d dataset keys missing: %s",
            len(missing),
            len(all_keys),
            sorted(missing),
        )

    # ── Prepare auxiliary DAG + error data ─────────────────────────────
    avici_dag_df: pd.DataFrame | None = None
    error_df: pd.DataFrame | None = None
    if with_avici_dag_row:
        _dag = raw_df[
            raw_df["Metric"].isin(["valid_dag_pct", "threshold_valid_dag_pct"])
            & raw_df["Model"].eq("AviCi")
            & raw_df["DatasetKey"].isin(all_keys)
        ].copy()
        avici_dag_df = _dag if not _dag.empty else None

        _err_metrics = [k for k, _, _ in ERROR_SPECS]
        _err = raw_df[
            raw_df["Metric"].isin(_err_metrics) & raw_df["DatasetKey"].isin(all_keys)
        ].copy()
        if model_filter is not None:
            _err = _err[_err["Model"].isin(model_filter)]
        error_df = _err if not _err.empty else None

    has_dag = avici_dag_df is not None
    has_err = error_df is not None

    # ── Build row structure ───────────────────────────────────────────
    height_ratios: list[float] = []
    row_names: list[str] = []
    if has_dag:
        height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[0])
        row_names.append("dag")
    height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[1])
    row_names.append("metric")
    if has_err:
        height_ratios.append(_THREE_ROW_HEIGHT_RATIOS[2])
        row_names.append("error")
    n_rows = len(row_names)

    fig, axes = plt.subplots(
        n_rows,
        n_panels,
        figsize=_three_row_figsize(n_panels),
        sharex="col",
        sharey=False,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    metric_row = row_names.index("metric")
    dag_row = row_names.index("dag") if has_dag else None
    err_row = row_names.index("error") if has_err else None

    x_base = np.arange(n_stages)
    all_agg: list[pd.DataFrame] = []

    for panel_idx, (keys, title, panel_stage_labels) in enumerate(
        zip(panel_keys, panel_titles, panel_stage_labels_all)
    ):
        key_to_stage: dict[str, str] = dict(zip(keys, panel_stage_labels))
        key_to_idx: dict[str, int] = {k: i for i, k in enumerate(keys)}

        # Closure must capture the current dict via default arg.
        def _label_fn(dk: str, _m: dict[str, str] = key_to_stage) -> str:
            return _m.get(dk, dk)

        # ── Metric row ────────────────────────────────────────────────
        ax = axes[metric_row, panel_idx]
        panel_metric = metric_df[metric_df["DatasetKey"].isin(keys)]

        if panel_metric.empty:
            for r in range(n_rows):
                axes[r, panel_idx].set_visible(False)
            continue

        agg = (
            panel_metric.groupby(["Model", "DatasetKey", "Dataset"], dropna=False)[
                "Value"
            ]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["StageLabel"] = agg["DatasetKey"].map(key_to_stage)
        agg["StageIdx"] = agg["DatasetKey"].map(key_to_idx)
        agg["PanelTitle"] = title
        all_agg.append(agg)

        width = 0.5
        offset_step = width / max(n_models, 1)

        for model_idx, model in enumerate(models):
            model_agg = agg[agg["Model"] == model].sort_values("StageIdx")
            if model_agg.empty:
                continue
            offset = (model_idx - n_models / 2 + 0.5) * offset_step
            color = _model_color(model)
            marker = MODEL_MARKERS.get(model, "o")

            xs = [
                float(x_base[int(row["StageIdx"])]) + offset
                for _, row in model_agg.iterrows()
            ]
            means = model_agg["Mean"].tolist()
            sems = model_agg["SEM"].tolist()

            if xs:
                # Connecting line (subtle)
                ax.plot(
                    xs,
                    means,
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.5,
                    zorder=3,
                )
                # Markers with error bars
                ax.errorbar(
                    xs,
                    means,
                    yerr=sems,
                    fmt=marker,
                    label=model if panel_idx == 0 else None,
                    color=color,
                    capsize=4,
                    markersize=8,
                    alpha=0.9,
                    zorder=4,
                )

        # ── Additive reference indicator at compound stage ────────────
        compound_x = float(x_base[3])  # stage index 3 = "Compound"
        for model_idx, model in enumerate(models):
            model_agg = agg[agg["Model"] == model].sort_values("StageIdx")
            if model_agg.empty:
                continue
            offset = (model_idx - n_models / 2 + 0.5) * offset_step
            color = _model_color(model)

            stage_means: dict[int, float] = dict(
                zip(model_agg["StageIdx"].astype(int), model_agg["Mean"])
            )
            if {0, 1, 2, 3}.issubset(stage_means):
                additive = stage_means[1] + stage_means[2] - stage_means[0]
                ax.plot(
                    compound_x + offset,
                    additive,
                    marker="D",
                    markersize=7,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.8,
                    zorder=5,
                    label=(
                        "Additive ref." if panel_idx == 0 and model_idx == 0 else None
                    ),
                )

        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if panel_idx == 0:
            ax.set_ylabel(_sid_ylabel("ne-sid"), fontsize=_LABEL_FONTSIZE)

        # Panel title on the topmost row.
        axes[0, panel_idx].set_title(title, fontsize=_PANEL_TITLE_FONTSIZE)

        # ── DAG row ───────────────────────────────────────────────────
        if dag_row is not None and avici_dag_df is not None:
            dag_panel = avici_dag_df[avici_dag_df["DatasetKey"].isin(keys)]
            _plot_dag_row_panel(
                axes[dag_row, panel_idx],
                dag_panel,
                panel_stage_labels,
                x_base,
                _label_fn,
                panel_idx=panel_idx,
                id_count=0,
                stress_count=0,
            )

        # ── Error row ─────────────────────────────────────────────────
        if err_row is not None and error_df is not None:
            err_panel = error_df[error_df["DatasetKey"].isin(keys)]
            _plot_error_row_panel(
                axes[err_row, panel_idx],
                err_panel,
                panel_stage_labels,
                x_base,
                models,
                _label_fn,
                panel_idx=panel_idx,
                id_count=0,
                stress_count=0,
            )

        # ── Stage shading for all rows in this panel ──────────────────
        for row_idx in range(n_rows):
            cur_ax = axes[row_idx, panel_idx]
            for stage_idx, bg in enumerate(_ISOLATION_STAGE_BG):
                cur_ax.axvspan(
                    stage_idx - 0.5,
                    stage_idx + 0.5,
                    color=bg,
                    alpha=0.35,
                    zorder=0,
                )
            for stage_idx in range(1, n_stages):
                cur_ax.axvline(
                    stage_idx - 0.5,
                    color="#999999",
                    linestyle=":",
                    linewidth=1.0,
                    zorder=1,
                )

    # ── X tick labels on bottom row ───────────────────────────────────
    bottom_row = n_rows - 1
    for pi in range(n_panels):
        axes[bottom_row, pi].set_xticks(x_base)
        axes[bottom_row, pi].set_xticklabels(
            panel_stage_labels_all[pi],
            fontsize=_SHIFT_XTICK_FONTSIZE,
            rotation=30,
            ha="right",
        )

    # ── Shared legend ─────────────────────────────────────────────────
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

    fig.suptitle(
        "Graph-Mechanism Target Environments",
        fontsize=_TITLE_FONTSIZE,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    result = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    save_figure_data(output_path, result)
    log.info("Saved compound isolation figure to %s", output_path)
    return result


# ── Shift-progression figure (ID → Single → Compound → Stress) ────────

_PROGRESSION_STAGES: list[tuple[str, str, list[str]]] = [
    ("ID", "ID", ["id"]),
    ("Graph", "Graph Target Environments", ["graph"]),
    ("Mechanism", "Mechanism Target Environments", ["mechanism"]),
    ("Graph+Mechanism", "Graph-Mechanism Target Environments", ["compound"]),
]
"""(tick_label, legend_label, axis_categories) for each non-stress stage."""


def generate_compound_progression_figure(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate a shift-progression figure: ID -> Single -> Compound -> Stress.

    Shows how performance degrades as the number of simultaneous
    distribution shifts increases, answering whether OOD degradation
    compounds uniformly.  A bottom row shows AviCi DAG validity
    (sampled and thresholded) across the same progression stages.

    Stages:
        1. **ID** -- in-distribution baseline.
        2. **Graph Target Environments** -- graph topology OOD only.
        3. **Mechanism Target Environments** -- functional mechanism OOD only.
        4. **Graph-Mechanism Target Environments** -- graph + mechanism OOD (fixed size).
        5. **Stress Test** -- compound shift + large nodes + small samples.

    For each stage, individual family means are shown as jittered dots
    behind the per-model mean +/- SEM markers, connected by lines to
    emphasise the degradation trend.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output PDF figure.
        model_filter: If given, only include these model display names.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    metric_name = "ne-sid"

    base = raw_df[raw_df["Metric"].eq(metric_name)].copy()
    fixed = base[is_fixed_size_task_frame(base)]
    stress = base[base["DatasetKey"].isin(_EXTREME_FAMILIES)]

    if model_filter is not None:
        fixed = fixed[fixed["Model"].isin(model_filter)]
        stress = stress[stress["Model"].isin(model_filter)]

    models = _resolve_models(model_filter)
    n_models = len(models)

    # ── Prepare AviCi DAG validity data ───────────────────────────────
    dag_base = raw_df[
        raw_df["Metric"].isin(["valid_dag_pct", "threshold_valid_dag_pct"])
        & raw_df["Model"].eq("AviCi")
    ].copy()
    dag_fixed = dag_base[is_fixed_size_task_frame(dag_base)]
    dag_stress = dag_base[dag_base["DatasetKey"].isin(_EXTREME_FAMILIES)]
    has_dag_row = not dag_fixed.empty

    # ── Per-family aggregation ────────────────────────────────────────
    all_stages: list[str] = [s[1] for s in _PROGRESSION_STAGES] + ["Stress Test"]
    tick_labels: list[str] = [s[0] for s in _PROGRESSION_STAGES] + ["Stress\nTest"]

    rows: list[dict[str, object]] = []
    for tick, stage, cats in _PROGRESSION_STAGES:
        stage_data = fixed[fixed["AxisCategory"].isin(cats)]
        if stage_data.empty:
            continue
        fam_means = (
            stage_data.groupby(["Model", "DatasetKey"])["Value"]
            .mean()
            .reset_index()
            .rename(columns={"Value": "FamilyMean"})
        )
        fam_means["Stage"] = stage
        rows.extend(fam_means.to_dict("records"))

    # Stress-test stage
    if not stress.empty:
        fam_means = (
            stress.groupby(["Model", "DatasetKey"])["Value"]
            .mean()
            .reset_index()
            .rename(columns={"Value": "FamilyMean"})
        )
        fam_means["Stage"] = "Stress Test"
        rows.extend(fam_means.to_dict("records"))

    if not rows:
        log.warning("No data for compound progression figure.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── DAG validity aggregation per stage ────────────────────────────
    dag_rows: list[dict[str, object]] = []
    if has_dag_row:
        for tick, stage, cats in _PROGRESSION_STAGES:
            stage_dag = dag_fixed[dag_fixed["AxisCategory"].isin(cats)]
            if stage_dag.empty:
                continue
            fam_dag = (
                stage_dag.groupby(["Metric", "DatasetKey"])["Value"]
                .mean()
                .reset_index()
                .rename(columns={"Value": "FamilyMean"})
            )
            fam_dag["Stage"] = stage
            dag_rows.extend(fam_dag.to_dict("records"))

        if not dag_stress.empty:
            fam_dag = (
                dag_stress.groupby(["Metric", "DatasetKey"])["Value"]
                .mean()
                .reset_index()
                .rename(columns={"Value": "FamilyMean"})
            )
            fam_dag["Stage"] = "Stress Test"
            dag_rows.extend(fam_dag.to_dict("records"))

    dag_df = pd.DataFrame(dag_rows) if dag_rows else pd.DataFrame()
    has_dag_row = not dag_df.empty

    # ── Plot ──────────────────────────────────────────────────────────
    n_stages = len(all_stages)
    x_base = np.arange(n_stages)

    if has_dag_row:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=WIDE_TWO_ROW_FIGSIZE,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1.2]},
        )
        ax = axes[0]
        dag_ax = axes[1]
    else:
        fig, ax = plt.subplots(figsize=WIDE_TWO_ROW_FIGSIZE)
        dag_ax = None

    width = 0.5
    offset_step = width / max(n_models, 1)
    rng = np.random.default_rng(42)

    for model_idx, model in enumerate(models):
        model_data = df[df["Model"] == model]
        offset = (model_idx - n_models / 2 + 0.5) * offset_step
        color = _model_color(model)
        marker = MODEL_MARKERS.get(model, "o")

        stage_xs: list[float] = []
        stage_means: list[float] = []
        stage_sems: list[float] = []

        for stage_idx, stage in enumerate(all_stages):
            vals = model_data.loc[model_data["Stage"] == stage, "FamilyMean"]
            if vals.empty:
                continue

            x_center = float(x_base[stage_idx]) + offset
            stage_xs.append(x_center)
            stage_means.append(float(vals.mean()))
            stage_sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)

            # Individual family dots (jittered)
            jitter = rng.uniform(
                -offset_step * 0.25, offset_step * 0.25, size=len(vals)
            )
            ax.scatter(
                [x_center + j for j in jitter],
                vals.values,
                color=color,
                alpha=0.25,
                s=12,
                zorder=2,
                edgecolors="none",
            )

        if stage_xs:
            # Connecting line (subtle)
            ax.plot(
                stage_xs,
                stage_means,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.5,
                zorder=3,
            )
            # Markers with error bars
            ax.errorbar(
                stage_xs,
                stage_means,
                yerr=stage_sems,
                fmt=marker,
                label=model,
                color=color,
                capsize=4,
                markersize=9,
                alpha=0.9,
                zorder=4,
            )

    # ── Stage-region shading ──────────────────────────────────────────
    stage_bg = ["#f2f2f2", "#e6f0ff", "#e6f0ff", "#fff3e6", "#ffe0e0"]
    shade_axes = [ax] + ([dag_ax] if dag_ax is not None else [])
    for cur_ax in shade_axes:
        for i, bg in enumerate(stage_bg[:n_stages]):
            cur_ax.axvspan(i - 0.5, i + 0.5, color=bg, alpha=0.35, zorder=0)
        for i in range(1, n_stages):
            cur_ax.axvline(
                i - 0.5, color="#999999", linestyle=":", linewidth=1.0, zorder=1
            )

    if not has_dag_row:
        ax.set_xticks(x_base)
        ax.set_xticklabels(tick_labels, fontsize=_TICK_FONTSIZE)
    ax.set_ylabel(_sid_ylabel("ne-sid"), fontsize=_LABEL_FONTSIZE)
    ax.set_title("Target-Environment Progression", fontsize=_TITLE_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            title="Model",
            loc="upper left",
            ncol=_model_legend_ncol(len(labels)),
            fontsize=_model_legend_fontsize(len(labels)),
            frameon=True,
            framealpha=0.8,
        )

    # ── Bottom row: AviCi DAG validity ────────────────────────────────
    if dag_ax is not None and not dag_df.empty:
        _DAG_METRICS_PROG = [
            ("threshold_valid_dag_pct", "Thresholded", 0.55),
            ("valid_dag_pct", "Sampled", 0.85),
        ]
        bar_width = 0.2
        for m_idx, (m_name, m_label, m_alpha) in enumerate(_DAG_METRICS_PROG):
            m_data = dag_df[dag_df["Metric"] == m_name]
            if m_data.empty:
                continue
            dag_stage_xs: list[float] = []
            dag_stage_means: list[float] = []
            dag_stage_sems: list[float] = []

            for stage_idx, stage in enumerate(all_stages):
                vals = m_data.loc[m_data["Stage"] == stage, "FamilyMean"]
                if vals.empty:
                    continue
                offset = (m_idx - 0.5) * bar_width
                dag_stage_xs.append(float(x_base[stage_idx]) + offset)
                dag_stage_means.append(float(vals.mean()))
                dag_stage_sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)

            if dag_stage_xs:
                dag_ax.bar(
                    dag_stage_xs,
                    dag_stage_means,
                    yerr=dag_stage_sems,
                    width=bar_width,
                    color=_model_color("AviCi"),
                    alpha=m_alpha,
                    capsize=2,
                    label=m_label,
                )

        dag_ax.set_ylim(0, 105)
        dag_ax.set_xticks(x_base)
        dag_ax.set_xticklabels(tick_labels, fontsize=10)
        dag_ax.set_ylabel(_dag_ylabel(), fontsize=_LABEL_FONTSIZE)
        dag_ax.grid(True, axis="y", linestyle="--", alpha=0.4)
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

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved compound progression figure to %s", output_path)
    save_figure_data(output_path, df)
    return df


# ── Valid DAG % shift figure ───────────────────────────────────────────

_VALID_DAG_SHIFT_SPECS: dict[str, tuple[str, str]] = {
    "graph": ("graph", "Graph Target Environments"),
    "mechanism": ("mechanism", "Mechanism Target Environments"),
    "compound": ("compound", "Graph-Mechanism Target Environments"),
    "nodes": ("nodes", "Node-Count Target Environments"),
    "samples": ("samples", "Sample-Size Target Environments"),
}
"""Shift axes for the valid DAG % figure (noise omitted — explicit models
lack noise-shift data, and both amortised models are constant at 100 / ~65)."""


def _compute_dibs_valid_dag_pct(
    run_dirs: Sequence[Path],
) -> pd.DataFrame:
    """Compute per-task valid DAG % for DiBS from cached inference artifacts.

    DiBS does not record ``valid_dag_pct`` in ``metrics.json``.  Instead we
    load each ``seed_*.pt.gz`` artifact and check acyclicity of individual
    posterior samples via the matrix-exponential trace characterisation:
    a binary matrix *A* is a DAG iff ``tr(exp(A)) == n``.

    Returns:
        Long-format DataFrame with columns matching the raw_df schema:
        ``Model``, ``ModelKey``, ``DatasetKey``, ``Dataset``, ``TaskIdx``,
        ``Metric`` (== ``"valid_dag_pct"``), ``Value``, ``AxisCategory``,
        ``NNodes``, ``SamplesPerTask``.
    """
    import json

    import torch

    from causal_meta.analysis.diagnostics.posterior import _discover_artifacts
    from causal_meta.analysis.utils import _as_mapping, map_dataset_description
    from causal_meta.runners.utils.artifacts import torch_load as _torch_load

    rows: list[dict[str, object]] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir).resolve()
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r") as f:
            payload = json.load(f)
        metadata = (
            _as_mapping(payload.get("metadata")) if isinstance(payload, dict) else {}
        )
        model_name = str(metadata.get("model_name", "unknown"))
        if "dibs" not in model_name.lower():
            continue

        fam_meta = (
            _as_mapping(payload.get("family_metadata"))
            if isinstance(payload, dict)
            else {}
        )

        inference_root_raw = metadata.get("inference_root")
        inference_root: Path | None = None
        if inference_root_raw is not None and str(inference_root_raw).strip():
            candidate = Path(str(inference_root_raw)).expanduser()
            if not candidate.is_absolute():
                candidate = run_dir / candidate
            inference_root = candidate

        layout_raw = str(metadata.get("inference_layout", "")).strip().lower()
        use_model_subdir: bool | None = None
        if layout_raw == "model_dataset":
            use_model_subdir = True
        elif layout_raw == "dataset":
            use_model_subdir = False

        artifacts = _discover_artifacts(
            run_dir,
            model_name=model_name,
            inference_root=inference_root,
            use_model_subdir=use_model_subdir,
        )
        if not artifacts:
            artifacts = _discover_artifacts(run_dir, model_name=model_name)
        if not artifacts:
            continue

        for dataset_key, artifact_path in artifacts:
            try:
                artifact = _torch_load(artifact_path)
            except Exception:
                log.warning("Failed to load DiBS artifact %s", artifact_path)
                continue

            graph_samples = artifact.get("graph_samples")
            if graph_samples is None:
                continue

            if graph_samples.ndim == 4 and graph_samples.shape[0] == 1:
                graph_samples = graph_samples.squeeze(0)
            elif graph_samples.ndim != 3:
                continue

            # Binarise
            graph_samples = (graph_samples > 0.5).float()
            s, n, _ = graph_samples.shape

            # DAG check: tr(expm(A)) == n  ⟺  A is a DAG.
            # For binary matrices this is exact.
            traces = torch.zeros(s)
            for i in range(s):
                traces[i] = torch.trace(torch.matrix_exp(graph_samples[i]))
            valid_count = int((torch.abs(traces - n) < 0.5).sum().item())
            valid_pct = (valid_count / s * 100.0) if s > 0 else 0.0

            seed = artifact.get("seed", -1)
            idx = artifact.get("idx", -1)

            ds_fam = _as_mapping(fam_meta.get(dataset_key))
            n_nodes = ds_fam.get("n_nodes") if ds_fam else None
            samples_per_task = ds_fam.get("samples_per_task") if ds_fam else None

            base_row = {
                "Model": paper_model_label(model_name),
                "ModelKey": model_name,
                "DatasetKey": dataset_key,
                "Dataset": map_dataset_description(dataset_key),
                "TaskIdx": int(idx) if idx != -1 else int(seed),
                "AxisCategory": axis_category(dataset_key),
                "NNodes": int(n_nodes) if n_nodes is not None else None,
                "SamplesPerTask": (
                    int(samples_per_task) if samples_per_task is not None else None
                ),
            }
            rows.append({**base_row, "Metric": "valid_dag_pct", "Value": valid_pct})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Dataset"] = df.apply(
            lambda row: thesis_dataset_label(
                str(row["DatasetKey"]), str(row["Dataset"])
            ),
            axis=1,
        )
    n_metrics = df["Metric"].nunique() if not df.empty else 0
    log.info(
        "Computed DiBS valid DAG metrics for %d rows (%d metrics)",
        len(df),
        n_metrics,
    )
    return df


def generate_valid_dag_shift_figure(
    raw_df: pd.DataFrame,
    run_dirs: Sequence[Path],
    *,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a multi-panel figure showing valid DAG % across shift axes.

    One panel per shift axis (graph, mechanism, compound, nodes, samples).
    AviCi and DiBS are plotted with errorbars; BCNP and BayesDAG are shown
    as horizontal reference lines at 100%.

    DiBS contributes only ``valid_dag_pct`` (sample-level DAG validity). AviCi
    may additionally contribute ``threshold_valid_dag_pct`` to distinguish
    posterior-sample acyclicity from point-estimate acyclicity.

    Args:
        raw_df: Long-format raw task DataFrame (must contain ``valid_dag_pct``
            and optionally ``threshold_valid_dag_pct``).
        run_dirs: Run directories (needed to compute DiBS valid_dag_pct from
            inference artifacts).
        output_path: Path for the output PDF figure.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    # ── 1. Gather data for both metrics ───────────────────────────────
    _METRICS = ["valid_dag_pct", "threshold_valid_dag_pct"]
    vdp = raw_df[raw_df["Metric"].isin(_METRICS)].copy()

    # Compute sampled DiBS DAG validity from artifacts and append.
    dibs_vdp = _compute_dibs_valid_dag_pct(run_dirs)
    if not dibs_vdp.empty:
        shared_cols = [c for c in vdp.columns if c in dibs_vdp.columns]
        vdp = pd.concat([vdp[shared_cols], dibs_vdp[shared_cols]], ignore_index=True)

    if vdp.empty:
        log.warning("No valid_dag_pct data available; skipping valid DAG shift figure.")
        return pd.DataFrame()

    # Values are already in percentage (0–100) for both raw_df and DiBS artifacts.

    # Models that actually have per-task variation
    variable_models = {"AviCi", "DiBS"}
    # Models that are constant at 100%
    constant_models = {"BCNP", "BayesDAG"}

    # Metric display config: (label_suffix, alpha, marker_size, linestyle)
    _METRIC_STYLE: dict[str, tuple[str, float, int, str]] = {
        "valid_dag_pct": ("Sample", 0.9, 7, "-"),
        "threshold_valid_dag_pct": ("Threshold", 0.35, 6, "--"),
    }

    shift_keys = list(_VALID_DAG_SHIFT_SPECS.keys())
    n_panels = len(shift_keys)
    # 2×3 grid: top row = graph/mechanism/compound, bottom = nodes/samples/empty
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=grid_figsize(n_cols, n_rows), squeeze=False
    )
    # Map panel index to (row, col)
    _panel_pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    # Hide unused cell(s)
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in _panel_pos[:n_panels]:
                axes[r, c].set_visible(False)

    all_agg_rows: list[pd.DataFrame] = []

    for panel_idx, shift_key in enumerate(shift_keys):
        r, c = _panel_pos[panel_idx]
        ax = axes[r, c]
        axis_cat, axis_title = _VALID_DAG_SHIFT_SPECS[shift_key]

        # Filter to ID + this shift axis
        panel_df = vdp[vdp["AxisCategory"].isin(["id", axis_cat])].copy()

        # For non-transfer axes, restrict to fixed size
        if shift_key not in ("nodes", "samples"):
            panel_df = panel_df[is_fixed_size_task_frame(panel_df)]

        # ── Isolate the shift axis ────────────────────────────────────
        if shift_key == "graph":
            # Use only Linear mechanism (representative) for this overview
            panel_df = panel_df[
                panel_df["DatasetKey"].map(lambda dk: id_mechanism_of(dk) == "linear")
            ].copy()
        elif shift_key == "mechanism":
            # Use ER-20 as representative graph anchor (matches first
            # anchor in the multi-panel mechanism shift figure).
            # Filter both ID and OOD-mech to ER-20 to avoid mixing anchors.
            panel_df = _restrict_to_graph_anchor(panel_df, "er20")
        elif shift_key == "compound":
            # Restrict ID to the same 3 representatives as the main
            # compound figure to avoid cluttering the overview panel.
            _COMPOUND_ID_REPS = {
                ("linear", "er20"),
                ("neuralnet", "sf2"),
                ("gpcde", "er60"),
            }
            id_mask = panel_df["AxisCategory"] == "id"
            keep_id = id_mask & panel_df["DatasetKey"].map(
                lambda dk: (id_mechanism_of(dk), graph_code_of(dk)) in _COMPOUND_ID_REPS
            )
            panel_df = panel_df[keep_id | ~id_mask].copy()

        if panel_df.empty:
            ax.set_visible(False)
            continue

        # Aggregate per (Model, DatasetKey, Metric)
        agg = (
            panel_df.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory", "Metric"],
                dropna=False,
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )

        # For graph shift panels use short graph-topology labels
        if shift_key == "graph":
            agg["DatasetLabel"] = agg["DatasetKey"].map(_graph_shift_label)
        elif shift_key == "mechanism":
            agg["DatasetLabel"] = agg["DatasetKey"].map(_mech_shift_label)
        else:
            agg["DatasetLabel"] = agg.apply(
                lambda row: thesis_dataset_label(
                    str(row["DatasetKey"]), str(row["Dataset"])
                ),
                axis=1,
            )
        agg["_sort"] = agg["AxisCategory"].map({"id": 0}).fillna(1)
        agg = agg.sort_values(["_sort", "DatasetLabel"]).drop(columns=["_sort"])
        all_agg_rows.append(agg)

        # Dataset ordering is derived from sample-level metric only to keep
        # x-axis stable; threshold rows share the same dataset labels.
        agg_sample = agg[agg["Metric"] == "valid_dag_pct"]
        datasets = list(agg_sample["DatasetLabel"].unique())
        axis_lookup = (
            agg_sample[["DatasetLabel", "AxisCategory"]]
            .drop_duplicates()
            .set_index("DatasetLabel")["AxisCategory"]
            .to_dict()
        )
        n_datasets = len(datasets)
        variable_present = [
            m
            for m in PAPER_MODEL_LABELS.values()
            if m in variable_models and m in agg["Model"].unique()
        ]
        n_var = len(variable_present)
        width = 0.6
        offset_step = width / max(n_var, 1)
        x_base = np.arange(n_datasets)

        # Plot both metrics for each variable model
        for metric_key, (label_sfx, alpha, msize, _ls) in _METRIC_STYLE.items():
            metric_agg = agg[agg["Metric"] == metric_key]
            if metric_agg.empty:
                continue

            for model_idx, model in enumerate(variable_present):
                model_agg = metric_agg[metric_agg["Model"] == model]
                xs: list[float] = []
                means: list[float] = []
                sems: list[float] = []
                for i, ds in enumerate(datasets):
                    row = model_agg[model_agg["DatasetLabel"] == ds]
                    if row.empty:
                        continue
                    offset = (model_idx - n_var / 2 + 0.5) * offset_step
                    xs.append(float(x_base[i]) + offset)
                    means.append(float(row.iloc[0]["Mean"]))
                    sems.append(float(row.iloc[0]["SEM"]))
                if xs:
                    # Use filled marker for sample-level, open for threshold
                    marker = MODEL_MARKERS.get(model, "o")
                    fillstyle = "full" if metric_key == "valid_dag_pct" else "none"
                    label = (
                        f"{model} ({label_sfx})"
                        if metric_key == "threshold_valid_dag_pct"
                        else model
                    )
                    ax.errorbar(
                        xs,
                        means,
                        yerr=sems,
                        fmt=marker,
                        label=label,
                        color=_model_color(model),
                        capsize=3,
                        markersize=msize,
                        alpha=alpha,
                        fillstyle=fillstyle,
                    )

        # Reference lines for constant-100% models
        for const_model in sorted(constant_models):
            ax.axhline(
                100.0,
                color=_model_color(const_model),
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
                label=f"{const_model} (100%)",
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(_dag_ylabel(), fontsize=_LABEL_FONTSIZE)
        ax.set_ylim(-5, 110)
        if shift_key == "graph":
            panel_title = f"{axis_title} (Linear)"
        elif shift_key == "mechanism":
            panel_title = f"{axis_title} (ER-20)"
        else:
            panel_title = axis_title
        ax.set_title(panel_title, fontsize=_PANEL_TITLE_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Grey ID region
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    # Shared legend on top — deduplicate handles from first visible axis
    all_handles: list = []
    all_labels: list[str] = []
    seen: set[str] = set()
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen:
                all_handles.append(handle)
                all_labels.append(label)
                seen.add(label)
    if all_handles:
        axes[0, 0].legend(
            all_handles,
            all_labels,
            title="Model",
            loc="best",
            ncol=_model_legend_ncol(len(all_labels)),
            fontsize=_model_legend_fontsize(len(all_labels)),
            frameon=True,
            framealpha=0.8,
        )

    fig.suptitle(
        "DAG Structural Validity Under Distribution Shift",
        fontsize=_TITLE_FONTSIZE,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    combined = (
        pd.concat(all_agg_rows, ignore_index=True) if all_agg_rows else pd.DataFrame()
    )
    save_figure_data(output_path, combined)
    return combined


# ── Error decomposition (FP / FN / Reversed) shift table ──────────────

_ERROR_DECOMP_SHIFT_SPECS: dict[str, tuple[str, str]] = {
    "graph": ("graph", "Graph Target Environments"),
    "mechanism": ("mechanism", "Mechanism Target Environments"),
    "noise": ("noise", "Noise Target Environments"),
    "compound": ("compound", "Graph-Mechanism Target Environments"),
}


def generate_error_decomposition_table(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate a LaTeX table decomposing SHD into FP/FN/reversed per model per shift.

    Layout: rows = models, column groups = shift axes (graph, mechanism,
    noise, compound), sub-columns = mean FP / FN / Reversed counts.

    Args:
        raw_df: Long-format raw task DataFrame (must contain ``fp_count``,
            ``fn_count``, ``reversed_count``).
        output_path: Path for the output ``.tex`` file.

    Returns:
        Aggregated DataFrame used for the table.
    """
    needed = {"fp_count", "fn_count", "reversed_count"}
    available = set(raw_df["Metric"].unique()) if not raw_df.empty else set()
    if not needed.issubset(available):
        log.warning(
            "Missing error decomposition metrics %s; skipping table.",
            needed - available,
        )
        return pd.DataFrame()

    error_df = raw_df[raw_df["Metric"].isin(needed)].copy()
    error_df = error_df[is_fixed_size_task_frame(error_df)]

    if error_df.empty:
        log.warning("No fixed-size error decomposition data; skipping table.")
        return pd.DataFrame()

    from causal_meta.analysis.diagnostics.failure_modes import _pivot_raw_wide

    wide = _pivot_raw_wide(error_df)
    if wide.empty:
        return pd.DataFrame()

    for col in needed:
        if col not in wide.columns:
            log.warning(
                "Column %s not found after pivot; skipping error decomp table.", col
            )
            return pd.DataFrame()

    wide["AxisCategory"] = wide["DatasetKey"].map(axis_category)

    shift_specs = list(_ERROR_DECOMP_SHIFT_SPECS.items())
    all_models = _resolve_models(model_filter)
    models = [m for m in all_models if m in wide["Model"].unique()]

    if not models:
        return pd.DataFrame()

    all_agg: list[pd.DataFrame] = []

    # Aggregate: mean ± SD across family-level means per (Model, ShiftAxis).
    for shift_key, (axis_cat, axis_title) in shift_specs:
        panel = wide[wide["AxisCategory"].isin(["id", axis_cat])].copy()

        # Apply the same filtering as the figure version
        if shift_key == "graph":
            panel = panel[
                panel["DatasetKey"].map(lambda dk: id_mechanism_of(dk) == "linear")
            ].copy()
        elif shift_key == "mechanism":
            panel = _restrict_to_graph_anchor(panel, "er20")
        elif shift_key == "compound":
            _COMPOUND_ID_REPS = {
                ("linear", "er20"),
                ("neuralnet", "sf2"),
                ("gpcde", "er60"),
            }
            id_mask = panel["AxisCategory"] == "id"
            keep_id = id_mask & panel["DatasetKey"].map(
                lambda dk: (id_mechanism_of(dk), graph_code_of(dk)) in _COMPOUND_ID_REPS
            )
            panel = panel[keep_id | ~id_mask].copy()

        # Only OOD rows for the degradation summary (ID is reference)
        ood_panel = panel[panel["AxisCategory"] == axis_cat]
        if ood_panel.empty:
            continue

        family_means = (
            ood_panel.groupby(["Model", "DatasetKey"], dropna=False)[list(needed)]
            .mean()
            .reset_index()
        )
        agg = (
            family_means.groupby("Model")[list(needed)]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg.columns
        ]
        agg = agg.rename(
            columns={f"{metric_key}_std": f"{metric_key}_sd" for metric_key in needed}
        )
        agg[[f"{metric_key}_sd" for metric_key in needed]] = agg[
            [f"{metric_key}_sd" for metric_key in needed]
        ].fillna(0.0)
        agg["ShiftAxis"] = axis_title
        all_agg.append(agg)

    if not all_agg:
        return pd.DataFrame()

    combined = pd.concat(all_agg, ignore_index=True)

    # Build LaTeX table
    n_shifts = len(shift_specs)
    comp_labels = [("FP", "fp_count"), ("FN", "fn_count"), ("Rev.", "reversed_count")]
    n_sub = len(comp_labels)

    lines: list[str] = []
    lines.append(r"\footnotesize")

    # Column spec: Model + 3 sub-cols per shift axis
    col_spec = "l" + ("|" + "r" * n_sub) * n_shifts
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: shift axis group headers
    header1 = r"\textbf{Model}"
    for _sk, (_ac, at) in shift_specs:
        header1 += rf" & \multicolumn{{{n_sub}}}{{c}}{{\textbf{{{at}}}}}"
    header1 += r" \\"
    lines.append(header1)

    # Cmidrules
    cmidrules = ""
    col_start = 2
    for _ in shift_specs:
        col_end = col_start + n_sub - 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
        col_start = col_end + 1
    lines.append(cmidrules)

    # Header row 2: sub-column labels
    header2 = ""
    for _ in shift_specs:
        for label, _ in comp_labels:
            header2 += rf" & \textbf{{{label}}}"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    # Data rows
    # Pre-compute best (lowest) value per metric per shift axis.
    best_vals: dict[tuple[str, str], float] = {}
    for _sk, (_ac, at) in shift_specs:
        for _, metric_key in comp_labels:
            col = f"{metric_key}_mean"
            axis_rows = combined[combined["ShiftAxis"] == at]
            if not axis_rows.empty and col in axis_rows.columns:
                valid = axis_rows[col].dropna()
                if not valid.empty:
                    best_vals[(at, metric_key)] = float(valid.min())

    for model in models:
        row_str = model
        for _sk, (_ac, at) in shift_specs:
            model_row = combined[
                (combined["Model"] == model) & (combined["ShiftAxis"] == at)
            ]
            if model_row.empty:
                for _ in comp_labels:
                    row_str += " & --"
            else:
                r = model_row.iloc[0]
                for _, metric_key in comp_labels:
                    mean_val = r.get(f"{metric_key}_mean", float("nan"))
                    sd_val = r.get(f"{metric_key}_sd", float("nan"))
                    if np.isnan(mean_val):
                        row_str += " & --"
                    else:
                        cell = rf"${mean_val:.1f} \pm {sd_val:.1f}$"
                        best = best_vals.get((at, metric_key))
                        if best is not None and abs(mean_val - best) < 1e-9:
                            cell = r"\textbf{" + cell + "}"
                        row_str += " & " + cell
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    output_path.write_text("\n".join(lines) + "\n")
    log.info("Saved error decomposition table to %s", output_path)
    return combined


def generate_error_decomposition_figure(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate a grouped stacked-bar chart of FP / FN / reversed per model per shift.

    Layout: one panel (axis group) per column, models on the x-axis within each
    panel.  Each bar is stacked with three segments: false positives (FP),
    false negatives (FN), and reversed edges.

    Args:
        raw_df: Long-format raw task DataFrame (must contain ``fp_count``,
            ``fn_count``, ``reversed_count``).
        output_path: Path for the output PDF figure.
        model_filter: If given, only plot these models (display names).

    Returns:
        Aggregated DataFrame used for plotting.
    """
    needed = {"fp_count", "fn_count", "reversed_count"}
    available = set(raw_df["Metric"].unique()) if not raw_df.empty else set()
    if not needed.issubset(available):
        log.warning(
            "Missing error decomposition metrics %s; skipping figure.",
            needed - available,
        )
        return pd.DataFrame()

    error_df = raw_df[raw_df["Metric"].isin(needed)].copy()
    error_df = error_df[is_fixed_size_task_frame(error_df)]

    if error_df.empty:
        log.warning("No fixed-size error decomposition data; skipping figure.")
        return pd.DataFrame()

    from causal_meta.analysis.diagnostics.failure_modes import _pivot_raw_wide

    wide = _pivot_raw_wide(error_df)
    if wide.empty:
        return pd.DataFrame()

    for col in needed:
        if col not in wide.columns:
            log.warning(
                "Column %s not found after pivot; skipping error decomp figure.", col
            )
            return pd.DataFrame()

    wide["AxisCategory"] = wide["DatasetKey"].map(axis_category)

    shift_specs = list(_ERROR_DECOMP_SHIFT_SPECS.items())
    all_models = _resolve_models(model_filter)
    models = [m for m in all_models if m in wide["Model"].unique()]

    if not models:
        return pd.DataFrame()

    all_agg: list[pd.DataFrame] = []

    # Aggregate: mean FP / FN / reversed per (Model, ShiftAxis)
    for shift_key, (axis_cat, axis_title) in shift_specs:
        panel = wide[wide["AxisCategory"].isin(["id", axis_cat])].copy()

        if shift_key == "graph":
            panel = panel[
                panel["DatasetKey"].map(lambda dk: id_mechanism_of(dk) == "linear")
            ].copy()
        elif shift_key == "mechanism":
            panel = _restrict_to_graph_anchor(panel, "er20")
        elif shift_key == "compound":
            _COMPOUND_ID_REPS = {
                ("linear", "er20"),
                ("neuralnet", "sf2"),
                ("gpcde", "er60"),
            }
            id_mask = panel["AxisCategory"] == "id"
            keep_id = id_mask & panel["DatasetKey"].map(
                lambda dk: (id_mechanism_of(dk), graph_code_of(dk)) in _COMPOUND_ID_REPS
            )
            panel = panel[keep_id | ~id_mask].copy()

        ood_panel = panel[panel["AxisCategory"] == axis_cat]
        if ood_panel.empty:
            continue

        agg = ood_panel.groupby("Model")[list(needed)].mean().reset_index()
        agg["ShiftAxis"] = axis_title
        all_agg.append(agg)

    if not all_agg:
        return pd.DataFrame()

    combined = pd.concat(all_agg, ignore_index=True)

    # ── Plot: one panel per shift axis ────────────────────────────────
    n_panels = len(shift_specs)
    fig, axes = plt.subplots(
        1, n_panels, figsize=grid_figsize(n_panels), sharey=False, squeeze=False
    )
    axes_flat = axes[0]

    comp_keys = ERROR_SPECS

    for panel_idx, (_shift_key, (_axis_cat, axis_title)) in enumerate(shift_specs):
        ax = axes_flat[panel_idx]
        panel_data = combined[combined["ShiftAxis"] == axis_title]

        x = np.arange(len(models))
        bottoms = np.zeros(len(models))

        for comp_col, comp_label, comp_color in comp_keys:
            heights = []
            for model in models:
                row = panel_data[panel_data["Model"] == model]
                if row.empty:
                    heights.append(0.0)
                else:
                    heights.append(float(row[comp_col].iloc[0]))
            heights_arr = np.array(heights)
            ax.bar(
                x,
                heights_arr,
                bottom=bottoms,
                width=0.55,
                label=comp_label if panel_idx == 0 else None,
                color=comp_color,
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += heights_arr

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9, rotation=30, ha="right")
        ax.set_title(axis_title, fontsize=_PANEL_TITLE_FONTSIZE)
        if panel_idx == 0:
            ax.set_ylabel("Mean edge error count", fontsize=_LABEL_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Shared legend on top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(
            handles,
            labels,
            title="Error Type",
            loc="best",
            ncol=len(labels),
            fontsize=_LEGEND_FONTSIZE,
            frameon=True,
            framealpha=0.8,
        )

    fig.suptitle(
        "SHD Error Decomposition Across Shift Axes",
        fontsize=_TITLE_FONTSIZE,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved error decomposition figure to %s", output_path)
    save_figure_data(output_path, combined)
    return combined


# ── Extreme stress-test constants (used by compound+stress merge) ──────

_EXTREME_FAMILIES: list[str] = [
    "ood_both_sbm_periodic_d60_n100",
    "ood_both_ws_pnl_tanh_d60_n50",
    "ood_both_grg_logistic_map_d60_n50",
]
"""Dataset keys for the three extreme compound-shift stress-test families.

These families combine OOD graph + OOD mechanism with large node counts and
severely reduced sample budgets (d=60, n=50 or n=100), so they are excluded
by :func:`is_fixed_size_task_frame` and do not appear in the standard
fixed-size shift figures.
"""

_EXTREME_LABELS: dict[str, str] = {
    "ood_both_sbm_periodic_d60_n100": "SBM+Periodic\n($N_G{=}60$, $N{=}100$)",
    "ood_both_ws_pnl_tanh_d60_n50": "WS+PNL\n($N_G{=}60$, $N{=}50$)",
    "ood_both_grg_logistic_map_d60_n50": "GRG+Logistic\n($N_G{=}60$, $N{=}50$)",
}


# ── RQ1/RQ2 cross-axis summary table ──────────────────────────────────

_SUMMARY_TABLE_AXES: dict[str, str] = {
    "id": "ID",
    "graph": "Graph",
    "mechanism": "Mech.",
    "noise": "Noise",
    "compound": "Graph-Mechanism",
    "nodes": "Nodes",
    "samples": "Samples",
}
"""Column groups for the cross-axis summary table."""

_SUMMARY_TABLE_METRICS: list[tuple[str, str, bool]] = [
    ("ne-sid", rf"n${_HAT_E}$-SID $\downarrow$", False),
    ("e-edgef1", rf"${_HAT_E}$-F1 $\uparrow$", True),
    ("ne-shd", rf"n${_HAT_E}$-SHD $\downarrow$", False),
]
"""(metric_key, display_label, higher_is_better) for the summary table."""


def generate_results_anchor_table(
    raw_df: pd.DataFrame,
    output_path: Path,
    *,
    model_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate the anchor overview table (ID vs OOD) for the results chapter.

    Rows = models.  Column groups = ne-SID, ne-SHD, E-Edge F1, Valid DAG (%),
    each split into ID and OOD sub-columns, plus a single mean inference-time
    column.  Best per sub-column (excl. Random) is bolded.

    Args:
        raw_df: Long-format raw task DataFrame with ``AxisCategory`` column.
        output_path: Where to write the ``.tex`` file.
        model_filter: If given, restrict to these model display names.

    Returns:
        Aggregated DataFrame used for the table.
    """
    _ANCHOR_METRICS: list[tuple[str, str, bool]] = [
        ("ne-sid", rf"n${_HAT_E}$-SID $\downarrow$", False),
        ("ne-shd", rf"n${_HAT_E}$-SHD $\downarrow$", False),
        ("e-edgef1", rf"${_HAT_E}$-Edge F1 $\uparrow$", True),
        ("overview_valid_dag_pct", r"Valid DAG (\%) $\uparrow$", True),
    ]
    metric_keys = [m for m, _, _ in _ANCHOR_METRICS if m != "overview_valid_dag_pct"]
    metric_keys.extend(["valid_dag_pct", "threshold_valid_dag_pct"])
    subset = raw_df[raw_df["Metric"].isin(metric_keys)].copy()
    if subset.empty:
        log.warning("No data for results anchor table; skipping.")
        return pd.DataFrame()

    all_models = _resolve_models(model_filter)
    models = [m for m in all_models if m in subset["Model"].unique()]
    if not models:
        log.warning("No models for results anchor table; skipping.")
        return pd.DataFrame()

    # Split into ID vs OOD (everything that is not "id" or "other")
    id_mask = subset["AxisCategory"] == "id"
    ood_mask = ~subset["AxisCategory"].isin(["id", "other"])

    all_agg: list[pd.DataFrame] = []
    for split_label, mask in [("ID", id_mask), ("OOD", ood_mask)]:
        pool = subset[mask]
        if pool.empty:
            continue
        dag_model_metric = {
            "AviCi": "threshold_valid_dag_pct",
            "DiBS": "valid_dag_pct",
            "BCNP": "valid_dag_pct",
            "BayesDAG": "valid_dag_pct",
            "Random": "valid_dag_pct",
        }
        dag_rows = []
        for model, metric_name in dag_model_metric.items():
            model_dag = pool[
                pool["Model"].eq(model) & pool["Metric"].eq(metric_name)
            ].copy()
            if model_dag.empty:
                continue
            model_dag["Metric"] = "overview_valid_dag_pct"
            dag_rows.append(model_dag)
        if dag_rows:
            pool = pd.concat([pool, *dag_rows], ignore_index=True)
        pool = pool[pool["Metric"].isin([m for m, _, _ in _ANCHOR_METRICS])]

        family_means = (
            pool.groupby(["Model", "Metric", "DatasetKey"], dropna=False)["Value"]
            .mean()
            .reset_index()
        )
        agg = (
            family_means.groupby(["Model", "Metric"], dropna=False)["Value"]
            .agg(Mean="mean", SD=lambda values: float(values.std(ddof=1)))
            .reset_index()
        )
        agg["SD"] = agg["SD"].fillna(0.0)
        agg["Split"] = split_label
        all_agg.append(agg)

    if not all_agg:
        return pd.DataFrame()

    combined = pd.concat(all_agg, ignore_index=True)

    # ── Mean inference time per model (single column, not split) ───────
    time_df = raw_df[raw_df["Metric"] == "inference_time_s"]
    time_per_model: dict[str, float] = {}
    if not time_df.empty:
        time_per_model = time_df.groupby("Model")["Value"].mean().to_dict()

    # Best per (Split, Metric), excluding Random
    best_lookup: dict[tuple[str, str], float] = {}
    non_random = combined[combined["Model"] != "Random"]
    for _, row in non_random.iterrows():
        key = (str(row["Split"]), str(row["Metric"]))
        mean = float(row["Mean"])
        higher = next(
            (hib for mk, _, hib in _ANCHOR_METRICS if mk == row["Metric"]),
            False,
        )
        current = best_lookup.get(key)
        if current is None:
            best_lookup[key] = mean
        elif higher and mean > current:
            best_lookup[key] = mean
        elif not higher and mean < current:
            best_lookup[key] = mean

    # ── Build LaTeX ────────────────────────────────────────────────────
    n_metrics = len(_ANCHOR_METRICS)
    has_time = bool(time_per_model)
    col_spec = "l " + " ".join(["cc"] * n_metrics)
    if has_time:
        col_spec += " c"
    lines: list[str] = []
    lines.append(r"\footnotesize")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: metric group headers
    header1 = r"\textbf{Model}"
    col_idx = 2
    for _, mlabel, _ in _ANCHOR_METRICS:
        header1 += rf" & \multicolumn{{2}}{{c}}{{\textbf{{{mlabel}}}}}"
    if has_time:
        header1 += r" & \textbf{Time (s) $\downarrow$}"
    header1 += r" \\"
    lines.append(header1)

    # Cmidrules
    cmidrules = ""
    col_start = 2
    for _ in _ANCHOR_METRICS:
        col_end = col_start + 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
        col_start = col_end + 1
    lines.append(cmidrules)

    # Header row 2: ID / OOD
    header2 = ""
    for _ in _ANCHOR_METRICS:
        header2 += " & ID & OOD"
    if has_time:
        header2 += " &"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    # Data rows
    for model in models:
        row_str = model
        for metric_key, _, _ in _ANCHOR_METRICS:
            for split_label in ("ID", "OOD"):
                cell_row = combined[
                    (combined["Model"] == model)
                    & (combined["Metric"] == metric_key)
                    & (combined["Split"] == split_label)
                ]
                if cell_row.empty:
                    row_str += " & --"
                    continue
                mean = float(cell_row.iloc[0]["Mean"])
                sd = float(cell_row.iloc[0]["SD"])
                cell = rf"${mean:.1f} \pm {sd:.1f}$"
                best_val = best_lookup.get((split_label, metric_key))
                if (
                    best_val is not None
                    and model != "Random"
                    and abs(mean - best_val) < 1e-6
                ):
                    cell = _bold_if_best(cell, is_best=True)
                row_str += f" & {cell}"
        if has_time:
            t = time_per_model.get(model)
            if t is not None:
                row_str += rf" & ${t:.3f}$" if t < 1.0 else rf" & ${t:.1f}$"
            else:
                row_str += " & --"
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    log.info("Wrote results anchor table to %s", output_path)

    # Also save underlying data as CSV for verification
    csv_path = output_path.with_suffix(".csv")
    combined.to_csv(csv_path, index=False)
    log.info("Wrote results anchor data to %s", csv_path)

    return combined


def generate_cross_axis_summary_table(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate a LaTeX table summarising three core metrics per model per shift axis.

    Rows = models, column groups = shift axes (ID, Graph, Mechanism, Noise,
    Compound, Nodes, Samples), sub-columns = ne-SID, E-F1, ne-SHD.
    Per-column best values are bolded.

    For the four fixed-size axes (graph, mechanism, noise, compound) only
    ``d=20, n=500`` families are included.  Transfer axes (nodes, samples)
    include all sizes.  The ID column always uses fixed-size families.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output ``.tex`` file.
        model_filter: If given, only include these model display names.

    Returns:
        Aggregated DataFrame used for the table.
    """
    metric_keys = [m for m, _, _ in _SUMMARY_TABLE_METRICS]
    subset = raw_df[raw_df["Metric"].isin(metric_keys)].copy()
    if subset.empty:
        log.warning("No data for cross-axis summary table; skipping.")
        return pd.DataFrame()

    fixed = subset[is_fixed_size_task_frame(subset)]
    all_models = _resolve_models(model_filter)
    models = [m for m in all_models if m in subset["Model"].unique()]
    if not models:
        log.warning("No models for cross-axis summary table; skipping.")
        return pd.DataFrame()

    # Aggregate: mean ± SD across family-level means per (Model, Axis, Metric).
    all_agg: list[pd.DataFrame] = []
    for axis_key, axis_label in _SUMMARY_TABLE_AXES.items():
        if axis_key in ("nodes", "samples"):
            pool = subset[subset["AxisCategory"] == axis_key]
        elif axis_key == "id":
            pool = fixed[fixed["AxisCategory"] == "id"]
        else:
            pool = fixed[fixed["AxisCategory"] == axis_key]

        if pool.empty:
            continue

        family_means = (
            pool.groupby(["Model", "Metric", "DatasetKey"], dropna=False)["Value"]
            .mean()
            .reset_index()
        )
        agg = (
            family_means.groupby(["Model", "Metric"], dropna=False)["Value"]
            .agg(Mean="mean", SD=lambda values: float(values.std(ddof=1)))
            .reset_index()
        )
        agg["SD"] = agg["SD"].fillna(0.0)
        agg["Axis"] = axis_label
        all_agg.append(agg)

    if not all_agg:
        return pd.DataFrame()

    combined = pd.concat(all_agg, ignore_index=True)

    # Determine per-column best (best model for each axis × metric)
    best_lookup: dict[tuple[str, str], float] = {}
    for _, row in combined.iterrows():
        key = (str(row["Axis"]), str(row["Metric"]))
        mean = float(row["Mean"])
        higher = next(
            (hib for mk, _, hib in _SUMMARY_TABLE_METRICS if mk == row["Metric"]),
            False,
        )
        current = best_lookup.get(key)
        if current is None:
            best_lookup[key] = mean
        elif higher and mean > current:
            best_lookup[key] = mean
        elif not higher and mean < current:
            best_lookup[key] = mean

    # ── Build LaTeX ────────────────────────────────────────────────────
    axes_list = [
        (ak, al)
        for ak, al in _SUMMARY_TABLE_AXES.items()
        if al in combined["Axis"].unique()
    ]
    n_axes = len(axes_list)
    n_sub = len(_SUMMARY_TABLE_METRICS)

    # Split axes into two groups for a landscape-friendly two-part table
    # when there are more than 4 axes.
    if n_axes > 4:
        split = (n_axes + 1) // 2  # e.g. 4 and 3 for 7 axes
        axis_groups = [axes_list[:split], axes_list[split:]]
    else:
        axis_groups = [axes_list]

    lines: list[str] = []
    lines.append(r"\footnotesize")

    for grp_idx, grp_axes in enumerate(axis_groups):
        n_grp = len(grp_axes)
        lines.append(r"\resizebox{\textwidth}{!}{%")

        col_spec = "l" + ("|" + "r" * n_sub) * n_grp
        lines.append(r"\begin{tabular}{" + col_spec + "}")
        lines.append(r"\toprule")

        # Header row 1: axis group headers
        header1 = r"\textbf{Model}"
        for _, axis_label in grp_axes:
            header1 += rf" & \multicolumn{{{n_sub}}}{{c}}{{\textbf{{{axis_label}}}}}"
        header1 += r" \\"
        lines.append(header1)

        # Cmidrules
        cmidrules = ""
        col_start = 2
        for _ in grp_axes:
            col_end = col_start + n_sub - 1
            cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
            col_start = col_end + 1
        lines.append(cmidrules)

        # Header row 2: metric sub-column labels
        header2 = ""
        for _ in grp_axes:
            for _, mlabel, _ in _SUMMARY_TABLE_METRICS:
                header2 += rf" & \textbf{{{mlabel}}}"
        header2 += r" \\"
        lines.append(header2)
        lines.append(r"\midrule")

        # Data rows
        for model in models:
            row_str = model
            for _, axis_label in grp_axes:
                for metric_key, _, higher_is_better in _SUMMARY_TABLE_METRICS:
                    cell_row = combined[
                        (combined["Model"] == model)
                        & (combined["Axis"] == axis_label)
                        & (combined["Metric"] == metric_key)
                    ]
                    if cell_row.empty:
                        row_str += " & --"
                        continue
                    mean = float(cell_row.iloc[0]["Mean"])
                    sd = float(cell_row.iloc[0]["SD"])
                    cell = format_value(mean, sd)
                    best_val = best_lookup.get((axis_label, metric_key))
                    # Exclude Random from bolding contest
                    if (
                        best_val is not None
                        and model != "Random"
                        and abs(mean - best_val) < 1e-6
                    ):
                        cell = _bold_if_best(cell, is_best=True)
                    row_str += f" & {cell}"
            row_str += r" \\"
            lines.append(row_str)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")

        # Add vertical spacing between sub-tables
        if grp_idx < len(axis_groups) - 1:
            lines.append(r"\vspace{0.5em}")

    output_path.write_text("\n".join(lines) + "\n")
    log.info("Saved cross-axis summary table to %s", output_path)
    return combined


# ── Structural decomposition summary table ────────────────────────────

_STRUCTURAL_TABLE_METRICS: list[tuple[str, str, bool]] = [
    ("skeleton_f1", r"Skel.\ F1 $\uparrow$", True),
    ("orientation_accuracy", r"Orient.\ Acc $\uparrow$", True),
    ("sparsity_ratio", r"Sparsity", False),  # neither direction is universally better
]
"""(metric_key, display_label, higher_is_better) for the structural table.

For ``sparsity_ratio`` bolding is disabled because neither direction is
universally better — it depends on the ground-truth graph density.
"""


def generate_structural_summary_table(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate a LaTeX table for skeleton F1, orientation accuracy, and sparsity.

    Layout mirrors :func:`generate_cross_axis_summary_table`: rows = models,
    column groups = shift axes, sub-columns = the three structural metrics.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output ``.tex`` file.
        model_filter: If given, only include these model display names.

    Returns:
        Aggregated DataFrame used for the table.
    """
    metric_keys = [m for m, _, _ in _STRUCTURAL_TABLE_METRICS]
    subset = raw_df[raw_df["Metric"].isin(metric_keys)].copy()
    if subset.empty:
        log.warning("No data for structural summary table; skipping.")
        return pd.DataFrame()

    fixed = subset[is_fixed_size_task_frame(subset)]
    all_models = _resolve_models(model_filter)
    models = [m for m in all_models if m in subset["Model"].unique()]
    if not models:
        log.warning("No models for structural summary table; skipping.")
        return pd.DataFrame()

    # Aggregate: mean ± SEM per (Model, AxisCategory, Metric)
    all_agg: list[pd.DataFrame] = []
    for axis_key, axis_label in _SUMMARY_TABLE_AXES.items():
        if axis_key in ("nodes", "samples"):
            pool = subset[subset["AxisCategory"] == axis_key]
        elif axis_key == "id":
            pool = fixed[fixed["AxisCategory"] == "id"]
        else:
            pool = fixed[fixed["AxisCategory"] == axis_key]

        if pool.empty:
            continue

        agg = (
            pool.groupby(["Model", "Metric"], dropna=False)["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["Axis"] = axis_label
        all_agg.append(agg)

    if not all_agg:
        return pd.DataFrame()

    combined = pd.concat(all_agg, ignore_index=True)

    # Determine per-column best (best model for each axis × metric)
    best_lookup: dict[tuple[str, str], float] = {}
    for _, row in combined.iterrows():
        key = (str(row["Axis"]), str(row["Metric"]))
        mean = float(row["Mean"])
        higher = next(
            (hib for mk, _, hib in _STRUCTURAL_TABLE_METRICS if mk == row["Metric"]),
            False,
        )
        # Skip bolding for sparsity_ratio (higher_is_better=False is overloaded)
        if row["Metric"] == "sparsity_ratio":
            continue
        current = best_lookup.get(key)
        if current is None:
            best_lookup[key] = mean
        elif higher and mean > current:
            best_lookup[key] = mean
        elif not higher and mean < current:
            best_lookup[key] = mean

    # ── Build LaTeX ────────────────────────────────────────────────────
    axes_list = [
        (ak, al)
        for ak, al in _SUMMARY_TABLE_AXES.items()
        if al in combined["Axis"].unique()
    ]
    n_axes = len(axes_list)
    n_sub = len(_STRUCTURAL_TABLE_METRICS)

    lines: list[str] = []
    lines.append(r"\footnotesize")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    col_spec = "l" + ("|" + "r" * n_sub) * n_axes
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header1 = r"\textbf{Model}"
    for _, axis_label in axes_list:
        header1 += rf" & \multicolumn{{{n_sub}}}{{c}}{{\textbf{{{axis_label}}}}}"
    header1 += r" \\"
    lines.append(header1)

    cmidrules = ""
    col_start = 2
    for _ in axes_list:
        col_end = col_start + n_sub - 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
        col_start = col_end + 1
    lines.append(cmidrules)

    header2 = ""
    for _ in axes_list:
        for _, mlabel, _ in _STRUCTURAL_TABLE_METRICS:
            header2 += rf" & \textbf{{{mlabel}}}"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    for model in models:
        row_str = model
        for _, axis_label in axes_list:
            for metric_key, _, higher_is_better in _STRUCTURAL_TABLE_METRICS:
                cell_row = combined[
                    (combined["Model"] == model)
                    & (combined["Axis"] == axis_label)
                    & (combined["Metric"] == metric_key)
                ]
                if cell_row.empty:
                    row_str += " & --"
                    continue
                mean = float(cell_row.iloc[0]["Mean"])
                sem = float(cell_row.iloc[0]["SEM"])
                cell = format_value(mean, sem)
                best_val = best_lookup.get((axis_label, metric_key))
                if (
                    best_val is not None
                    and model != "Random"
                    and abs(mean - best_val) < 1e-6
                ):
                    cell = _bold_if_best(cell, is_best=True)
                row_str += f" & {cell}"
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    output_path.write_text("\n".join(lines) + "\n")
    log.info("Saved structural summary table to %s", output_path)
    return combined


# ── RQ1: Non-uniform degradation bar chart ────────────────────────────


def generate_rq1_nonuniform_degradation_bar(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
    metric_name: str = "ne-sid",
) -> pd.DataFrame:
    """Generate a grouped bar chart showing per-family degradation ratios.

    For each OOD dataset family the degradation ratio
    ``mean(OOD metric) / mean(ID metric)`` is computed, where the ID baseline
    is anchored per shift axis.  Families are grouped by shift axis and bars
    are coloured by model, directly answering *"Does OOD performance degrade
    uniformly across shift axes?"* — the answer is visibly **no**.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output PDF figure.
        model_filter: Optional list of model display names to include.
        metric_name: Metric to use for the ratio (default ``"ne-sid"``).

    Returns:
        DataFrame with columns ``Model``, ``Family``, ``Axis``, ``Ratio``.
    """
    subset = raw_df[raw_df["Metric"].eq(metric_name)].copy()
    fixed_subset = subset[is_fixed_size_task_frame(subset)]

    if fixed_subset.empty:
        log.warning(
            "No %s data for non-uniform degradation bar; skipping.", metric_name
        )
        return pd.DataFrame()

    models = _resolve_models(model_filter)

    # Only fixed-size axes (graph, mechanism, noise, compound)
    _BAR_AXES: dict[str, str] = {
        "graph": "Graph",
        "mechanism": "Mechanism",
        "noise": "Noise",
        "compound": "Graph-Mechanism",
    }

    rows: list[dict[str, object]] = []
    for shift_key, shift_label in _BAR_AXES.items():
        id_data = _degradation_id_subset(fixed_subset, shift_key)
        id_means = id_data.groupby("Model")["Value"].mean().to_dict()

        ood_data = fixed_subset[fixed_subset["AxisCategory"] == shift_key]
        # Per-family breakdown
        for (model, dk), grp in ood_data.groupby(["Model", "DatasetKey"]):
            model_str = str(model)
            if model_str not in models:
                continue
            id_val = id_means.get(model_str)
            if id_val is None or id_val <= 0:
                continue
            ood_mean = float(grp["Value"].mean())
            ratio = ood_mean / id_val
            # Short family label
            family_label = thesis_dataset_label(
                str(dk),
                str(grp["Dataset"].iloc[0]) if "Dataset" in grp.columns else str(dk),
            )
            rows.append(
                {
                    "Model": model_str,
                    "Family": family_label,
                    "Axis": shift_label,
                    "Ratio": ratio,
                }
            )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        log.warning("No data for non-uniform degradation bar; skipping.")
        return pd.DataFrame()

    # Sort families within each axis by mean degradation (worst first)
    axis_order = list(_BAR_AXES.values())
    family_sort = (
        result_df.groupby(["Axis", "Family"])["Ratio"]
        .mean()
        .reset_index()
        .sort_values(["Axis", "Ratio"], ascending=[True, False])
    )

    # Build figure: one panel per axis
    n_axes = len(axis_order)
    fig, axes_arr = plt.subplots(1, n_axes, figsize=grid_figsize(n_axes), squeeze=False)

    for ax_idx, axis_label in enumerate(axis_order):
        ax = axes_arr[0, ax_idx]
        ax_data = result_df[result_df["Axis"] == axis_label]
        if ax_data.empty:
            ax.set_visible(False)
            continue

        # Family order: worst average degradation first
        fam_order = list(
            family_sort[family_sort["Axis"] == axis_label]["Family"].values
        )
        present_models = [m for m in models if m in ax_data["Model"].unique()]
        n_models = len(present_models)
        n_fam = len(fam_order)
        bar_width = 0.7 / max(n_models, 1)
        x_base = np.arange(n_fam)

        for m_idx, model in enumerate(present_models):
            model_data = ax_data[ax_data["Model"] == model]
            vals = []
            for fam in fam_order:
                fam_row = model_data[model_data["Family"] == fam]
                vals.append(
                    float(fam_row["Ratio"].iloc[0]) if not fam_row.empty else 0.0
                )
            offset = (m_idx - n_models / 2 + 0.5) * bar_width
            ax.bar(
                x_base + offset,
                vals,
                width=bar_width,
                color=_model_color(model),
                label=model if ax_idx == 0 else None,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_xticks(x_base)
        ax.set_xticklabels(fam_order, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{axis_label} Shift", fontsize=_PANEL_TITLE_FONTSIZE)
        ax.set_ylabel("OOD / ID Ratio" if ax_idx == 0 else "", fontsize=_LABEL_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Shared legend
    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    if handles:
        axes_arr[0, 0].legend(
            handles,
            labels,
            loc="best",
            ncol=_model_legend_ncol(len(labels)),
            fontsize=_model_legend_fontsize(len(labels)),
            frameon=True,
            framealpha=0.8,
        )

    fig.suptitle(
        "Per-Family Degradation Ratio by Shift Axis",
        fontsize=_TITLE_FONTSIZE,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved non-uniform degradation bar chart to %s", output_path)
    save_figure_data(output_path, result_df)
    return result_df


# ── RQ1: AviCi-only DAG validity figure ──────────────────────────────


def generate_rq1_avici_dag_validity_figure(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    """Generate an AviCi-only valid DAG figure with sampled vs. thresholded markers.

    Unlike the all-model :func:`generate_valid_dag_shift_figure`, this figure:

    * Shows **only AviCi** (no DiBS, no BCNP/BayesDAG reference lines).
    * Uses a **three-panel layout** (Graph, Mechanism, Noise+Compound) to keep
      the figure compact.
    * **Includes all OOD mechanism families** in the graph panel, not just
      linear—this is important because the linear case is where AviCi
      struggles most, and showing all families reveals the mechanism-dependent
      variation in DAG validity.
    * Distinguishes sampled (solid) vs. thresholded (open) DAG rates per the
      methodology.

    Does **not** require DiBS posterior artifacts (no run_dirs parameter).

    Args:
        raw_df: Long-format raw task DataFrame (must contain ``valid_dag_pct``
            and optionally ``threshold_valid_dag_pct``).
        output_path: Path for the output PDF figure.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    _METRICS = ["valid_dag_pct", "threshold_valid_dag_pct"]
    vdp = raw_df[
        (raw_df["Metric"].isin(_METRICS)) & (raw_df["Model"] == "AviCi")
    ].copy()

    if vdp.empty:
        log.warning("No AviCi valid_dag_pct data; skipping RQ1 DAG validity figure.")
        return pd.DataFrame()

    # Metric display config
    _METRIC_STYLE: dict[str, tuple[str, float, int]] = {
        "valid_dag_pct": ("Sampled", 0.9, 7),
        "threshold_valid_dag_pct": ("Thresholded", 0.45, 6),
    }

    # Panel specs: (shift_key, axis_categories, title, filter_fn)
    # Graph panel: show ALL mechanism families (ID + OOD-graph), not just linear
    # Mechanism panel: show ER-20 anchor
    # Combined panel: noise + compound together for compactness

    _PANELS: list[tuple[str, list[str], str]] = [
        ("graph", ["id", "graph"], "Graph Target Environments"),
        ("mechanism", ["id", "mechanism"], "Mechanism Target Environments (ER20)"),
        (
            "combined",
            ["id", "noise", "compound"],
            "Noise + Graph-Mechanism Target Environments",
        ),
    ]

    n_panels = len(_PANELS)
    fig, axes_arr = plt.subplots(
        1, n_panels, figsize=grid_figsize(n_panels), squeeze=False
    )

    all_agg: list[pd.DataFrame] = []
    avici_color = _model_color("AviCi")

    for panel_idx, (panel_key, cats, panel_title) in enumerate(_PANELS):
        ax = axes_arr[0, panel_idx]
        panel_df = vdp[vdp["AxisCategory"].isin(cats)].copy()

        # Apply fixed-size filter
        panel_df = panel_df[is_fixed_size_task_frame(panel_df)]

        # Panel-specific filtering
        if panel_key == "mechanism":
            panel_df = _restrict_to_graph_anchor(panel_df, "er20")

        if panel_df.empty:
            ax.set_visible(False)
            continue

        # Aggregate per (DatasetKey, Metric)
        agg = (
            panel_df.groupby(
                ["DatasetKey", "Dataset", "AxisCategory", "Metric"],
                dropna=False,
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )

        # Labels
        if panel_key == "graph":
            agg["DatasetLabel"] = agg["DatasetKey"].map(_graph_shift_label)
        elif panel_key == "mechanism":
            agg["DatasetLabel"] = agg["DatasetKey"].map(_mech_shift_label)
        else:
            agg["DatasetLabel"] = agg.apply(
                lambda row: thesis_dataset_label(
                    str(row["DatasetKey"]), str(row["Dataset"])
                ),
                axis=1,
            )

        agg["_sort"] = agg["AxisCategory"].map({"id": 0}).fillna(1)
        agg = agg.sort_values(["_sort", "DatasetLabel"]).drop(columns=["_sort"])
        all_agg.append(agg)

        # x-axis from sample-level metric
        agg_sample = agg[agg["Metric"] == "valid_dag_pct"]
        datasets = list(agg_sample["DatasetLabel"].unique())
        axis_lookup = (
            agg_sample[["DatasetLabel", "AxisCategory"]]
            .drop_duplicates()
            .set_index("DatasetLabel")["AxisCategory"]
            .to_dict()
        )
        n_ds = len(datasets)
        x_base = np.arange(n_ds)

        for metric_key, (label_sfx, alpha, msize) in _METRIC_STYLE.items():
            metric_agg = agg[agg["Metric"] == metric_key]
            if metric_agg.empty:
                continue

            xs, means, sems = [], [], []
            for i, ds in enumerate(datasets):
                row = metric_agg[metric_agg["DatasetLabel"] == ds]
                if row.empty:
                    continue
                xs.append(float(x_base[i]))
                means.append(float(row.iloc[0]["Mean"]))
                sems.append(float(row.iloc[0]["SEM"]))

            if xs:
                marker = "o"
                fillstyle = "full" if metric_key == "valid_dag_pct" else "none"
                ax.errorbar(
                    xs,
                    means,
                    yerr=sems,
                    fmt=marker,
                    label=label_sfx,
                    color=avici_color,
                    capsize=3,
                    markersize=msize,
                    alpha=alpha,
                    fillstyle=fillstyle,
                )

        ax.set_xticks(x_base)
        ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(_dag_ylabel(), fontsize=_LABEL_FONTSIZE)
        ax.set_ylim(-5, 110)
        ax.set_title(panel_title, fontsize=_PANEL_TITLE_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Grey ID region
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    # Shared legend
    handles, labels = [], []
    seen: set[str] = set()
    for ax in axes_arr.flatten():
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    if handles:
        axes_arr[0, 0].legend(
            handles,
            labels,
            loc="best",
            ncol=len(labels),
            fontsize=_LEGEND_FONTSIZE,
            frameon=True,
            framealpha=0.8,
        )

    fig.suptitle(
        "AviCi DAG Validity Under Distribution Shift",
        fontsize=_TITLE_FONTSIZE,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    combined = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    log.info("Saved RQ1 AviCi DAG validity figure to %s", output_path)
    save_figure_data(output_path, combined)
    return combined
