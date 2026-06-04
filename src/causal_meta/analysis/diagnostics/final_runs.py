from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from causal_meta.analysis.common.plot_style import (
    LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    SUBTITLE_FONTSIZE,
    TITLE_FONTSIZE,
    TOP_LEGEND_ANCHOR,
    TOP_LEGEND_RECT,
)
from causal_meta.analysis.common.thesis import (
    graph_code_of,
    id_mechanism_of,
    thesis_dataset_label,
)
from causal_meta.analysis.utils import (
    MODEL_COLORS,
    MODEL_MARKERS,
    map_dataset_description,
    save_figure_data,
)

log = logging.getLogger(__name__)

_TITLE_FONTSIZE = TITLE_FONTSIZE
_PANEL_TITLE_FONTSIZE = SUBTITLE_FONTSIZE
_LABEL_FONTSIZE = LABEL_FONTSIZE
_LEGEND_FONTSIZE = LEGEND_FONTSIZE


CORE_MODELS: tuple[str, ...] = ("AviCi", "BCNP", "DiBS", "BayesDAG")
AMORTISED_MODELS: tuple[str, ...] = ("AviCi", "BCNP")
EXPLICIT_MODELS: tuple[str, ...] = ("DiBS", "BayesDAG")

GRAPH_ORDER: tuple[str, ...] = (
    "er20",
    "er40",
    "er60",
    "sf1",
    "sf2",
    "sf3",
    "sbm",
    "ws",
    "grg",
)
GRAPH_LABELS: dict[str, str] = {
    "er20": "ER-20",
    "er40": "ER-40",
    "er60": "ER-60",
    "sf1": "SF-1",
    "sf2": "SF-2",
    "sf3": "SF-3",
    "sbm": "SBM",
    "ws": "WS",
    "grg": "GRG",
}

MECH_ORDER: tuple[str, ...] = (
    "linear",
    "neuralnet",
    "gpcde",
    "periodic",
    "square",
    "pnl_tanh",
    "logistic_map",
)
MECH_LABELS: dict[str, str] = {
    "linear": "Linear",
    "neuralnet": "MLP",
    "gpcde": "RFF",
    "periodic": "Periodic",
    "square": "Square",
    "pnl_tanh": "PNL",
    "logistic_map": "Logistic",
}

AXIS_ORDER: tuple[str, ...] = (
    "id",
    "graph",
    "mechanism",
    "noise",
    "compound",
    "nodes",
    "samples",
)
AXIS_LABELS: dict[str, str] = {
    "id": "ID",
    "graph": "Graph",
    "mechanism": "Mechanism",
    "noise": "Noise",
    "compound": "Compound",
    "nodes": "Nodes",
    "samples": "Samples",
}
AXIS_COLORS: dict[str, str] = {
    "id": "#4daf4a",
    "graph": "#377eb8",
    "mechanism": "#984ea3",
    "noise": "#ff7f00",
    "compound": "#e41a1c",
    "nodes": "#a65628",
    "samples": "#999999",
}


def _core_metric_frame(
    raw_df: pd.DataFrame,
    *,
    metric: str,
    fixed_size_only: bool,
) -> pd.DataFrame:
    """Return per-family metric means for the four core models."""
    subset = raw_df[
        (raw_df["Metric"] == metric) & (raw_df["Model"].isin(CORE_MODELS))
    ].copy()
    if fixed_size_only:
        subset = subset[subset["NNodes"].eq(20) & subset["SamplesPerTask"].eq(500)]

    if subset.empty:
        return pd.DataFrame()

    agg = (
        subset.groupby(
            ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
        )["Value"]
        .mean()
        .reset_index(name="MetricMean")
    )
    agg["DatasetLabel"] = agg.apply(
        lambda row: thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"])),
        axis=1,
    )
    agg["DatasetLongLabel"] = agg["DatasetKey"].map(map_dataset_description)
    agg["GraphCode"] = agg["DatasetKey"].map(graph_code_of)
    agg["MechanismCode"] = agg["DatasetKey"].map(_mechanism_code_of)
    return agg


def _mechanism_code_of(dataset_key: str) -> str | None:
    """Extract a canonical mechanism code from a dataset key."""
    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)

    if body.startswith("ood_mech_"):
        remainder = body[len("ood_mech_") :]
        graph_code = graph_code_of(dataset_key)
        if graph_code is not None and remainder.endswith(f"_{graph_code}"):
            remainder = remainder[: -(len(graph_code) + 1)]
        return remainder

    if body.startswith("ood_both_"):
        remainder = body[len("ood_both_") :]
        graph_code = graph_code_of(dataset_key)
        if graph_code is not None and remainder.startswith(f"{graph_code}_"):
            remainder = remainder[len(graph_code) + 1 :]
        return remainder

    mech = id_mechanism_of(dataset_key)
    if mech is not None:
        return mech
    return None


def _winner_summary(raw_df: pd.DataFrame, *, fixed_size_only: bool) -> pd.DataFrame:
    """Compute family-level winners on ne-SID across core models."""
    family_df = _core_metric_frame(
        raw_df,
        metric="ne-sid",
        fixed_size_only=fixed_size_only,
    )
    if family_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (
        dataset_key,
        axis_cat,
        dataset_label,
        graph_code,
        mech_code,
    ), group in family_df.groupby(
        [
            "DatasetKey",
            "AxisCategory",
            "DatasetLabel",
            "GraphCode",
            "MechanismCode",
        ],
        dropna=False,
    ):
        ranked = group.sort_values("MetricMean", ascending=True)
        if ranked.empty:
            continue
        winner_model = str(ranked.iloc[0]["Model"])
        winner_value = float(ranked.iloc[0]["MetricMean"])
        if len(ranked) >= 2:
            second_value = float(ranked.iloc[1]["MetricMean"])
            margin = second_value - winner_value
        else:
            second_value = float("nan")
            margin = float("nan")
        rows.append(
            {
                "DatasetKey": str(dataset_key),
                "DatasetLabel": str(dataset_label),
                "AxisCategory": str(axis_cat),
                "GraphCode": graph_code,
                "MechanismCode": mech_code,
                "WinnerModel": winner_model,
                "WinnerValue": winner_value,
                "SecondValue": second_value,
                "Margin": margin,
            }
        )

    return pd.DataFrame(rows)


def _best_model_label(row: pd.Series, models: Sequence[str]) -> str | None:
    """Return the model with the minimum available value in *row*."""
    values: dict[str, float] = {}
    for model in models:
        val = row.get(model)
        if pd.notna(val):
            values[model] = float(val)
    if not values:
        return None
    return min(values, key=values.get)


def _explicit_advantage_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return family-level explicit-vs-amortized advantage data."""
    family_df = _core_metric_frame(
        raw_df,
        metric="ne-sid",
        fixed_size_only=False,
    )
    if family_df.empty:
        return pd.DataFrame()

    wide = family_df.pivot_table(
        index=["DatasetKey", "DatasetLabel", "DatasetLongLabel", "AxisCategory"],
        columns="Model",
        values="MetricMean",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    for model in CORE_MODELS:
        if model not in wide.columns:
            wide[model] = np.nan

    wide["BestAmortized"] = wide[list(AMORTISED_MODELS)].min(axis=1)
    wide["BestExplicit"] = wide[list(EXPLICIT_MODELS)].min(axis=1)
    wide["ExplicitAdvantage"] = wide["BestExplicit"] - wide["BestAmortized"]
    wide["BestAmortizedModel"] = wide.apply(
        lambda row: _best_model_label(row, AMORTISED_MODELS),
        axis=1,
    )
    wide["BestExplicitModel"] = wide.apply(
        lambda row: _best_model_label(row, EXPLICIT_MODELS),
        axis=1,
    )

    dist = raw_df[
        ["DatasetKey", "SpectralDist", "KLDegreeDist", "MechanismDist"]
    ].drop_duplicates(subset=["DatasetKey"])
    for col in ("SpectralDist", "KLDegreeDist", "MechanismDist"):
        dist[col] = pd.to_numeric(dist[col], errors="coerce").fillna(0.0)
    dist["CombinedDistance"] = np.sqrt(
        dist["SpectralDist"] ** 2
        + dist["KLDegreeDist"] ** 2
        + dist["MechanismDist"] ** 2
    )

    merged = wide.merge(dist, on="DatasetKey", how="left")
    return merged


def generate_winner_matrix_figure(
    raw_df: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    """Plot the best-performing model for each fixed-size graph/mechanism family."""
    winners = _winner_summary(raw_df, fixed_size_only=True)
    if winners.empty:
        log.warning(
            "No fixed-size winner data available; skipping winner matrix figure."
        )
        return pd.DataFrame()

    matrix = pd.DataFrame(np.nan, index=MECH_ORDER, columns=GRAPH_ORDER)
    winner_names = pd.DataFrame("", index=MECH_ORDER, columns=GRAPH_ORDER)

    for _, row in winners.iterrows():
        graph_code = row["GraphCode"]
        mech_code = row["MechanismCode"]
        if graph_code not in matrix.columns or mech_code not in matrix.index:
            continue
        winner_model = str(row["WinnerModel"])
        if winner_model not in CORE_MODELS:
            continue
        matrix.loc[mech_code, graph_code] = CORE_MODELS.index(winner_model)
        winner_names.loc[mech_code, graph_code] = winner_model

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    cmap = ListedColormap([MODEL_COLORS[m] for m in CORE_MODELS])
    masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))

    ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(CORE_MODELS) - 0.5)
    ax.set_facecolor("#f5f5f5")

    ax.set_xticks(np.arange(len(GRAPH_ORDER)))
    ax.set_xticklabels([GRAPH_LABELS[g] for g in GRAPH_ORDER], fontsize=8)
    ax.set_yticks(np.arange(len(MECH_ORDER)))
    ax.set_yticklabels([MECH_LABELS[m] for m in MECH_ORDER], fontsize=8)
    ax.set_xlabel("Graph family", fontsize=9)
    ax.set_ylabel("Mechanism family", fontsize=9)

    short = {"AviCi": "A", "BCNP": "B", "DiBS": "D", "BayesDAG": "BD"}
    for i, mech in enumerate(MECH_ORDER):
        for j, graph in enumerate(GRAPH_ORDER):
            if pd.isna(matrix.iloc[i, j]):
                continue
            model = winner_names.iloc[i, j]
            text_color = "white" if model in {"DiBS", "BayesDAG"} else "black"
            ax.text(
                j,
                i,
                short.get(model, model),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=text_color,
            )

    handles = [Patch(facecolor=MODEL_COLORS[m], label=m) for m in CORE_MODELS]
    handles.append(Patch(facecolor="#f5f5f5", edgecolor="#bbbbbb", label="No family"))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=TOP_LEGEND_ANCHOR,
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )

    ax.set_title(
        "Family-Level Winner Map (ne-SID, fixed-size families)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout(rect=TOP_LEGEND_RECT)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, winners)
    log.info("Saved winner matrix figure to %s", output_path)
    return winners


def generate_explicit_advantage_distance_figure(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Plot explicit-vs-amortized advantage against shift distance."""
    adv = _explicit_advantage_frame(raw_df)
    if adv.empty:
        log.warning("No explicit-advantage data; skipping distance figure.")
        return pd.DataFrame()

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for axis in AXIS_ORDER:
        axis_df = adv[adv["AxisCategory"] == axis]
        if axis_df.empty:
            continue
        ax.scatter(
            axis_df["CombinedDistance"],
            axis_df["ExplicitAdvantage"],
            s=60,
            alpha=0.85,
            color=AXIS_COLORS[axis],
            edgecolors="white",
            linewidths=0.5,
            label=AXIS_LABELS[axis],
        )

    reg_df = adv.dropna(subset=["CombinedDistance", "ExplicitAdvantage"])
    if len(reg_df) >= 3:
        x = reg_df["CombinedDistance"].to_numpy(dtype=float)
        y = reg_df["ExplicitAdvantage"].to_numpy(dtype=float)
        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(
            x_line,
            np.polyval(coef, x_line),
            color="#222222",
            linestyle="--",
            linewidth=1.4,
            label="Trend",
        )

    ax.axhline(0.0, color="#555555", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Combined distributional distance", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel(
        "Best explicit - best amortized (ne-SID)",
        fontsize=_LABEL_FONTSIZE,
    )
    ax.set_title(
        "Where Explicit Inference Overtakes Amortized Inference",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=_LEGEND_FONTSIZE, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, adv)
    log.info("Saved explicit-advantage distance figure to %s", output_path)
    return adv


def generate_explicit_advantage_table(
    raw_df: pd.DataFrame,
    output_path: Path,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Create a LaTeX table of families where explicit methods are better."""
    adv = _explicit_advantage_frame(raw_df)
    if adv.empty:
        log.warning("No explicit-advantage data; skipping explicit-advantage table.")
        return pd.DataFrame()

    better = adv[adv["ExplicitAdvantage"] < 0].copy()
    better = better.sort_values("ExplicitAdvantage", ascending=True).head(top_k)
    if better.empty:
        log.warning("No families with explicit advantage found.")
        return pd.DataFrame()

    lines = [
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Axis} & \textbf{Best amortized} & \textbf{Best explicit} & \textbf{$\Delta$ (exp-amort)} & \textbf{Distance} \\",
        r"\midrule",
    ]

    # Best per numeric column for bold highlighting.
    best_amort = float(better["BestAmortized"].min())
    best_explicit = float(better["BestExplicit"].min())
    best_delta = float(better["ExplicitAdvantage"].min())

    def _bf(val: float, ref: float, fmt: str) -> str:
        s = f"{val:{fmt}}"
        return r"\textbf{" + s + "}" if abs(val - ref) < 1e-9 else s

    for _, row in better.iterrows():
        family = str(row["DatasetLongLabel"]).replace("_", r"\_")
        axis = AXIS_LABELS.get(str(row["AxisCategory"]), str(row["AxisCategory"]))
        lines.append(
            f"{family} & {axis} & {_bf(row['BestAmortized'], best_amort, '.3f')}"
            f" & {_bf(row['BestExplicit'], best_explicit, '.3f')}"
            f" & {_bf(row['ExplicitAdvantage'], best_delta, '.3f')}"
            f" & {row['CombinedDistance']:.2f} \\\\",
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    output_path.write_text("\n".join(lines) + "\n")
    log.info("Saved explicit-advantage table to %s", output_path)
    return better


def generate_runtime_pareto_figure(
    raw_df: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    """Plot runtime vs. ne-SID for all axes and core models."""
    needed = {"ne-sid", "inference_time_s"}
    subset = raw_df[
        (raw_df["Metric"].isin(needed)) & (raw_df["Model"].isin(CORE_MODELS))
    ].copy()
    if subset.empty:
        log.warning("No runtime/accuracy data; skipping runtime Pareto figure.")
        return pd.DataFrame()

    agg = (
        subset.groupby(["Model", "AxisCategory", "Metric"], dropna=False)["Value"]
        .mean()
        .unstack("Metric")
        .reset_index()
    )
    if "ne-sid" not in agg.columns or "inference_time_s" not in agg.columns:
        log.warning("Missing ne-sid or inference_time_s after aggregation.")
        return pd.DataFrame()

    marker_map: dict[str, str] = {
        "id": "o",
        "graph": "s",
        "mechanism": "^",
        "noise": "D",
        "compound": "P",
        "nodes": "X",
        "samples": "v",
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for model in CORE_MODELS:
        model_df = agg[agg["Model"] == model].copy()
        if model_df.empty:
            continue
        model_df = model_df.sort_values("inference_time_s")
        ax.plot(
            model_df["inference_time_s"],
            model_df["ne-sid"],
            color=MODEL_COLORS[model],
            linewidth=1.2,
            alpha=0.6,
        )
        for _, row in model_df.iterrows():
            axis = str(row["AxisCategory"])
            ax.scatter(
                row["inference_time_s"],
                row["ne-sid"],
                color=MODEL_COLORS[model],
                marker=marker_map.get(axis, "o"),
                s=70,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.6,
            )

    ax.set_xscale("log")
    ax.set_xlabel(
        "Inference time per dataset (seconds, log scale)", fontsize=_LABEL_FONTSIZE
    )
    ax.set_ylabel("Mean ne-SID", fontsize=_LABEL_FONTSIZE)
    ax.set_title(
        "Runtime-Accuracy Pareto Profile Across Shift Axes",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.35)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[m],
            marker=MODEL_MARKERS.get(m, "o"),
            linestyle="-",
            label=m,
        )
        for m in CORE_MODELS
    ]
    axis_handles = [
        Line2D(
            [0],
            [0],
            color="#666666",
            marker=marker_map[a],
            linestyle="None",
            markersize=7,
            label=AXIS_LABELS[a],
        )
        for a in AXIS_ORDER
    ]

    legend_models = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        fontsize=9,
        frameon=False,
    )
    ax.add_artist(legend_models)
    ax.legend(
        handles=axis_handles,
        title="Axis",
        loc="lower right",
        fontsize=8,
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, agg)
    log.info("Saved runtime Pareto figure to %s", output_path)
    return agg


def _uncertainty_alignment_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute MSE alignment of uncertainty scores with ne-SID.

    Both uncertainty scores and ne-SID are min-max normalised globally
    before computing MSE so that values are comparable across models and
    shift axes.  Lower MSE indicates tighter tracking of structural error
    by the uncertainty signal.
    """
    needed = {"ne-sid", "edge_entropy", "graph_nll_per_edge"}
    subset = raw_df[
        (raw_df["Metric"].isin(needed)) & (raw_df["Model"].isin(CORE_MODELS))
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    family = (
        subset.groupby(["Model", "DatasetKey", "AxisCategory", "Metric"], dropna=False)[
            "Value"
        ]
        .mean()
        .unstack("Metric")
        .reset_index()
    )

    # Min-max normalise each metric globally for comparable MSE values.
    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else s * 0.0

    family["ne_sid_n"] = _minmax(family["ne-sid"])
    family["edge_entropy_n"] = _minmax(family["edge_entropy"])
    family["graph_nll_n"] = _minmax(family["graph_nll_per_edge"])

    rows: list[dict[str, object]] = []
    for model in CORE_MODELS:
        model_df = family[family["Model"] == model]
        for axis in AXIS_ORDER:
            axis_df = model_df[model_df["AxisCategory"] == axis]
            n_families = int(len(axis_df))
            if n_families >= 3:
                mse_entropy = float(
                    np.mean((axis_df["ne_sid_n"] - axis_df["edge_entropy_n"]) ** 2)
                )
                mse_nll = float(
                    np.mean((axis_df["ne_sid_n"] - axis_df["graph_nll_n"]) ** 2)
                )
            else:
                mse_entropy = np.nan
                mse_nll = np.nan
            rows.append(
                {
                    "Model": model,
                    "AxisCategory": axis,
                    "NFamilies": n_families,
                    "EdgeEntropyMSE": mse_entropy,
                    "GraphNllMSE": mse_nll,
                }
            )

    return pd.DataFrame(rows)


def generate_uncertainty_alignment_heatmap(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Plot per-axis uncertainty/error alignment for entropy and GraphNLL."""
    align = _uncertainty_alignment_frame(raw_df)
    if align.empty:
        log.warning("No uncertainty-alignment data; skipping heatmap.")
        return pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), squeeze=False)
    score_specs = [
        ("EdgeEntropyMSE", r"MSE(ne-SID, Edge Entropy)"),
        ("GraphNllMSE", r"MSE(ne-SID, GraphNLL / edge)"),
    ]

    # Determine shared colour range from data (MSE ≥ 0, lower is better).
    all_vals = pd.concat([align["EdgeEntropyMSE"], align["GraphNllMSE"]]).dropna()
    vmax = float(np.ceil(all_vals.max() * 20) / 20)  # round up to nearest 0.05
    vmax = max(vmax, 0.10)  # ensure a minimum visible range

    for col_idx, (score_col, title) in enumerate(score_specs):
        ax = axes[0, col_idx]
        mat = align.pivot(
            index="Model", columns="AxisCategory", values=score_col
        ).reindex(index=list(CORE_MODELS), columns=list(AXIS_ORDER))

        im = ax.imshow(
            mat.to_numpy(dtype=float),
            aspect="auto",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xticks(np.arange(len(AXIS_ORDER)))
        ax.set_xticklabels(
            [AXIS_LABELS[a] for a in AXIS_ORDER], rotation=30, ha="right", fontsize=9
        )
        ax.set_yticks(np.arange(len(CORE_MODELS)))
        ax.set_yticklabels(list(CORE_MODELS), fontsize=10)
        ax.set_title(title, fontsize=11)

        for i, model in enumerate(CORE_MODELS):
            for j, axis in enumerate(AXIS_ORDER):
                val = mat.loc[model, axis]
                if pd.isna(val):
                    text = "--"
                    color = "#333333"
                else:
                    text = f"{float(val):.3f}"
                    color = "white" if float(val) > vmax * 0.65 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        # Annotate colorbar direction: lower MSE = better alignment.
        cbar.ax.annotate(
            "better",
            xy=(0.5, 0.0),
            xycoords="axes fraction",
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7,
            fontstyle="italic",
            color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
        )
        cbar.ax.annotate(
            "worse",
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            fontstyle="italic",
            color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
        )

    fig.suptitle(
        "Uncertainty-Error Alignment by Model and Shift Axis",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=TOP_LEGEND_RECT)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, align)
    log.info("Saved uncertainty alignment heatmap to %s", output_path)
    return align


def generate_final_run_insight_artifacts(
    raw_df: pd.DataFrame, output_dir: Path
) -> None:
    """Generate additional deep-dive artifacts for the four core final runs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_winner_matrix_figure(raw_df, output_dir / "winner_matrix.pdf")

    generate_explicit_advantage_distance_figure(
        raw_df,
        output_dir / "explicit_advantage_distance.pdf",
    )
    generate_explicit_advantage_table(raw_df, output_dir / "explicit_advantage.tex")

    generate_runtime_pareto_figure(raw_df, output_dir / "runtime_pareto.pdf")

    generate_uncertainty_alignment_heatmap(
        raw_df,
        output_dir / "uncertainty_alignment_heatmap.pdf",
    )
