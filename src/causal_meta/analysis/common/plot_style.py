from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from causal_meta.analysis.utils import ERROR_COLORS, MODEL_COLORS, MODEL_MARKERS

# ── Font sizes ────────────────────────────────────────────────────────

TITLE_FONTSIZE = 14
SUBTITLE_FONTSIZE = 12
LABEL_FONTSIZE = 10
TICK_FONTSIZE = 8
LEGEND_FONTSIZE = 8
ANNOTATION_FONTSIZE = 7
TABLE_CELL_FONTSIZE = 6

# ── Figure sizes ──────────────────────────────────────────────────────

SINGLE_FIGSIZE = (4.8, 3.2)
COMPACT_SINGLE_FIGSIZE = (4.2, 2.8)
HEATMAP_FIGSIZE = (4.4, 2.6)
TWO_PANEL_FIGSIZE = (7.4, 3.6)
WIDE_TWO_ROW_FIGSIZE = (8.0, 6.4)
FOUR_PANEL_FIGSIZE = (10.0, 3.6)
PANEL_WIDTH = 3.8
PANEL_HEIGHT = 3.1
ROW_HEIGHT = 2.1

# ── Layout constants ──────────────────────────────────────────────────

TOP_LEGEND_ANCHOR = (0.5, 0.985)
TOP_LEGEND_RECT = [0, 0, 1, 0.96]
BOTTOM_LEGEND_ANCHOR = (0.5, -0.10)
DEFAULT_SUPTITLE_Y = 1.0

# ── Metric labels ─────────────────────────────────────────────────────

HAT_E = r"\widehat{\mathbb{E}}"
METRIC_LABELS: dict[str, str] = {
    "e-sid": rf"${HAT_E}$-SID",
    "ne-sid": rf"Normalized ${HAT_E}$-SID",
    "e-shd": rf"${HAT_E}$-SHD",
    "ne-shd": rf"Normalized ${HAT_E}$-SHD",
    "e-edgef1": rf"${HAT_E}$-Edge F1",
    "edge_entropy": "Edge Entropy",
    "graph_nll_per_edge": "Graph NLL / edge",
}


def metric_label(metric_name: str, *, down: bool | None = None) -> str:
    label = METRIC_LABELS.get(metric_name, metric_name)
    if down is True:
        return rf"{label} $\downarrow$"
    if down is False:
        return rf"{label} $\uparrow$"
    return label


def apply_thesis_plot_style() -> None:
    """Apply the thesis-wide Matplotlib style defaults."""
    plt.rcParams.update(
        {
            "axes.titlesize": SUBTITLE_FONTSIZE,
            "axes.titleweight": "normal",
            "axes.labelsize": LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "legend.title_fontsize": LEGEND_FONTSIZE,
            "figure.titlesize": TITLE_FONTSIZE,
            "figure.titleweight": "bold",
        }
    )


def top_legend_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "loc": "upper center",
        "bbox_to_anchor": TOP_LEGEND_ANCHOR,
        "fontsize": LEGEND_FONTSIZE,
        "frameon": False,
    }
    kwargs.update(overrides)
    return kwargs


def grid_figsize(
    n_cols: int,
    n_rows: int = 1,
    *,
    panel_width: float = PANEL_WIDTH,
    panel_height: float = PANEL_HEIGHT,
) -> tuple[float, float]:
    """Return a consistent size for panel grids."""
    return (panel_width * max(n_cols, 1), panel_height * max(n_rows, 1))


def row_figsize(n_cols: int, row_heights: list[float]) -> tuple[float, float]:
    """Return a consistent size for multi-row figures with custom height ratios."""
    return (PANEL_WIDTH * max(n_cols, 1), ROW_HEIGHT * sum(row_heights))


__all__ = [
    "ANNOTATION_FONTSIZE",
    "BOTTOM_LEGEND_ANCHOR",
    "COMPACT_SINGLE_FIGSIZE",
    "ERROR_COLORS",
    "FOUR_PANEL_FIGSIZE",
    "HAT_E",
    "HEATMAP_FIGSIZE",
    "LABEL_FONTSIZE",
    "LEGEND_FONTSIZE",
    "METRIC_LABELS",
    "MODEL_COLORS",
    "MODEL_MARKERS",
    "PANEL_HEIGHT",
    "PANEL_WIDTH",
    "ROW_HEIGHT",
    "SINGLE_FIGSIZE",
    "SUBTITLE_FONTSIZE",
    "TABLE_CELL_FONTSIZE",
    "TICK_FONTSIZE",
    "TITLE_FONTSIZE",
    "TOP_LEGEND_ANCHOR",
    "TOP_LEGEND_RECT",
    "TWO_PANEL_FIGSIZE",
    "WIDE_TWO_ROW_FIGSIZE",
    "apply_thesis_plot_style",
    "grid_figsize",
    "metric_label",
    "row_figsize",
    "top_legend_kwargs",
]
