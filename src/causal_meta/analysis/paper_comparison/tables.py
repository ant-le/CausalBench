from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from causal_meta.analysis.paper_comparison.reference_data import (
    AVICI_VARIANTS,
    COMPARABLE_FAMILIES,
    METRIC_DIRECTIONS,
    METRIC_LABELS,
    MODELS,
)
from causal_meta.analysis.utils import PAPER_MODEL_LABELS

log = logging.getLogger(__name__)

# ── LaTeX helpers ──────────────────────────────────────────────────────


def _bold(text: str) -> str:
    return rf"\textbf{{{text}}}"


def _fmt(val: float | None, digits: int = 2) -> str:
    if val is None:
        return "---"
    return f"{val:.{digits}f}"


def _best_idx(values: list[float | None], direction: str) -> int | None:
    """Return the index of the best non-None value, or None."""
    filtered = [(i, v) for i, v in enumerate(values) if v is not None]
    if not filtered:
        return None
    if direction == "lower_is_better":
        return min(filtered, key=lambda x: x[1])[0]
    return max(filtered, key=lambda x: x[1])[0]


# ── Cross-model performance tables (one per metric) ──────────────────


def generate_cross_model_table(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Write a single LaTeX table comparing Paper vs Ours for one metric.

    Rows: 9 families grouped by mechanism.
    Columns: per model → (Paper, Ours).
    """
    metric_label = METRIC_LABELS.get(metric, metric)
    direction = METRIC_DIRECTIONS.get(metric, "lower_is_better")
    metric_df = df[df["metric"] == metric].copy()
    is_shd = metric == "e_shd"
    digits = 1 if is_shd else 2

    # Column headers: one pair per model.
    model_labels = [PAPER_MODEL_LABELS.get(m, m) for m in MODELS]
    n_models = len(MODELS)

    lines: list[str] = []
    lines.append(r"\small")

    arrow = r"$\downarrow$" if direction == "lower_is_better" else r"$\uparrow$"
    lines.append(r"\resizebox{\textwidth}{!}{%")

    col_spec = "l" + "".join(
        f"rr{'|' if i < n_models - 1 else ''}" for i in range(n_models)
    )
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row 1: model names spanning two columns each.
    header1_parts = [r"\multirow{2}{*}{\textbf{Setting}}"]
    for label in model_labels:
        header1_parts.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{{label}}}}}")
    lines.append(" & ".join(header1_parts) + r" \\")

    # Header row 2: Paper / Ours under each model.
    header2_parts = [""]
    for _ in MODELS:
        header2_parts.append(r"\textit{Paper}")
        header2_parts.append(r"\textit{Ours}")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    # Body rows, grouped by mechanism.
    prev_mechanism = ""
    for fam_key, fam_label in COMPARABLE_FAMILIES.items():
        mechanism = fam_label.split()[0]  # "Linear", "MLP", "GP"
        if mechanism != prev_mechanism and prev_mechanism:
            lines.append(r"\addlinespace")
        prev_mechanism = mechanism

        cells = [fam_label]
        paper_vals: list[float | None] = []
        thesis_vals: list[float | None] = []

        for model in MODELS:
            row = metric_df[
                (metric_df["model"] == model) & (metric_df["family"] == fam_key)
            ]
            pv = row["paper_value"].values[0] if len(row) else None
            tv = row["thesis_value"].values[0] if len(row) else None
            paper_vals.append(pv)
            thesis_vals.append(tv)
            cells.append(_fmt(pv, digits))
            cells.append(_fmt(tv, digits))

        # Bold the best paper value and best thesis value separately.
        best_p = _best_idx(paper_vals, direction)
        best_t = _best_idx(thesis_vals, direction)
        if best_p is not None:
            idx = 1 + best_p * 2  # paper column index in cells
            cells[idx] = _bold(cells[idx])
        if best_t is not None:
            idx = 2 + best_t * 2  # thesis column index in cells
            cells[idx] = _bold(cells[idx])

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote cross-model table (%s): %s", metric, output_path)


# ── Hyperparameter comparison tables (one per model) ──────────────────


def generate_hyperparam_table(
    hp_df: pd.DataFrame,
    model: str,
    output_path: Path,
) -> None:
    """Write a LaTeX table comparing paper vs our hyperparameters for one model."""
    model_label = PAPER_MODEL_LABELS.get(model, model)

    lines: list[str] = []
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lll}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Parameter} & \textbf{Paper} & \textbf{This Thesis} \\")
    lines.append(r"\midrule")

    for _, row in hp_df.iterrows():
        param = str(row["parameter"]).replace("_", r"\_")
        paper_val = str(row["paper_value"]).replace("_", r"\_")
        our_val = str(row["our_value"]).replace("_", r"\_")
        lines.append(f"{param} & {paper_val} & {our_val} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote hyperparam table (%s): %s", model, output_path)


# ── Source-paper notes table ──────────────────────────────────────────


def generate_source_notes_table(
    output_path: Path,
) -> None:
    """Write a compact table summarising what each source paper reports and
    what differs from our setup."""
    from causal_meta.analysis.paper_comparison.reference_data import (
        load_reference,
        source_paper_notes,
    )

    ref = load_reference()
    notes = source_paper_notes()

    lines: list[str] = []
    lines.append(r"\small")
    lines.append(r"\begin{tabularx}{\textwidth}{lX}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Notes} \\")
    lines.append(r"\midrule")

    for model in MODELS:
        model_label = PAPER_MODEL_LABELS.get(model, model)
        cite = ref.get("source_papers", {}).get(model, {}).get("citation", "")
        note = notes.get(model, "")
        note_escaped = note.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
        lines.append(rf"{model_label} & {note_escaped} \\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote source notes table: %s", output_path)


# ── 3-way AVICI comparison table (one per metric) ────────────────────


def generate_avici_3way_table(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Write a LaTeX table comparing three AVICI variants for one metric.

    Rows: 9 families grouped by mechanism.
    Columns: BCNP paper "AVICI" | Thesis AVICI | Source AVICI (scm-v0)
    """
    metric_label = METRIC_LABELS.get(metric, metric)
    direction = METRIC_DIRECTIONS.get(metric, "lower_is_better")
    metric_df = df[df["metric"] == metric].copy()
    is_shd = metric == "e_shd"
    digits = 1 if is_shd else 2

    variant_labels = list(AVICI_VARIANTS.values())
    n_variants = len(variant_labels)
    value_cols = ["bcnp_paper_value", "thesis_value", "source_value"]

    lines: list[str] = []
    lines.append(r"\small")

    arrow = r"$\downarrow$" if direction == "lower_is_better" else r"$\uparrow$"

    col_spec = "l" + "r" * n_variants
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row
    header = [_bold("Setting")] + [_bold(lbl) for lbl in variant_labels]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Body rows
    prev_mechanism = ""
    for fam_key, fam_label in COMPARABLE_FAMILIES.items():
        mechanism = fam_label.split()[0]
        if mechanism != prev_mechanism and prev_mechanism:
            lines.append(r"\addlinespace")
        prev_mechanism = mechanism

        row = metric_df[metric_df["family"] == fam_key]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        vals: list[float | None] = [row[c] for c in value_cols]
        cells = [fam_label] + [_fmt(v, digits) for v in vals]

        # Bold the best value across the three variants
        best = _best_idx(vals, direction)
        if best is not None:
            cells[1 + best] = _bold(cells[1 + best])

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote AVICI 3-way table (%s): %s", metric, output_path)
