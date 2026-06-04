"""Paper plausibility comparison for Appendix F.

Compares thesis experiment metrics against values reported in the source papers
to verify that our model implementations produce plausible results.
"""

from __future__ import annotations

import logging
from pathlib import Path

from causal_meta.analysis.paper_comparison.comparison import (
    build_avici_3way_dataframe,
    build_cross_model_dataframe,
    build_hyperparam_comparison,
)
from causal_meta.analysis.paper_comparison.reference_data import MODELS
from causal_meta.analysis.paper_comparison.tables import (
    generate_avici_3way_table,
    generate_cross_model_table,
    generate_hyperparam_table,
    generate_source_notes_table,
)

log = logging.getLogger(__name__)


def generate_paper_comparison(
    thesis_runs_root: Path,
    thesis_root: Path,
    configs_root: Path,
) -> None:
    """Run the full paper plausibility comparison and write LaTeX artifacts.

    Args:
        thesis_runs_root: Path to ``experiments/thesis_runs/``.
        thesis_root: Path to ``paper/final_thesis/``.
        configs_root: Path to ``src/causal_meta/configs/``.
    """
    from causal_meta.analysis.common.thesis import GRAPHICS_APPENDIX_F

    output_dir = thesis_root / GRAPHICS_APPENDIX_F
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Cross-model performance tables (one per metric) ────────────
    cm_df = build_cross_model_dataframe(thesis_runs_root)

    for metric in ("auc", "e_shd", "e_edgef1"):
        generate_cross_model_table(
            cm_df,
            metric=metric,
            output_path=output_dir / f"plausibility_{metric}.tex",
        )

    # ── 2. Hyperparameter comparison tables (one per model) ───────────
    hp_dfs = build_hyperparam_comparison(configs_root)
    for model, hp_df in hp_dfs.items():
        generate_hyperparam_table(
            hp_df,
            model=model,
            output_path=output_dir / f"hp_comparison_{model}.tex",
        )

    # ── 3. 3-way AVICI comparison tables (one per metric) ────────────
    avici_df = build_avici_3way_dataframe(thesis_runs_root)
    n_avici_tables = 0
    for metric in ("auc", "e_shd", "e_edgef1"):
        generate_avici_3way_table(
            avici_df,
            metric=metric,
            output_path=output_dir / f"avici_3way_{metric}.tex",
        )
        n_avici_tables += 1

    # ── 4. Source paper notes table ───────────────────────────────────
    generate_source_notes_table(output_dir / "source_paper_notes.tex")

    log.info(
        "Paper plausibility comparison complete. %d artifacts in %s",
        3 + len(hp_dfs) + n_avici_tables + 1,
        output_dir,
    )
