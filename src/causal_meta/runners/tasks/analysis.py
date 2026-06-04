from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from causal_meta.analysis.utils import (
    EmptyAnalysisDataError,
    generate_all_artifacts_from_runs,
    RawGranularityError,
    resolve_run_directories,
    RunSelectionError,
)

log = logging.getLogger(__name__)


def run(cfg: DictConfig, output_dir: Path) -> None:
    """Run post-hoc analysis for selected run IDs/directories.

    Expected config keys under ``analysis``:
      - ``runs_root``: root folder for run ID resolution/discovery.
      - ``run_ids``: list of run IDs (directory names under ``runs_root``).
      - ``run_dirs``: list of explicit run directories.
      - ``output_dir``: where graphics/tables should be written.
    """
    analysis_cfg = cfg.get("analysis", {})
    runs_root_raw = analysis_cfg.get("runs_root", None)
    runs_root = Path(str(runs_root_raw)) if runs_root_raw else output_dir.parent

    run_ids = [str(item) for item in analysis_cfg.get("run_ids", [])]
    run_dirs = [Path(str(item)) for item in analysis_cfg.get("run_dirs", [])]
    strict = bool(analysis_cfg.get("strict", False))

    output_dir_raw = analysis_cfg.get("output_dir", None)
    graphics_dir = (
        Path(str(output_dir_raw)) if output_dir_raw else (runs_root / "graphics")
    )

    log.info("Running analysis from runs_root=%s", runs_root)
    try:
        selected_runs = resolve_run_directories(
            runs_root=runs_root,
            run_ids=run_ids,
            run_dirs=run_dirs,
        )
        log.info("Selected %d runs for analysis.", len(selected_runs))
        generate_all_artifacts_from_runs(selected_runs, graphics_dir, strict=strict)
    except (FileNotFoundError, RunSelectionError) as exc:
        if strict:
            raise
        log.warning("Analysis input resolution failed: %s", exc)
        return
    except EmptyAnalysisDataError:
        if strict:
            raise
        log.warning("No rows available after loading selected run metrics.")
        return
    except RawGranularityError as exc:
        if strict:
            raise
        log.warning("Raw granularity check failed: %s", exc)
        return

    log.info("Analysis artifacts saved to %s", graphics_dir)
