from __future__ import annotations

import argparse
import logging
from pathlib import Path

from causal_meta.analysis.utils import (
    EmptyAnalysisDataError,
    generate_all_artifacts_from_runs,
    RawGranularityError,
    resolve_run_directories,
    RunSelectionError,
)

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate analysis plots/tables from selected run directories "
            "containing metrics.json"
        )
    )
    parser.add_argument(
        "runs_root",
        type=str,
        nargs="?",
        default="experiments/runs",
        help=(
            "Root directory used for run discovery and resolving --run-id values "
            "(default: experiments/runs)"
        ),
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help=(
            "Run ID (directory name under runs_root). Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Explicit run directory path. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for artifacts. Defaults to <runs_root>/graphics.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail on analysis sub-step errors (e.g., non-per-task raw metrics, "
            "posterior diagnostics failures) instead of skipping with warnings."
        ),
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_dirs = [Path(path) for path in args.run_dir]

    selected = resolve_run_directories(
        runs_root=runs_root,
        run_ids=list(args.run_id),
        run_dirs=run_dirs,
    )
    output_dir = Path(args.output_dir) if args.output_dir else (runs_root / "graphics")

    log.info("Selected %d runs for analysis.", len(selected))
    generate_all_artifacts_from_runs(selected, output_dir, strict=bool(args.strict))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except EmptyAnalysisDataError as exc:
        log.error("No analysis data found: %s", exc)
        raise SystemExit(1) from exc
    except (FileNotFoundError, RunSelectionError) as exc:
        log.error("Run selection failed: %s", exc)
        raise SystemExit(1) from exc
    except RawGranularityError as exc:
        log.error("Raw granularity check failed: %s", exc)
        raise SystemExit(1) from exc
