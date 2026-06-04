from __future__ import annotations

import argparse
import logging
from pathlib import Path

from causal_meta.analysis.appendix.generator import generate_appendix_artifacts

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis appendix LaTeX snippets from Hydra configuration files."
        )
    )
    parser.add_argument(
        "--thesis-root",
        type=str,
        default="paper/final_thesis",
        help="Thesis root containing graphics/ chapter directories.",
    )
    parser.add_argument(
        "--configs-root",
        type=str,
        default="src/causal_meta/configs",
        help="Hydra configs root used as the appendix source.",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=None,
        help=(
            "Thesis-runs root (e.g. experiments/thesis_runs). "
            "When provided, also regenerates Appendix F paper-comparison tables."
        ),
    )
    args = parser.parse_args()

    thesis_root = Path(args.thesis_root)
    configs_root = Path(args.configs_root)
    generated = generate_appendix_artifacts(thesis_root, configs_root)
    log.info(
        "Generated %d appendix snippet files under %s", len(generated), thesis_root
    )

    if args.input_root is not None:
        from causal_meta.analysis.paper_comparison import generate_paper_comparison

        input_root = Path(args.input_root)
        log.info("Generating Appendix F paper-comparison tables from %s", input_root)
        generate_paper_comparison(input_root, thesis_root, configs_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
