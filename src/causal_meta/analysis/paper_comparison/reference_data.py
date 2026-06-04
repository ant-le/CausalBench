from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_JSON_PATH = Path(__file__).parent / "paper_reference.json"

# ── Comparable families (thesis key → human label) ─────────────────────

COMPARABLE_FAMILIES: dict[str, str] = {
    "id_linear_er20_d20_n500": "Linear ER-20",
    "id_linear_er40_d20_n500": "Linear ER-40",
    "id_linear_er60_d20_n500": "Linear ER-60",
    "id_neuralnet_er20_d20_n500": "MLP ER-20",
    "id_neuralnet_er40_d20_n500": "MLP ER-40",
    "id_neuralnet_er60_d20_n500": "MLP ER-60",
    "id_gpcde_er20_d20_n500": "RFF ER-20",
    "id_gpcde_er40_d20_n500": "RFF ER-40",
    "id_gpcde_er60_d20_n500": "RFF ER-60",
}

MODELS: tuple[str, ...] = ("avici", "bcnp", "dibs", "bayesdag")

# Source AVICI pretrained checkpoint evaluated under thesis conditions.
SOURCE_AVICI_DIR = "avici_pretrained_scm-v0"

# Three implementations of AVICI for the 3-way comparison table.
# Keys are used as column identifiers; values describe the source.
AVICI_VARIANTS: dict[str, str] = {
    "bcnp_paper": "BCNP paper re-impl.",
    "thesis": "Thesis re-impl.",
    "source_pretrained": r"Source \texttt{scm-v0}",
}

# Metrics that are directly comparable between the paper and our runs.
METRIC_MAP: dict[str, str] = {
    "auc": "auc_mean",
    "e_shd": "e-shd_mean",
    "e_edgef1": "e-edgef1_mean",
}

METRIC_LABELS: dict[str, str] = {
    "auc": "AUROC",
    "e_shd": r"$\widehat{\mathbb{E}}$-SHD",
    "e_edgef1": r"$\widehat{\mathbb{E}}$-Edge\,F1",
}

METRIC_DIRECTIONS: dict[str, str] = {
    "auc": "higher_is_better",
    "e_shd": "lower_is_better",
    "e_edgef1": "higher_is_better",
}


def load_reference() -> dict[str, Any]:
    """Load the full paper reference JSON and return as a dict."""
    with open(_JSON_PATH) as fh:
        return json.load(fh)


def cross_model_paper_values() -> dict[str, dict[str, dict[str, list[float]]]]:
    """Return ``{family_key: {model: {metric: [mean, std]}}}`` for the 9 families.

    Values come from the BCNP paper Tables 7--15 (All Data variant).
    """
    ref = load_reference()
    families = ref["cross_model_comparison"]["families"]
    return {fam_key: fam_data["models"] for fam_key, fam_data in families.items()}


def source_paper_configs() -> dict[str, dict[str, Any]]:
    """Return ``{model: {param: value}}`` hyperparams from each model's source paper."""
    ref = load_reference()
    return {
        model: info["paper_hyperparams"] for model, info in ref["source_papers"].items()
    }


def source_paper_config_differences() -> dict[str, list[str]]:
    """Return ``{model: [difference strings]}`` for each model."""
    ref = load_reference()
    return {
        model: info["known_config_differences"]
        for model, info in ref["source_papers"].items()
    }


def source_paper_notes() -> dict[str, str]:
    """Return ``{model: note}`` from each model's comparable_results.note."""
    ref = load_reference()
    return {
        model: info.get("comparable_results", {}).get("note", "")
        for model, info in ref["source_papers"].items()
    }
