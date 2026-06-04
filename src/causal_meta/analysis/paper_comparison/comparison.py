from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from causal_meta.analysis.paper_comparison.reference_data import (
    AVICI_VARIANTS,
    COMPARABLE_FAMILIES,
    METRIC_MAP,
    MODELS,
    SOURCE_AVICI_DIR,
    cross_model_paper_values,
    source_paper_configs,
)

log = logging.getLogger(__name__)


# ── Thesis metrics loading ─────────────────────────────────────────────


def _load_model_summary(
    thesis_runs_root: Path, model: str
) -> dict[str, dict[str, Any]]:
    """Load ``summary`` from ``<model>/metrics.json`` and return family→metrics."""
    metrics_path = thesis_runs_root / model / "metrics.json"
    if not metrics_path.exists():
        log.warning("Missing metrics for %s: %s", model, metrics_path)
        return {}
    with open(metrics_path) as fh:
        data = json.load(fh)
    return data.get("summary", {})


# ── Cross-model comparison DataFrame ──────────────────────────────────


def build_cross_model_dataframe(
    thesis_runs_root: Path,
) -> pd.DataFrame:
    """Build a flat DataFrame comparing paper and thesis values.

    Columns:
        model, family, family_label, metric, paper_value, paper_std,
        thesis_value, thesis_std, delta
    """
    paper_vals = cross_model_paper_values()
    rows: list[dict[str, Any]] = []

    for model in MODELS:
        summary = _load_model_summary(thesis_runs_root, model)
        for fam_key, fam_label in COMPARABLE_FAMILIES.items():
            fam_summary = summary.get(fam_key, {})
            for paper_metric, thesis_metric_key in METRIC_MAP.items():
                paper_entry = (
                    paper_vals.get(fam_key, {}).get(model, {}).get(paper_metric)
                )
                paper_mean = paper_entry[0] if paper_entry else None
                paper_std = paper_entry[1] if paper_entry else None

                thesis_mean = fam_summary.get(thesis_metric_key)
                thesis_std_key = thesis_metric_key.replace("_mean", "_std")
                thesis_std = fam_summary.get(thesis_std_key)

                delta = None
                if paper_mean is not None and thesis_mean is not None:
                    delta = thesis_mean - paper_mean

                rows.append(
                    {
                        "model": model,
                        "family": fam_key,
                        "family_label": fam_label,
                        "metric": paper_metric,
                        "paper_value": paper_mean,
                        "paper_std": paper_std,
                        "thesis_value": thesis_mean,
                        "thesis_std": thesis_std,
                        "delta": delta,
                    }
                )

    return pd.DataFrame(rows)


# ── Hyperparameter comparison ──────────────────────────────────────────


def build_hyperparam_comparison(
    configs_root: Path,
) -> dict[str, pd.DataFrame]:
    """Build per-model DataFrames comparing paper vs our hyperparameters.

    Returns ``{model: DataFrame}`` with columns:
        parameter, paper_value, our_value
    """
    from omegaconf import OmegaConf

    paper_cfgs = source_paper_configs()
    result: dict[str, pd.DataFrame] = {}

    for model in MODELS:
        yaml_path = configs_root / "model" / f"{model}.yaml"
        if not yaml_path.exists():
            log.warning("Config not found: %s", yaml_path)
            continue

        our_cfg = OmegaConf.load(yaml_path)
        our_flat: dict[str, Any] = {
            k: v
            for k, v in OmegaConf.to_container(our_cfg, resolve=False).items()
            if not k.startswith("_") and k not in ("type", "id")
        }

        paper_cfg = paper_cfgs.get(model, {})

        # Build a reference JSON that holds the key mapping.
        ref = _load_key_map(model)
        overrides = _load_config_overrides(model)

        rows: list[dict[str, Any]] = []
        for paper_key, paper_value in paper_cfg.items():
            our_key = ref.get(paper_key, paper_key)
            our_value = our_flat.get(our_key, "N/A")
            # Fall back to manually specified overrides when YAML has no match.
            if our_value == "N/A" and paper_key in overrides:
                our_value = overrides[paper_key]
            rows.append(
                {
                    "parameter": paper_key,
                    "paper_value": _fmt_value(paper_value),
                    "our_value": _fmt_value(our_value),
                }
            )

        df = pd.DataFrame(rows)
        result[model] = df

    return result


def _load_key_map(model: str) -> dict[str, str]:
    """Load the ``our_config_key_map`` for a model from the reference JSON."""
    from causal_meta.analysis.paper_comparison.reference_data import load_reference

    ref = load_reference()
    return ref.get("source_papers", {}).get(model, {}).get("our_config_key_map", {})


def _load_config_overrides(model: str) -> dict[str, Any]:
    """Load ``our_config_overrides`` for a model from the reference JSON."""
    from causal_meta.analysis.paper_comparison.reference_data import load_reference

    ref = load_reference()
    return ref.get("source_papers", {}).get(model, {}).get("our_config_overrides", {})


def _fmt_value(v: Any) -> str:
    """Format a value for display."""
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:g}"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


# ── 3-way AVICI comparison DataFrame ──────────────────────────────────


def build_avici_3way_dataframe(
    thesis_runs_root: Path,
) -> pd.DataFrame:
    """Build a DataFrame comparing three AVICI variants on 9 comparable families.

    Columns:
        family, family_label, metric,
        bcnp_paper_value, bcnp_paper_std,
        thesis_value, thesis_std,
        source_value, source_std
    """
    paper_vals = cross_model_paper_values()
    thesis_summary = _load_model_summary(thesis_runs_root, "avici")
    source_summary = _load_model_summary(thesis_runs_root, SOURCE_AVICI_DIR)

    rows: list[dict[str, Any]] = []

    for fam_key, fam_label in COMPARABLE_FAMILIES.items():
        fam_thesis = thesis_summary.get(fam_key, {})
        fam_source = source_summary.get(fam_key, {})

        for paper_metric, thesis_metric_key in METRIC_MAP.items():
            std_key = thesis_metric_key.replace("_mean", "_std")

            # BCNP paper's AVICI values
            paper_entry = paper_vals.get(fam_key, {}).get("avici", {}).get(paper_metric)
            bcnp_paper_mean = paper_entry[0] if paper_entry else None
            bcnp_paper_std = paper_entry[1] if paper_entry else None

            # Thesis AVICI (our re-implementation, trained)
            thesis_mean = fam_thesis.get(thesis_metric_key)
            thesis_std = fam_thesis.get(std_key)

            # Source AVICI pretrained scm-v0
            source_mean = fam_source.get(thesis_metric_key)
            source_std = fam_source.get(std_key)

            rows.append(
                {
                    "family": fam_key,
                    "family_label": fam_label,
                    "metric": paper_metric,
                    "bcnp_paper_value": bcnp_paper_mean,
                    "bcnp_paper_std": bcnp_paper_std,
                    "thesis_value": thesis_mean,
                    "thesis_std": thesis_std,
                    "source_value": source_mean,
                    "source_std": source_std,
                }
            )

    return pd.DataFrame(rows)
