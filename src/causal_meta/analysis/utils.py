from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

log = logging.getLogger(__name__)


class EmptyAnalysisDataError(RuntimeError):
    """Raised when selected runs contain no usable analysis rows."""


class RunSelectionError(ValueError):
    """Raised when run selection arguments are invalid."""


class RawGranularityError(RuntimeError):
    """Raised when per-task analysis is requested for non-per-task raw metrics."""


# Run/model keys -> display names
MODEL_NAME_MAP: dict[str, str] = {
    "avici": "AviCi",
    "avici_smoke": "AviCi",
    "avici_full": "AviCi",
    "bcnp": "BCNP",
    "bcnp_smoke": "BCNP",
    "bcnp_full": "BCNP",
    "dibs": "DiBS",
    "random": "Random",
    "random_smoke": "Random",
    "bayesdag": "BayesDAG",
}

# Canonical thesis labels used in paper tables/figures.
# Keys must match the directory names under experiments/thesis_runs/.
PAPER_MODEL_LABELS: dict[str, str] = {
    "avici": "AviCi",
    "bcnp": "BCNP",
    "dibs": "DiBS",
    "bayesdag": "BayesDAG",
    "random": "Random",
}

# ── Unified plot styling constants ─────────────────────────────────────

AMORTISED_MODELS: frozenset[str] = frozenset({"AviCi", "BCNP"})
"""Display names of amortised (meta-learning) models."""

EXPLICIT_MODELS: frozenset[str] = frozenset({"DiBS", "BayesDAG"})
"""Display names of explicit (per-task) models."""

MODEL_COLORS: dict[str, str] = {
    "AviCi": "#1f77b4",  # blue
    "BCNP": "#ff7f0e",  # orange
    "DiBS": "#7f7f7f",  # medium grey
    "BayesDAG": "#bfbfbf",  # light grey
    "Random": "#2ca02c",  # green
}
"""Consistent per-model colours: amortised in colour, explicit in grey."""

MODEL_MARKERS: dict[str, str] = {
    "AviCi": "o",
    "BCNP": "s",
    "DiBS": "D",
    "BayesDAG": "^",
    "Random": "X",
}
"""Consistent per-model markers for scatter/line plots."""

# Error-decomposition colours use a muted non-model palette so stacked error
# bars are distinguishable without overlapping the model-identity colours.
ERROR_COLORS: dict[str, str] = {
    "FP": "#6a3d9a",  # purple
    "FN": "#a6761d",  # brown
    "Reversed": "#e7298a",  # magenta
}
"""Consistent colours for FP / FN / Reversed error components."""

ERROR_SPECS: list[tuple[str, str, str]] = [
    ("fp_count", "FP", ERROR_COLORS["FP"]),
    ("fn_count", "FN", ERROR_COLORS["FN"]),
    ("reversed_count", "Reversed", ERROR_COLORS["Reversed"]),
]
"""(metric_key, display_label, colour) triples for error decomposition bars."""


# ── CSV companion helper ───────────────────────────────────────────────


def save_figure_data(
    figure_path: str | Path,
    data: pd.DataFrame,
    *,
    suffix: str = ".csv",
) -> Path | None:
    """Save the underlying data for a figure as a CSV alongside the PDF.

    The CSV path mirrors *figure_path* with the extension replaced by *suffix*.
    If *data* is empty the file is **not** written and ``None`` is returned.

    Args:
        figure_path: Path to the figure file (typically ``.pdf``).
        data: The DataFrame that backs the figure.
        suffix: File extension for the data file (default ``".csv"``).

    Returns:
        The path to the written CSV, or ``None`` if *data* was empty.
    """
    if data.empty:
        return None
    csv_path = Path(figure_path).with_suffix(suffix)
    data.to_csv(csv_path, index=False)
    log.info("Saved figure data (%d rows) to %s", len(data), csv_path)
    return csv_path


# Dataset keys -> human-readable labels
DATASET_DESCRIPTION_MAP: dict[str, str] = {
    # ── ID: ER × {Linear, MLP, GP} ────────────────────────────────────
    "id_linear_er20": "ID Linear ER-20",
    "id_linear_er40": "ID Linear ER-40",
    "id_linear_er60": "ID Linear ER-60",
    "id_neuralnet_er20": "ID MLP ER-20",
    "id_neuralnet_er40": "ID MLP ER-40",
    "id_neuralnet_er60": "ID MLP ER-60",
    "id_gpcde_er20": "ID GP ER-20",
    "id_gpcde_er40": "ID GP ER-40",
    "id_gpcde_er60": "ID GP ER-60",
    # ── ID: SF × {Linear, MLP, GP} ────────────────────────────────────
    "id_linear_sf1": "ID Linear SF-1",
    "id_linear_sf2": "ID Linear SF-2",
    "id_linear_sf3": "ID Linear SF-3",
    "id_neuralnet_sf1": "ID MLP SF-1",
    "id_neuralnet_sf2": "ID MLP SF-2",
    "id_neuralnet_sf3": "ID MLP SF-3",
    "id_gpcde_sf1": "ID GP SF-1",
    "id_gpcde_sf2": "ID GP SF-2",
    "id_gpcde_sf3": "ID GP SF-3",
    # ── OOD Graph: SBM × {Linear, MLP, GP} ────────────────────────────
    "ood_graph_sbm_linear": "OOD-G SBM Linear",
    "ood_graph_sbm_neuralnet": "OOD-G SBM MLP",
    "ood_graph_sbm_gpcde": "OOD-G SBM GP",
    # ── OOD Graph: WS × {Linear, MLP, GP} ─────────────────────────────
    "ood_graph_ws_linear": "OOD-G WS Linear",
    "ood_graph_ws_neuralnet": "OOD-G WS MLP",
    "ood_graph_ws_gpcde": "OOD-G WS GP",
    # ── OOD Graph: GRG × {Linear, MLP, GP} ────────────────────────────
    "ood_graph_grg_linear": "OOD-G GRG Linear",
    "ood_graph_grg_neuralnet": "OOD-G GRG MLP",
    "ood_graph_grg_gpcde": "OOD-G GRG GP",
    # ── OOD Mechanism: ER-20 × {Periodic, Square, LogisticMap, PNL} ───
    "ood_mech_periodic_er20": "OOD-M Periodic (ER-20)",
    "ood_mech_square_er20": "OOD-M Square (ER-20)",
    "ood_mech_logistic_map_er20": "OOD-M Logistic Map (ER-20)",
    "ood_mech_pnl_tanh_er20": "OOD-M PNL (ER-20)",
    # ── OOD Mechanism: ER-60 × {Periodic, Square, LogisticMap, PNL} ───
    "ood_mech_periodic_er60": "OOD-M Periodic (ER-60)",
    "ood_mech_square_er60": "OOD-M Square (ER-60)",
    "ood_mech_logistic_map_er60": "OOD-M Logistic Map (ER-60)",
    "ood_mech_pnl_tanh_er60": "OOD-M PNL (ER-60)",
    # ── OOD Mechanism: SF-2 × {Periodic, Square, LogisticMap, PNL} ────
    "ood_mech_periodic_sf2": "OOD-M Periodic (SF-2)",
    "ood_mech_square_sf2": "OOD-M Square (SF-2)",
    "ood_mech_logistic_map_sf2": "OOD-M Logistic Map (SF-2)",
    "ood_mech_pnl_tanh_sf2": "OOD-M PNL (SF-2)",
    # ── OOD Noise: Linear × ER-20 × {Laplace, Uniform} ────────────────
    "ood_noise_laplace_linear_er20": "OOD-N Laplace (Linear/ER-20)",
    "ood_noise_uniform_linear_er20": "OOD-N Uniform (Linear/ER-20)",
    # ── OOD Noise: MLP × SF-2 × {Laplace, Uniform} ───────────────────
    "ood_noise_laplace_neuralnet_sf2": "OOD-N Laplace (MLP/SF-2)",
    "ood_noise_uniform_neuralnet_sf2": "OOD-N Uniform (MLP/SF-2)",
    # ── OOD Both: SBM × Periodic ──────────────────────────────────────
    "ood_both_sbm_periodic": "OOD-Both SBM Periodic",
    "ood_both_sbm_pnl_tanh": "OOD-Both SBM PNL",
    # ── OOD Both: WS/GRG × Periodic ───────────────────────────────────
    "ood_both_ws_periodic": "OOD-Both WS Periodic",
    "ood_both_grg_periodic": "OOD-Both GRG Periodic",
    # ── OOD Both: WS/GRG × PNL ────────────────────────────────────────
    "ood_both_ws_pnl_tanh": "OOD-Both WS PNL",
    "ood_both_grg_pnl_tanh": "OOD-Both GRG PNL",
    # ── OOD Both: SBM/WS/GRG × Logistic Map ───────────────────────────
    "ood_both_sbm_logistic_map": "OOD-Both SBM Logistic Map",
    "ood_both_ws_logistic_map": "OOD-Both WS Logistic Map",
    "ood_both_grg_logistic_map": "OOD-Both GRG Logistic Map",
}

GRAPH_DESCRIPTION_MAP: dict[str, str] = {
    "er20": "ER20",
    "er40": "ER40",
    "er60": "ER60",
    "sf1": "SF1",
    "sf2": "SF2",
    "sf3": "SF3",
    "sbm": "SBM",
    "ws": "WS",
    "grg": "GRG",
}

MECH_DESCRIPTION_MAP: dict[str, str] = {
    "linear": "Linear",
    "neuralnet": "MLP",
    "gpcde": "RFF",
    "periodic": "Periodic",
    "square": "Square",
    "logistic_map": "Logistic Map",
    "pnl_tanh": "PNL",
    "pnl_sigmoid": "PNL",
}

SHIFT_DESCRIPTION_MAP: dict[str, str] = {
    "id": "ID",
    "ood_graph": "OOD-Graph",
    "ood_mech": "OOD-Mech",
    "ood_noise": "OOD-Noise",
    "ood_both": "OOD-Both",
    "ood_nodes": "OOD-Nodes",
    "ood_samples": "OOD-Samples",
}


def map_model_name(raw_model: str) -> str:
    """Map a model key to a display name."""
    if raw_model in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[raw_model]
    for suffix in ("_smoke",):
        if raw_model.endswith(suffix):
            base = raw_model[: -len(suffix)]
            return MODEL_NAME_MAP.get(base, base)
    return raw_model


def map_dataset_description(raw_dataset: str) -> str:
    """Map a dataset key to a human-readable description."""
    if raw_dataset in DATASET_DESCRIPTION_MAP:
        return DATASET_DESCRIPTION_MAP[raw_dataset]
    generated = _auto_format_dataset_key(raw_dataset)
    return generated if generated is not None else raw_dataset


def _format_token(token: str, mapping: Mapping[str, str]) -> str:
    mapped = mapping.get(token)
    if mapped is not None:
        return mapped
    return token.replace("_", " ").title()


def _auto_format_dataset_key(raw_dataset: str) -> str | None:
    match = re.match(r"^(?P<body>.+?)_d(?P<d>\d+)_n(?P<n>\d+)$", raw_dataset)
    if match is None:
        return None

    body = str(match.group("body"))
    n_nodes = int(match.group("d"))
    n_samples = int(match.group("n"))

    shift_prefix = next(
        (
            prefix
            for prefix in sorted(SHIFT_DESCRIPTION_MAP.keys(), key=len, reverse=True)
            if body == prefix or body.startswith(f"{prefix}_")
        ),
        None,
    )
    if shift_prefix is None:
        return None

    remainder = body[len(shift_prefix) :].lstrip("_")
    tokens = [token for token in remainder.split("_") if token]
    if not tokens:
        return None

    graph_token: str | None = None
    mech_token = ""
    graph_positions = [
        idx for idx, token in enumerate(tokens) if token in GRAPH_DESCRIPTION_MAP
    ]
    if not graph_positions:
        return None

    if shift_prefix in {"ood_graph", "ood_both"}:
        graph_index = graph_positions[0]
        graph_token = tokens[graph_index]
        mech_token = "_".join(tokens[graph_index + 1 :])
    else:
        graph_index = graph_positions[-1]
        graph_token = tokens[graph_index]
        mech_token = "_".join(tokens[:graph_index])

    pieces = [SHIFT_DESCRIPTION_MAP[shift_prefix]]
    if mech_token:
        pieces.append(_format_token(mech_token, MECH_DESCRIPTION_MAP))
    if graph_token:
        pieces.append(_format_token(graph_token, GRAPH_DESCRIPTION_MAP))
    pieces.append(f"(d={n_nodes}, n={n_samples})")
    return " ".join(pieces)


def resolve_run_directories(
    *,
    runs_root: Path | None,
    run_ids: Sequence[str] | None = None,
    run_dirs: Sequence[Path] | None = None,
) -> list[Path]:
    """Resolve run directories from explicit IDs/paths or by discovery.

    Args:
        runs_root: Root directory under which run IDs are resolved.
        run_ids: Optional run IDs (directory names under ``runs_root``).
        run_dirs: Optional explicit run directories.

    Returns:
        A de-duplicated list of absolute run directories containing ``metrics.json``.

    Raises:
        RunSelectionError: If selection arguments are inconsistent.
        FileNotFoundError: If no valid runs are found.
    """
    selected_dirs: list[Path] = []

    if run_dirs:
        selected_dirs.extend(Path(run_dir) for run_dir in run_dirs)

    if run_ids:
        if runs_root is None:
            raise RunSelectionError("runs_root is required when run_ids are provided.")
        selected_dirs.extend(Path(runs_root) / str(run_id) for run_id in run_ids)

    if not selected_dirs:
        if runs_root is None:
            raise RunSelectionError(
                "Provide run_ids/run_dirs, or provide runs_root for automatic discovery."
            )
        selected_dirs = sorted(
            path.parent for path in Path(runs_root).rglob("metrics.json")
        )

    if not selected_dirs:
        raise FileNotFoundError("No run directories with metrics.json were found.")

    resolved: list[Path] = []
    seen: set[str] = set()
    for run_dir in selected_dirs:
        absolute_run_dir = Path(run_dir).expanduser().resolve()
        metrics_path = absolute_run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing metrics.json for run '{absolute_run_dir}'. "
                f"Expected file: {metrics_path}"
            )
        key = str(absolute_run_dir)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(absolute_run_dir)

    return resolved


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_base_metrics(metrics: Mapping[str, Any]) -> set[str]:
    base_metrics: set[str] = set()
    for key, value in metrics.items():
        if key.endswith("_mean"):
            base_metrics.add(key[:-5])
        elif key.endswith("_sem"):
            base_metrics.add(key[:-4])
        elif key.endswith("_std"):
            base_metrics.add(key[:-4])
        elif not key.startswith("_") and isinstance(value, (int, float)):
            base_metrics.add(str(key))
    return base_metrics


def _infer_model_key(run_id: str) -> str:
    run_id_norm = run_id.lower()
    candidates = sorted(MODEL_NAME_MAP.keys(), key=len, reverse=True)
    for candidate in candidates:
        candidate_norm = candidate.lower()
        if (
            run_id_norm == candidate_norm
            or run_id_norm.endswith(f"_{candidate_norm}")
            or f"_{candidate_norm}_" in run_id_norm
        ):
            return candidate
    return run_id


def _is_explicit_model_key(model_key: str) -> bool:
    """Return True for explicit (non-amortized) model families."""
    model_key_norm = model_key.lower()
    explicit_tokens = ("dibs", "bayesdag", "random")
    return any(token in model_key_norm for token in explicit_tokens)


def load_runs_dataframe(
    run_dirs: Sequence[Path], *, translate_names: bool = True
) -> pd.DataFrame:
    """Load selected run metrics into a normalized long-format DataFrame.

    Args:
        run_dirs: Run directories containing ``metrics.json``.
        translate_names: Whether to map model/dataset keys to display names.

    Returns:
        DataFrame with rows per (run, dataset, metric).
    """
    rows: list[dict[str, object]] = []

    for run_dir in run_dirs:
        metrics_path = Path(run_dir) / "metrics.json"
        with open(metrics_path, "r") as f:
            payload = json.load(f)

        if not isinstance(payload, Mapping):
            log.warning("Skipping malformed metrics payload at %s", metrics_path)
            continue

        metadata = _as_mapping(payload.get("metadata"))
        run_id = str(metadata.get("run_id", Path(run_dir).name))
        run_name = str(metadata.get("run_name", run_id))

        model_key_raw = metadata.get("model_name")
        model_key = (
            str(model_key_raw)
            if model_key_raw is not None and str(model_key_raw).strip()
            else _infer_model_key(run_id)
        )
        model_name = map_model_name(model_key) if translate_names else model_key

        summary = _as_mapping(payload.get("summary"))
        fam_meta = _as_mapping(payload.get("family_metadata"))
        distances = _as_mapping(payload.get("distances"))

        for dataset_key, dataset_metrics_any in summary.items():
            dataset_key_str = str(dataset_key)
            dataset_metrics = _as_mapping(dataset_metrics_any)
            if not dataset_metrics:
                continue

            dataset_name = (
                map_dataset_description(dataset_key_str)
                if translate_names
                else dataset_key_str
            )

            # Enrich with family metadata (Phase C)
            ds_fam = _as_mapping(fam_meta.get(dataset_key_str))
            graph_type = str(ds_fam.get("graph_type", "")) if ds_fam else ""
            mech_type = str(ds_fam.get("mech_type", "")) if ds_fam else ""
            n_nodes = ds_fam.get("n_nodes")
            samples_per_task = ds_fam.get("samples_per_task")
            sparsity_param = ds_fam.get("sparsity_param")

            # Enrich with distributional distances (Phase D)
            ds_dist = _as_mapping(distances.get(dataset_key_str))
            spectral_dist = _to_float(ds_dist.get("spectral"), float("nan"))
            kl_degree_dist = _to_float(ds_dist.get("kl_degree"), float("nan"))
            mechanism_dist = _to_float(ds_dist.get("mechanism"), float("nan"))

            for metric in sorted(_extract_base_metrics(dataset_metrics)):
                mean_raw = dataset_metrics.get(f"{metric}_mean")
                sem_raw = dataset_metrics.get(f"{metric}_sem", 0.0)
                std_raw = dataset_metrics.get(f"{metric}_std", 0.0)
                if mean_raw is None:
                    mean_raw = dataset_metrics.get(metric, float("nan"))

                rows.append(
                    {
                        "RunID": run_id,
                        "RunName": run_name,
                        "RunDir": str(run_dir),
                        "Model": model_name,
                        "Dataset": dataset_name,
                        "ModelKey": model_key,
                        "DatasetKey": dataset_key_str,
                        "Metric": metric,
                        "Mean": _to_float(mean_raw, float("nan")),
                        "SEM": _to_float(sem_raw, 0.0),
                        "Std": _to_float(std_raw, 0.0),
                        "GraphType": graph_type,
                        "MechType": mech_type,
                        "NNodes": int(n_nodes) if n_nodes is not None else None,
                        "SamplesPerTask": (
                            int(samples_per_task)
                            if samples_per_task is not None
                            else None
                        ),
                        "SparsityParam": (
                            float(sparsity_param)
                            if sparsity_param is not None
                            else None
                        ),
                        "SpectralDist": spectral_dist,
                        "KLDegreeDist": kl_degree_dist,
                        "MechanismDist": mechanism_dist,
                    }
                )

    return pd.DataFrame(rows)


def load_raw_task_dataframe(
    run_dirs: Sequence[Path],
    metrics: Sequence[str] | None = None,
    *,
    translate_names: bool = True,
    require_per_task: bool = False,
    skip_non_per_task: bool = False,
) -> pd.DataFrame:
    """Load per-task raw metric values into a long-format DataFrame.

    Unlike :func:`load_runs_task_dataframe`, this reads the ``"raw"`` block from
    each ``metrics.json`` and returns **one row per (run, dataset, task, metric)**.

    Args:
        run_dirs: Run directories containing ``metrics.json``.
        metrics: If given, only load these metric keys.  ``None`` loads all.
        translate_names: Whether to map model/dataset keys to display names.
        require_per_task: If ``True``, only accept runs whose
            ``metadata.raw_granularity`` is ``"per_task"``.
        skip_non_per_task: When ``require_per_task`` is enabled, skip
            non-conforming runs instead of raising :class:`RawGranularityError`.

    Returns:
        DataFrame with columns ``RunID``, ``Model``, ``ModelKey``, ``DatasetKey``,
        ``Dataset``, ``TaskIdx``, ``Metric``, ``Value``, plus enrichment columns
        ``GraphType``, ``MechType``, ``NNodes``, ``SparsityParam``,
        ``SpectralDist``, ``KLDegreeDist``.
    """
    rows: list[dict[str, object]] = []

    for run_dir in run_dirs:
        metrics_path = Path(run_dir) / "metrics.json"
        with open(metrics_path, "r") as f:
            payload = json.load(f)

        if not isinstance(payload, Mapping):
            continue

        metadata = _as_mapping(payload.get("metadata"))
        run_id = str(metadata.get("run_id", Path(run_dir).name))
        run_name = str(metadata.get("run_name", run_id))

        model_key_raw = metadata.get("model_name")
        model_key = (
            str(model_key_raw)
            if model_key_raw is not None and str(model_key_raw).strip()
            else _infer_model_key(run_id)
        )
        model_name = map_model_name(model_key) if translate_names else model_key

        raw_granularity = str(metadata.get("raw_granularity", "")).strip().lower()
        if not raw_granularity:
            if _is_explicit_model_key(model_key):
                raw_granularity = "per_task"
            else:
                batch_size_test = metadata.get("batch_size_test", 1)
                try:
                    raw_granularity = (
                        "per_batch" if int(batch_size_test) > 1 else "per_task"
                    )
                except (TypeError, ValueError):
                    raw_granularity = "unknown"

        if require_per_task and raw_granularity != "per_task":
            message = (
                "Raw metric granularity mismatch for run "
                f"'{run_id}' ({run_dir}): expected per_task, found "
                f"'{raw_granularity}'."
            )
            if skip_non_per_task:
                log.warning("%s Skipping run.", message)
                continue
            raise RawGranularityError(message)

        raw = _as_mapping(payload.get("raw"))
        fam_meta = _as_mapping(payload.get("family_metadata"))
        distances = _as_mapping(payload.get("distances"))

        for dataset_key, raw_metrics_any in raw.items():
            dataset_key_str = str(dataset_key)
            raw_metrics = _as_mapping(raw_metrics_any)
            if not raw_metrics:
                continue

            dataset_name = (
                map_dataset_description(dataset_key_str)
                if translate_names
                else dataset_key_str
            )

            # Enrichment (Phase C/D)
            ds_fam = _as_mapping(fam_meta.get(dataset_key_str))
            graph_type = str(ds_fam.get("graph_type", "")) if ds_fam else ""
            mech_type = str(ds_fam.get("mech_type", "")) if ds_fam else ""
            n_nodes = ds_fam.get("n_nodes")
            samples_per_task = ds_fam.get("samples_per_task")
            sparsity_param = ds_fam.get("sparsity_param")

            ds_dist = _as_mapping(distances.get(dataset_key_str))
            spectral_dist = _to_float(ds_dist.get("spectral"), float("nan"))
            kl_degree_dist = _to_float(ds_dist.get("kl_degree"), float("nan"))
            mechanism_dist = _to_float(ds_dist.get("mechanism"), float("nan"))

            for metric_name, values_any in raw_metrics.items():
                if metrics is not None and metric_name not in metrics:
                    continue
                if not isinstance(values_any, list):
                    continue
                # Skip prefixed duplicates (e.g. "dataset_key/metric")
                if "/" in metric_name:
                    continue

                for task_idx, val in enumerate(values_any):
                    rows.append(
                        {
                            "RunID": run_id,
                            "RunName": run_name,
                            "Model": model_name,
                            "ModelKey": model_key,
                            "DatasetKey": dataset_key_str,
                            "Dataset": dataset_name,
                            "TaskIdx": task_idx,
                            "Metric": metric_name,
                            "Value": _to_float(val, float("nan")),
                            "GraphType": graph_type,
                            "MechType": mech_type,
                            "NNodes": (int(n_nodes) if n_nodes is not None else None),
                            "SamplesPerTask": (
                                int(samples_per_task)
                                if samples_per_task is not None
                                else None
                            ),
                            "SparsityParam": (
                                float(sparsity_param)
                                if sparsity_param is not None
                                else None
                            ),
                            "SpectralDist": spectral_dist,
                            "KLDegreeDist": kl_degree_dist,
                            "MechanismDist": mechanism_dist,
                        }
                    )

    return pd.DataFrame(rows)


def generate_all_artifacts_from_runs(
    run_dirs: Sequence[Path],
    output_dir: Path,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Generate all standard plots/tables from selected run directories.

    Args:
        run_dirs: Selected run directories containing ``metrics.json``.
        output_dir: Directory where figures and tables are written.
        strict: If ``True``, raise on analysis sub-step errors instead of
            warning and continuing.

    Returns:
        The normalized DataFrame used to create outputs.

    Raises:
        EmptyAnalysisDataError: If selected runs contain no usable rows.
    """
    from causal_meta.analysis.diagnostics.failure_modes import (
        classify_failure_modes, failure_mode_fractions)
    from causal_meta.analysis.diagnostics.posterior import \
        run_posterior_diagnostics_from_runs
    from causal_meta.analysis.generalisation.tables import (
        generate_distance_regression_table, generate_robustness_table)
    from causal_meta.analysis.plots.results import (
        generate_calibration_scatter, generate_density_stratified_figure,
        generate_distance_degradation_scatter, generate_entropy_histogram,
        generate_event_probability_bar, generate_failure_mode_bar,
        generate_performance_figure, generate_posterior_diagnostic_violins,
        generate_selective_prediction_pareto, generate_structural_figure)
    from causal_meta.analysis.uncertainty.ood_detection import (
        compute_ood_detection_metrics, compute_selective_prediction,
        generate_ood_detection_table)

    df = load_runs_dataframe(run_dirs)
    if df.empty:
        raise EmptyAnalysisDataError("No analysis rows found in selected run metrics.")

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing artifacts to %s", output_dir)

    # Original figures
    generate_structural_figure(df, output_dir / "structural_metrics.png")
    generate_performance_figure(df, output_dir / "performance_metrics.png")
    generate_robustness_table(df, output_dir / "robustness_table.tex")

    # Phase E additions
    generate_calibration_scatter(df, output_dir / "calibration_scatter.png")
    generate_distance_degradation_scatter(df, output_dir / "distance_degradation.png")
    generate_density_stratified_figure(df, output_dir / "density_stratified.png")
    generate_distance_regression_table(df, output_dir / "distance_regression.tex")

    # Phase F: Failure mode analysis (needs raw per-task data)
    raw_df = load_raw_task_dataframe(
        run_dirs,
        metrics=["sparsity_ratio", "skeleton_f1", "orientation_accuracy"],
        require_per_task=True,
        skip_non_per_task=not strict,
    )
    if not raw_df.empty:
        classified = classify_failure_modes(raw_df)
        if not classified.empty:
            fractions = failure_mode_fractions(classified)
            generate_failure_mode_bar(fractions, output_dir / "failure_modes.png")

    # Phase G: OOD detection analysis (needs raw per-task data)
    ood_raw_df = load_raw_task_dataframe(
        run_dirs,
        metrics=["edge_entropy", "graph_nll_per_edge", "ne-shd", "ne-sid"],
        require_per_task=True,
        skip_non_per_task=not strict,
    )
    if not ood_raw_df.empty:
        # Entropy histogram (E.3a)
        generate_entropy_histogram(ood_raw_df, output_dir / "entropy_histogram.png")

        # OOD detection AUROC/AUPRC table
        for score in ("edge_entropy", "graph_nll_per_edge"):
            detection_df = compute_ood_detection_metrics(ood_raw_df, score_metric=score)
            if not detection_df.empty:
                generate_ood_detection_table(
                    detection_df,
                    output_dir / f"ood_detection_{score}.tex",
                )

        # Selective prediction Pareto curve (E.3b)
        pareto_df = compute_selective_prediction(ood_raw_df)
        generate_selective_prediction_pareto(
            pareto_df, output_dir / "selective_prediction.png"
        )

    # Phase F (updated): Posterior failure diagnostics from .pt artifacts
    try:
        posterior_df = run_posterior_diagnostics_from_runs(run_dirs)
        if not posterior_df.empty:
            generate_event_probability_bar(
                posterior_df, output_dir / "event_probabilities.png"
            )
            generate_posterior_diagnostic_violins(
                posterior_df, output_dir / "posterior_diagnostics.png"
            )
            log.info("Generated posterior diagnostics from %d tasks", len(posterior_df))
        else:
            log.info("No inference artifacts found; skipping posterior diagnostics.")
    except Exception:
        if strict:
            raise
        log.warning(
            "Posterior diagnostics failed; continuing with remaining artifacts.",
            exc_info=True,
        )

    return df
