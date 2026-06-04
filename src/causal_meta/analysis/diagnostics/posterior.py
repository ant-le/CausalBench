"""Posterior failure diagnostics from cached inference artifacts.

Analyses raw posterior graph samples (not collapsed posterior means) to
characterise *where* the model places probability mass under OOD shift.

Primary outputs
---------------
- **Per-sample diagnostics**: density, density ratio, skeleton F1,
  orientation accuracy, connected-component count for every posterior draw.
- **Posterior event probabilities**: per-task probabilities such as
  P(empty), P(dense), P(skeleton correct & orientation wrong), P(fragmented).
- **Posterior summary statistics**: mean, std, quantiles of diagnostic
  quantities over posterior samples for each task.

The existing threshold-based taxonomy in ``failure_modes.py`` becomes an
optional secondary view derived from these posterior-native diagnostics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import torch

from causal_meta.runners.utils.artifacts import torch_load as _torch_load

log = logging.getLogger(__name__)

# ── Artifact loading ───────────────────────────────────────────────────


def _discover_artifacts(
    run_dir: Path,
    *,
    model_name: str | None = None,
    inference_root: Path | None = None,
    use_model_subdir: bool | None = None,
) -> list[tuple[str, Path]]:
    """Find all cached inference artifacts for a run.

    Supports both cache layouts:

    - run-local: ``<run>/inference/<dataset>/seed_*.pt[.gz]``
    - shared cache: ``<root>/<model>/<dataset>/seed_*.pt[.gz]``

    Args:
        run_dir: Run directory.
        model_name: Optional model identifier for shared-cache discovery.
        inference_root: Optional explicit inference root. If omitted,
            ``<run>/inference`` is used.
        use_model_subdir: Optional layout hint from metadata.
            ``True`` -> only ``<root>/<model>/<dataset>``
            ``False`` -> only ``<root>/<dataset>``
            ``None`` -> probe both.

    Returns:
        List of ``(dataset_key, artifact_path)`` tuples.
    """
    scan_root = inference_root or (run_dir / "inference")
    if not scan_root.is_dir():
        return []

    candidate_bases: list[Path] = []
    if use_model_subdir is True:
        if model_name:
            candidate_bases.append(scan_root / model_name)
    elif use_model_subdir is False:
        candidate_bases.append(scan_root)
    else:
        candidate_bases.append(scan_root)
        if model_name:
            candidate_bases.append(scan_root / model_name)

    discovered: dict[tuple[str, str], tuple[str, Path]] = {}
    for base in candidate_bases:
        if not base.is_dir():
            continue
        for dataset_dir in sorted(base.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset_key = dataset_dir.name

            artifact_paths = sorted(dataset_dir.glob("seed_*.pt")) + sorted(
                dataset_dir.glob("seed_*.pt.gz")
            )
            for artifact_path in artifact_paths:
                key = (dataset_key, str(artifact_path.resolve()))
                discovered[key] = (dataset_key, artifact_path)

    return sorted(discovered.values(), key=lambda item: (item[0], str(item[1])))


def load_posterior_artifacts(
    run_dirs: Sequence[Path],
    *,
    dataset_keys: Sequence[str] | None = None,
    max_tasks_per_family: int | None = None,
) -> pd.DataFrame:
    """Load cached inference artifacts into a DataFrame of posterior samples.

    Each row represents one *task* (one SCM instance) and carries the raw
    posterior graph samples and ground-truth adjacency as tensors.

    Args:
        run_dirs: Run directories containing ``metrics.json`` and ``inference/``.
        dataset_keys: If given, only load artifacts for these dataset families.
        max_tasks_per_family: If given, load at most this many tasks per
            (run, dataset) combination (useful for limiting memory).

    Returns:
        DataFrame with columns:

        - ``RunID``, ``RunDir``, ``Model``, ``DatasetKey`` — identification
        - ``Seed``, ``TaskIdx`` — per-task identifiers
        - ``GraphSamples`` — ``torch.Tensor`` of shape ``(S, N, N)``
          (posterior DAG samples, squeezed from the ``(1, S, N, N)`` on disk)
        - ``TrueAdj`` — ``torch.Tensor`` of shape ``(N, N)``
        - ``NumSamples`` — int, number of posterior samples ``S``
        - ``NNodes`` — int, number of nodes ``N``
    """
    from causal_meta.analysis.utils import _as_mapping

    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir).expanduser().resolve()
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            log.warning("Skipping %s — no metrics.json", run_dir)
            continue

        with open(metrics_path, "r") as f:
            payload = json.load(f)

        metadata = (
            _as_mapping(payload.get("metadata")) if isinstance(payload, Mapping) else {}
        )
        run_id = str(metadata.get("run_id", run_dir.name))
        run_name = str(metadata.get("run_name", run_id))
        model_name = str(metadata.get("model_name", "unknown"))

        inference_root_raw = metadata.get("inference_root")
        inference_root: Path | None = None
        if inference_root_raw is not None and str(inference_root_raw).strip():
            candidate_root = Path(str(inference_root_raw)).expanduser()
            if not candidate_root.is_absolute():
                candidate_root = run_dir / candidate_root
            inference_root = candidate_root

        layout_raw = str(metadata.get("inference_layout", "")).strip().lower()
        if layout_raw == "model_dataset":
            use_model_subdir: bool | None = True
        elif layout_raw == "dataset":
            use_model_subdir = False
        else:
            use_model_subdir = None

        artifacts = _discover_artifacts(
            run_dir,
            model_name=model_name,
            inference_root=inference_root,
            use_model_subdir=use_model_subdir,
        )
        if not artifacts:
            # Backward-compatible fallback to run-local discovery.
            artifacts = _discover_artifacts(run_dir, model_name=model_name)
        if not artifacts:
            log.debug("No inference artifacts in %s", run_dir)
            continue

        # Group by dataset to support max_tasks_per_family
        from collections import defaultdict

        per_dataset: dict[str, list[Path]] = defaultdict(list)
        for dk, path in artifacts:
            if dataset_keys is not None and dk not in dataset_keys:
                continue
            per_dataset[dk].append(path)

        for dataset_key, paths in per_dataset.items():
            if max_tasks_per_family is not None:
                paths = paths[:max_tasks_per_family]

            for path in paths:
                try:
                    artifact = _torch_load(path)
                except Exception:
                    log.warning("Failed to load artifact %s", path, exc_info=True)
                    continue

                graph_samples = artifact.get("graph_samples")
                true_adj = artifact.get("true_adj")
                if graph_samples is None or true_adj is None:
                    log.warning("Artifact %s missing graph_samples or true_adj", path)
                    continue

                # graph_samples on disk: (1, S, N, N) → squeeze batch dim → (S, N, N)
                if graph_samples.ndim == 4 and graph_samples.shape[0] == 1:
                    graph_samples = graph_samples.squeeze(0)
                elif graph_samples.ndim == 3:
                    pass  # already (S, N, N)
                else:
                    log.warning(
                        "Unexpected graph_samples shape %s in %s; skipping",
                        graph_samples.shape,
                        path,
                    )
                    continue

                # Ensure binary
                graph_samples = (graph_samples > 0.5).to(torch.uint8)
                true_adj = (true_adj > 0.5).to(torch.uint8)

                seed = artifact.get("seed", -1)
                idx = artifact.get("idx", -1)

                rows.append(
                    {
                        "RunID": run_id,
                        "RunName": run_name,
                        "RunDir": str(run_dir),
                        "Model": model_name,
                        "DatasetKey": dataset_key,
                        "Seed": int(seed),
                        "TaskIdx": int(idx),
                        "GraphSamples": graph_samples,
                        "TrueAdj": true_adj,
                        "NumSamples": graph_samples.shape[0],
                        "NNodes": graph_samples.shape[-1],
                    }
                )

    df = pd.DataFrame(rows)
    log.info(
        "Loaded %d posterior artifact tasks from %d run(s)",
        len(df),
        len(run_dirs),
    )
    return df


# ── Per-sample diagnostic primitives ──────────────────────────────────


def _graph_density(adj: torch.Tensor) -> torch.Tensor:
    """Edge density for one or more adjacency matrices.

    Args:
        adj: ``(*, N, N)`` binary adjacency.

    Returns:
        ``(*)`` density values in [0, 1].
    """
    n = adj.shape[-1]
    n_potential = n * (n - 1)  # off-diagonal entries for a DAG
    if n_potential == 0:
        return torch.zeros(adj.shape[:-2], device=adj.device)

    adj_f = adj.float()
    off_diag_edges = adj_f.sum(dim=(-1, -2)) - torch.diagonal(
        adj_f, dim1=-2, dim2=-1
    ).sum(dim=-1)
    return off_diag_edges / n_potential


def _skeleton(adj: torch.Tensor) -> torch.Tensor:
    """Undirected skeleton: edge (i,j) exists if adj[i,j] or adj[j,i].

    Args:
        adj: ``(*, N, N)`` binary adjacency.

    Returns:
        ``(*, N, N)`` binary symmetric skeleton.
    """
    return torch.clamp(adj.float() + adj.float().transpose(-1, -2), max=1.0)


def _connected_components(adj: torch.Tensor) -> int:
    """Number of connected components in a single undirected graph.

    Args:
        adj: ``(N, N)`` binary adjacency (interpreted as undirected).

    Returns:
        Number of connected components.
    """
    n = int(adj.shape[0])
    if n == 0:
        return 0

    skel = torch.clamp(adj.float() + adj.float().T, max=1.0)
    visited = torch.zeros(n, dtype=torch.bool, device=adj.device)
    n_components = 0

    for start in range(n):
        if visited[start]:
            continue
        n_components += 1
        # BFS
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            neighbours = (
                torch.nonzero(skel[node] > 0.5, as_tuple=False).flatten().tolist()
            )
            for nb in neighbours:
                nb_int = int(nb)
                if not visited[nb_int]:
                    visited[nb_int] = True
                    queue.append(nb_int)

    return n_components


def compute_per_sample_diagnostics(
    graph_samples: torch.Tensor,
    true_adj: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Compute diagnostics for each posterior sample individually.

    Args:
        graph_samples: ``(S, N, N)`` binary posterior DAG samples.
        true_adj: ``(N, N)`` binary ground-truth adjacency.
        eps: Small constant to avoid division by zero in density ratio.

    Returns:
        Dict mapping diagnostic names to 1-D arrays of length ``S``:

        - ``density``: edge density of each sample
        - ``density_ratio``: sample density / true density
        - ``skeleton_f1``: F1 of the undirected skeleton
        - ``orientation_accuracy``: fraction of common skeleton edges with
          correct direction
        - ``connected_components``: number of connected components in the
          predicted skeleton
    """
    s, n, _ = graph_samples.shape
    samples = graph_samples.float()
    truth = true_adj.float()

    # ── Density ────────────────────────────────────────────────────────
    densities = _graph_density(samples).cpu().numpy()  # (S,)
    true_density = _graph_density(truth).item()
    density_ratios = densities / (true_density + eps)

    # ── Skeleton F1 ───────────────────────────────────────────────────
    true_skel = _skeleton(truth)  # (N, N)
    idx = torch.triu_indices(n, n, offset=1)
    true_skel_ut = true_skel[idx[0], idx[1]]  # (E,)

    skeleton_f1s = np.empty(s, dtype=np.float64)
    orient_accs = np.empty(s, dtype=np.float64)
    cc_counts = np.empty(s, dtype=np.int32)

    for i in range(s):
        p = samples[i]  # (N, N)
        pred_skel = _skeleton(p)
        pred_skel_ut = pred_skel[idx[0], idx[1]]

        # Skeleton F1
        tp = (pred_skel_ut * true_skel_ut).sum()
        fp = (pred_skel_ut * (1 - true_skel_ut)).sum()
        fn = ((1 - pred_skel_ut) * true_skel_ut).sum()
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom).item() if denom > 0 else 1.0
        skeleton_f1s[i] = f1

        # Orientation accuracy (on common skeleton edges)
        common = pred_skel_ut * true_skel_ut
        total_common = common.sum().item()
        if total_common > 0:
            dir_match = (truth[idx[0], idx[1]] == p[idx[0], idx[1]]).float() * (
                truth[idx[1], idx[0]] == p[idx[1], idx[0]]
            ).float()
            correct_dir = (common * dir_match).sum().item()
            orient_accs[i] = correct_dir / total_common
        else:
            orient_accs[i] = 1.0

        # Connected components
        cc_counts[i] = _connected_components(p)

    return {
        "density": densities,
        "density_ratio": density_ratios,
        "skeleton_f1": skeleton_f1s,
        "orientation_accuracy": orient_accs,
        "connected_components": cc_counts,
    }


# ── Posterior event probabilities ─────────────────────────────────────

# Event thresholds (can be tuned)
EMPTY_DENSITY_THRESHOLD: float = 0.01
"""A sample is ``empty`` if its density is below this."""

DENSE_RATIO_THRESHOLD: float = 2.0
"""A sample is ``dense`` if density_ratio > this."""

SKELETON_CORRECT_F1_THRESHOLD: float = 0.8
"""Skeleton is "correct" if skeleton_f1 >= this."""

ORIENTATION_WRONG_THRESHOLD: float = 0.5
"""Orientation is "wrong" if orientation_accuracy < this."""


def compute_event_probabilities(
    diagnostics: dict[str, np.ndarray],
    true_adj: torch.Tensor,
) -> dict[str, float | bool]:
    """Compute posterior event probabilities from per-sample diagnostics.

    Args:
        diagnostics: Output of :func:`compute_per_sample_diagnostics`.
        true_adj: Ground-truth adjacency (needed for fragmentation check).

    Returns:
        Dict of event name → probability (fraction of samples):

        - ``p_empty``: P(density < threshold)
        - ``p_dense``: P(density_ratio > threshold)
        - ``p_skeleton_correct_orient_wrong``: P(skeleton_f1 >= 0.8 AND
          orientation_accuracy < 0.5)
        - ``p_fragmented``: P(predicted graph is fragmented | truth connected)
        - ``truth_connected``: whether the truth graph is connected
    """
    s = len(diagnostics["density"])
    true_cc = _connected_components(true_adj)
    truth_connected = true_cc == 1

    if s == 0:
        return {
            "p_empty": 0.0,
            "p_dense": 0.0,
            "p_skeleton_correct_orient_wrong": 0.0,
            "p_fragmented": 0.0 if truth_connected else float("nan"),
            "truth_connected": truth_connected,
        }

    density = diagnostics["density"]
    density_ratio = diagnostics["density_ratio"]
    skel_f1 = diagnostics["skeleton_f1"]
    orient_acc = diagnostics["orientation_accuracy"]
    cc = diagnostics["connected_components"]

    p_empty = float(np.mean(density < EMPTY_DENSITY_THRESHOLD))
    p_dense = float(np.mean(density_ratio > DENSE_RATIO_THRESHOLD))
    p_skel_orient = float(
        np.mean(
            (skel_f1 >= SKELETON_CORRECT_F1_THRESHOLD)
            & (orient_acc < ORIENTATION_WRONG_THRESHOLD)
        )
    )
    if truth_connected:
        p_fragmented = float(np.mean(cc > 1))
    else:
        p_fragmented = float("nan")

    return {
        "p_empty": p_empty,
        "p_dense": p_dense,
        "p_skeleton_correct_orient_wrong": p_skel_orient,
        "p_fragmented": p_fragmented,
        "truth_connected": truth_connected,
    }


# ── Posterior summary statistics ──────────────────────────────────────


def compute_posterior_summary(
    diagnostics: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compute summary statistics over posterior samples for each diagnostic.

    Args:
        diagnostics: Output of :func:`compute_per_sample_diagnostics`.

    Returns:
        Nested dict ``{diagnostic_name: {stat_name: value}}``.
        Statistics: ``mean``, ``std``, ``median``, ``q25``, ``q75``,
        ``q05``, ``q95``.
    """
    summaries: dict[str, dict[str, float]] = {}
    for name, values in diagnostics.items():
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == 0:
            summaries[name] = {
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
                "q05": float("nan"),
                "q95": float("nan"),
            }
        else:
            summaries[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
                "q05": float(np.percentile(arr, 5)),
                "q95": float(np.percentile(arr, 95)),
            }
    return summaries


# ── High-level pipeline ──────────────────────────────────────────────


def run_posterior_diagnostics(
    artifacts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run full posterior diagnostics on a loaded artifacts DataFrame.

    Args:
        artifacts_df: DataFrame from :func:`load_posterior_artifacts` with
            ``GraphSamples`` and ``TrueAdj`` columns.

    Returns:
        DataFrame with one row per task containing:

        - Identification columns: ``RunID``, ``Model``, ``DatasetKey``,
          ``Seed``, ``TaskIdx``
        - Event probabilities: ``p_empty``, ``p_dense``,
          ``p_skeleton_correct_orient_wrong``, ``p_fragmented``
        - Summary statistics for each diagnostic (e.g.
          ``density_ratio_mean``, ``density_ratio_q05``, …)
    """
    if artifacts_df.empty:
        return pd.DataFrame()

    required_cols = {"GraphSamples", "TrueAdj"}
    if not required_cols.issubset(artifacts_df.columns):
        log.warning(
            "Missing required columns: %s", required_cols - set(artifacts_df.columns)
        )
        return pd.DataFrame()

    result_rows: list[dict[str, Any]] = []

    for _, row in artifacts_df.iterrows():
        graph_samples_raw = row.get("GraphSamples")
        true_adj_raw = row.get("TrueAdj")
        if not torch.is_tensor(graph_samples_raw) or not torch.is_tensor(true_adj_raw):
            continue

        graph_samples = cast(torch.Tensor, graph_samples_raw)
        true_adj = cast(torch.Tensor, true_adj_raw)

        # Per-sample diagnostics
        diag = compute_per_sample_diagnostics(graph_samples, true_adj)

        # Event probabilities
        events = compute_event_probabilities(diag, true_adj)
        truth_connected = bool(events.pop("truth_connected", True))

        # Summary statistics
        summary = compute_posterior_summary(diag)

        # Build row
        result: dict[str, Any] = {
            "RunID": row.get("RunID"),
            "RunDir": row.get("RunDir"),
            "Model": row.get("Model"),
            "DatasetKey": row.get("DatasetKey"),
            "Seed": row.get("Seed"),
            "TaskIdx": row.get("TaskIdx"),
            "NumSamples": row.get("NumSamples"),
            "NNodes": row.get("NNodes"),
            "TruthConnected": truth_connected,
            "ConnectedTaskCount": int(truth_connected),
        }

        # Event probabilities
        result.update(events)

        # Flatten summary stats: {diagnostic}_{stat}
        for diag_name, stats in summary.items():
            for stat_name, value in stats.items():
                result[f"{diag_name}_{stat_name}"] = value

        result_rows.append(result)

    return pd.DataFrame(result_rows)


def run_posterior_diagnostics_from_runs(
    run_dirs: Sequence[Path],
    *,
    dataset_keys: Sequence[str] | None = None,
    max_tasks_per_family: int | None = None,
) -> pd.DataFrame:
    """End-to-end pipeline: load artifacts and compute diagnostics.

    Convenience wrapper combining :func:`load_posterior_artifacts` and
    :func:`run_posterior_diagnostics`.

    Args:
        run_dirs: Run directories containing ``metrics.json`` and ``inference/``.
        dataset_keys: Optional filter for dataset families.
        max_tasks_per_family: Optional limit on tasks per family.

    Returns:
        Diagnostics DataFrame (see :func:`run_posterior_diagnostics`).
    """
    artifacts_df = load_posterior_artifacts(
        run_dirs,
        dataset_keys=dataset_keys,
        max_tasks_per_family=max_tasks_per_family,
    )
    return run_posterior_diagnostics(artifacts_df)
