from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_meta.analysis.utils import (
    generate_all_artifacts_from_runs,
    load_raw_task_dataframe,
    load_runs_dataframe,
    RawGranularityError,
    resolve_run_directories,
)


def _write_metrics(
    run_dir: Path,
    *,
    run_id: str,
    run_name: str,
    model_name: str,
    dataset_key: str,
    mean_offset: float,
    raw_granularity: str | None = None,
    batch_size_test: int | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "run_id": run_id,
        "run_name": run_name,
        "model_name": model_name,
        "output_dir": str(run_dir),
    }
    if raw_granularity is not None:
        metadata["raw_granularity"] = raw_granularity
    if batch_size_test is not None:
        metadata["batch_size_test"] = int(batch_size_test)
    payload = {
        "metadata": metadata,
        "summary": {
            dataset_key: {
                "e-shd_mean": 10.0 + mean_offset,
                "e-shd_sem": 1.0,
                "e-shd_std": 2.0,
                "e-sid_mean": 20.0 + mean_offset,
                "e-sid_sem": 2.0,
                "e-sid_std": 3.0,
                "ne-shd_mean": 0.1 + mean_offset / 100.0,
                "ne-shd_sem": 0.01,
                "ne-shd_std": 0.02,
                "ne-sid_mean": 0.2 + mean_offset / 100.0,
                "ne-sid_sem": 0.02,
                "ne-sid_std": 0.03,
                "graph_nll_per_edge_mean": 0.3 + mean_offset / 100.0,
                "graph_nll_per_edge_sem": 0.03,
                "graph_nll_per_edge_std": 0.04,
                "auc_mean": 0.7,
                "auc_sem": 0.01,
                "auc_std": 0.02,
            }
        },
        "raw": {dataset_key: {"e-shd": [10.0 + mean_offset]}},
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


def test_resolve_run_directories_from_ids(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "rq1_20260317_avici"
    _write_metrics(
        run_dir,
        run_id="rq1_20260317_avici",
        run_name="rq1",
        model_name="avici",
        dataset_key="id_linear_er20",
        mean_offset=0.0,
    )

    resolved = resolve_run_directories(runs_root=runs_root, run_ids=[run_dir.name])
    assert resolved == [run_dir.resolve()]


def test_resolve_run_directories_discovers_metrics_files(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_a = runs_root / "run_a"
    run_b = runs_root / "nested" / "run_b"
    _write_metrics(
        run_a,
        run_id="run_a",
        run_name="A",
        model_name="bcnp",
        dataset_key="id_linear_er20",
        mean_offset=0.0,
    )
    _write_metrics(
        run_b,
        run_id="run_b",
        run_name="B",
        model_name="dibs",
        dataset_key="ood_mech_periodic_er40",
        mean_offset=2.0,
    )

    resolved = resolve_run_directories(runs_root=runs_root)
    resolved_set = {path.resolve() for path in resolved}
    assert resolved_set == {run_a.resolve(), run_b.resolve()}


def test_load_runs_dataframe_maps_model_and_dataset_names(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "rq1_20260317_avici"
    _write_metrics(
        run_dir,
        run_id="rq1_20260317_avici",
        run_name="rq1",
        model_name="avici",
        dataset_key="id_linear_er20",
        mean_offset=0.0,
    )

    df = load_runs_dataframe([run_dir])
    assert not df.empty

    shd_rows = df[df["Metric"] == "e-shd"]
    assert not shd_rows.empty
    first = shd_rows.iloc[0]
    assert first["RunID"] == "rq1_20260317_avici"
    assert first["Model"] == "AviCi"
    assert first["Dataset"] == "ID Linear ER-20"
    assert float(first["Mean"]) == 10.0


def test_generate_all_artifacts_from_runs_writes_expected_files(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_a = runs_root / "rq1_20260317_avici"
    run_b = runs_root / "rq1_20260317_bcnp"
    _write_metrics(
        run_a,
        run_id="rq1_20260317_avici",
        run_name="rq1",
        model_name="avici",
        dataset_key="id_linear_er20",
        mean_offset=0.0,
    )
    _write_metrics(
        run_b,
        run_id="rq1_20260317_bcnp",
        run_name="rq1",
        model_name="bcnp",
        dataset_key="ood_mech_periodic_er40",
        mean_offset=1.0,
    )

    output_dir = tmp_path / "graphics"
    generate_all_artifacts_from_runs([run_a, run_b], output_dir)

    assert (output_dir / "structural_metrics.png").exists()
    assert (output_dir / "performance_metrics.png").exists()
    assert (output_dir / "robustness_table.tex").exists()


def _write_metrics_with_enrichment(
    run_dir: Path,
    *,
    run_id: str,
    model_name: str,
    dataset_key: str,
) -> None:
    """Write a metrics.json that includes family_metadata and distances (Phase C+D)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "run_id": run_id,
            "run_name": run_id,
            "model_name": model_name,
            "output_dir": str(run_dir),
        },
        "family_metadata": {
            dataset_key: {
                "n_nodes": 20,
                "graph_type": "er",
                "mech_type": "linear",
                "sparsity_param": 0.0526,
            }
        },
        "distances": {
            dataset_key: {
                "spectral": 0.42,
                "kl_degree": 1.73,
                "mechanism": 0.31,
            }
        },
        "summary": {
            dataset_key: {
                "e-shd_mean": 15.0,
                "e-shd_sem": 1.5,
                "e-shd_std": 3.0,
                "ne-shd_mean": 0.15,
                "ne-shd_sem": 0.015,
                "ne-shd_std": 0.03,
                "ne-sid_mean": 0.25,
                "ne-sid_sem": 0.025,
                "ne-sid_std": 0.04,
                "graph_nll_per_edge_mean": 0.35,
                "graph_nll_per_edge_sem": 0.035,
                "graph_nll_per_edge_std": 0.05,
            }
        },
        "raw": {dataset_key: {"e-shd": [15.0]}},
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


def test_load_runs_dataframe_includes_family_metadata_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_enriched"
    _write_metrics_with_enrichment(
        run_dir,
        run_id="run_enriched",
        model_name="avici",
        dataset_key="id_linear_er20",
    )

    df = load_runs_dataframe([run_dir])
    assert not df.empty

    row = df.iloc[0]
    assert row["GraphType"] == "er"
    assert row["MechType"] == "linear"
    assert row["NNodes"] == 20
    assert float(row["SparsityParam"]) == 0.0526


def test_load_runs_dataframe_includes_distance_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_dist"
    _write_metrics_with_enrichment(
        run_dir,
        run_id="run_dist",
        model_name="bcnp",
        dataset_key="id_linear_er20",
    )

    df = load_runs_dataframe([run_dir])
    assert not df.empty

    row = df.iloc[0]
    assert float(row["SpectralDist"]) == 0.42
    assert float(row["KLDegreeDist"]) == 1.73
    assert float(row["MechanismDist"]) == 0.31


def test_load_runs_dataframe_handles_missing_enrichment(tmp_path: Path) -> None:
    """Older metrics.json without family_metadata/distances still loads cleanly."""
    run_dir = tmp_path / "runs" / "run_legacy"
    _write_metrics(
        run_dir,
        run_id="run_legacy",
        run_name="legacy",
        model_name="dibs",
        dataset_key="ood_mech_periodic_er40",
        mean_offset=0.0,
    )

    df = load_runs_dataframe([run_dir])
    assert not df.empty

    row = df.iloc[0]
    # Columns should exist but have empty/NaN defaults
    assert row["GraphType"] == ""
    assert row["MechType"] == ""
    assert row["NNodes"] is None
    assert row["SparsityParam"] is None
    import math

    assert math.isnan(float(row["SpectralDist"]))
    assert math.isnan(float(row["KLDegreeDist"]))
    assert math.isnan(float(row["MechanismDist"]))


def test_load_raw_task_dataframe_rejects_non_per_task_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_batch_raw"
    _write_metrics(
        run_dir,
        run_id="run_batch_raw",
        run_name="batch_raw",
        model_name="bcnp",
        dataset_key="id_linear_er20",
        mean_offset=0.0,
        raw_granularity="per_batch",
    )

    with pytest.raises(RawGranularityError):
        load_raw_task_dataframe([run_dir], require_per_task=True)


def test_generate_all_artifacts_strict_fails_on_non_per_task_raw(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "run_batch_raw"
    _write_metrics(
        run_dir,
        run_id="run_batch_raw",
        run_name="batch_raw",
        model_name="bcnp",
        dataset_key="id_linear_er20",
        mean_offset=0.0,
        raw_granularity="per_batch",
    )

    with pytest.raises(RawGranularityError):
        generate_all_artifacts_from_runs([run_dir], tmp_path / "graphics", strict=True)


def test_load_raw_task_dataframe_legacy_explicit_defaults_to_per_task(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "run_legacy_dibs"
    _write_metrics(
        run_dir,
        run_id="run_legacy_dibs",
        run_name="legacy_dibs",
        model_name="dibs",
        dataset_key="id_linear_er20",
        mean_offset=0.0,
        batch_size_test=8,
    )

    raw_df = load_raw_task_dataframe([run_dir], require_per_task=True)
    assert not raw_df.empty
    assert set(raw_df["ModelKey"]) == {"dibs"}
