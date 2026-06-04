"""Smoke tests for the paper plausibility comparison module (Appendix F)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# ── Helpers ────────────────────────────────────────────────────────────


def _write_thesis_metrics(
    model_dir: Path, families: dict[str, dict[str, float]]
) -> None:
    """Write a minimal metrics.json under *model_dir* with the given families."""
    model_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metadata": {"model_name": model_dir.name},
        "summary": families,
    }
    (model_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


def _write_model_yaml(config_dir: Path, model: str, content: str) -> None:
    """Write a minimal model YAML under ``config_dir/model/<model>.yaml``."""
    (config_dir / "model").mkdir(parents=True, exist_ok=True)
    (config_dir / "model" / f"{model}.yaml").write_text(content)


_FAMILY_KEY = "id_linear_er20_d20_n500"

_FAMILY_METRICS = {
    "auc_mean": 0.85,
    "auc_std": 0.02,
    "e-shd_mean": 28.0,
    "e-shd_std": 1.5,
    "e-edgef1_mean": 0.35,
    "e-edgef1_std": 0.04,
}


def _build_thesis_runs(tmp_path: Path) -> Path:
    """Create a ``thesis_runs/`` tree with all 4 models and one overlapping family."""
    runs = tmp_path / "thesis_runs"
    for model in ("avici", "bcnp", "dibs", "bayesdag"):
        _write_thesis_metrics(runs / model, {_FAMILY_KEY: _FAMILY_METRICS})
    # Source AVICI pretrained checkpoint results.
    _write_thesis_metrics(
        runs / "avici_pretrained_scm-v0", {_FAMILY_KEY: _FAMILY_METRICS}
    )
    return runs


def _build_configs(tmp_path: Path) -> Path:
    """Create minimal model YAML configs matching paper_reference keys."""
    cfgs = tmp_path / "configs"
    _write_model_yaml(
        cfgs, "avici", "dim: 128\nlayers: 8\nnum_heads: 8\ndropout: 0.1\n"
    )
    _write_model_yaml(cfgs, "bcnp", "d_model: 512\nnhead: 8\ndropout: 0.1\n")
    _write_model_yaml(cfgs, "dibs", "steps: 3000\nn_particles: 32\nmode: nonlinear\n")
    _write_model_yaml(cfgs, "bayesdag", "num_chains: 10\nmax_epochs: 400\n")
    return cfgs


# ── reference_data tests ──────────────────────────────────────────────


class TestReferenceData:
    """Basic checks that the bundled JSON loads and accessors work."""

    def test_load_reference_is_dict(self) -> None:
        from causal_meta.analysis.paper_comparison.reference_data import \
            load_reference

        ref = load_reference()
        assert isinstance(ref, dict)
        assert "cross_model_comparison" in ref
        assert "source_papers" in ref

    def test_comparable_families_has_nine_entries(self) -> None:
        from causal_meta.analysis.paper_comparison.reference_data import \
            COMPARABLE_FAMILIES

        assert len(COMPARABLE_FAMILIES) == 9

    def test_cross_model_paper_values_structure(self) -> None:
        from causal_meta.analysis.paper_comparison.reference_data import (
            COMPARABLE_FAMILIES, cross_model_paper_values)

        vals = cross_model_paper_values()
        # Should have exactly the 9 families.
        assert set(vals.keys()) == set(COMPARABLE_FAMILIES.keys())
        # Each family should have 4 models.
        for fam_key, models_dict in vals.items():
            assert set(models_dict.keys()) == {"avici", "bcnp", "dibs", "bayesdag"}
            for model, metrics in models_dict.items():
                for m in ("auc", "e_shd", "e_edgef1"):
                    assert m in metrics, f"Missing {m} for {model} in {fam_key}"
                    assert len(metrics[m]) == 2  # [mean, std]

    def test_source_paper_configs_has_all_models(self) -> None:
        from causal_meta.analysis.paper_comparison.reference_data import \
            source_paper_configs

        cfgs = source_paper_configs()
        assert set(cfgs.keys()) == {"avici", "bcnp", "dibs", "bayesdag"}

    def test_source_paper_notes_returns_strings(self) -> None:
        from causal_meta.analysis.paper_comparison.reference_data import \
            source_paper_notes

        notes = source_paper_notes()
        for model, note in notes.items():
            assert isinstance(note, str)
            assert len(note) > 0, f"Empty note for {model}"


# ── comparison tests ──────────────────────────────────────────────────


class TestCrossModelDataframe:
    """Verify ``build_cross_model_dataframe`` output shape and content."""

    def test_basic_shape(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_cross_model_dataframe

        runs = _build_thesis_runs(tmp_path)
        df = build_cross_model_dataframe(runs)

        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "model",
            "family",
            "family_label",
            "metric",
            "paper_value",
            "paper_std",
            "thesis_value",
            "thesis_std",
            "delta",
        }
        assert expected_cols.issubset(set(df.columns))
        # 4 models × 9 families × 3 metrics = 108 rows.
        assert len(df) == 4 * 9 * 3

    def test_thesis_values_populated_for_matching_family(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_cross_model_dataframe

        runs = _build_thesis_runs(tmp_path)
        df = build_cross_model_dataframe(runs)

        sub = df[(df["family"] == _FAMILY_KEY) & (df["metric"] == "auc")]
        assert len(sub) == 4
        # All 4 models should have thesis_value = 0.85 (our fixture).
        assert (sub["thesis_value"] == 0.85).all()

    def test_delta_computed(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_cross_model_dataframe

        runs = _build_thesis_runs(tmp_path)
        df = build_cross_model_dataframe(runs)

        sub = df[(df["family"] == _FAMILY_KEY) & (df["metric"] == "auc")]
        # delta should be thesis - paper, non-null for the matching family.
        assert sub["delta"].notna().all()

    def test_missing_family_yields_none_thesis_value(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_cross_model_dataframe

        # Only one family in the fixture; the other 8 should have null thesis values.
        runs = _build_thesis_runs(tmp_path)
        df = build_cross_model_dataframe(runs)
        other = df[df["family"] != _FAMILY_KEY]
        assert other["thesis_value"].isna().all()


class TestHyperparamComparison:
    """Verify ``build_hyperparam_comparison`` output."""

    def test_returns_all_models(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_hyperparam_comparison

        cfgs = _build_configs(tmp_path)
        hp = build_hyperparam_comparison(cfgs)
        assert set(hp.keys()) == {"avici", "bcnp", "dibs", "bayesdag"}

    def test_dataframe_columns(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_hyperparam_comparison

        cfgs = _build_configs(tmp_path)
        hp = build_hyperparam_comparison(cfgs)
        for model, df in hp.items():
            assert isinstance(df, pd.DataFrame)
            assert set(df.columns) == {"parameter", "paper_value", "our_value"}

    def test_our_value_populated_from_overrides(self, tmp_path: Path) -> None:
        """Overrides in paper_reference.json should fill N/A values."""
        from causal_meta.analysis.paper_comparison.comparison import \
            build_hyperparam_comparison

        cfgs = _build_configs(tmp_path)
        hp = build_hyperparam_comparison(cfgs)
        for _, df in hp.items():
            # Every row should have a non-empty our_value string.
            assert (df["our_value"].astype(str) != "").all()


class TestAvici3WayDataframe:
    """Verify ``build_avici_3way_dataframe`` output shape and content."""

    def test_basic_shape(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_avici_3way_dataframe

        runs = _build_thesis_runs(tmp_path)
        df = build_avici_3way_dataframe(runs)

        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "family",
            "family_label",
            "metric",
            "bcnp_paper_value",
            "bcnp_paper_std",
            "thesis_value",
            "thesis_std",
            "source_value",
            "source_std",
        }
        assert expected_cols.issubset(set(df.columns))
        # 9 families × 3 metrics = 27 rows.
        assert len(df) == 9 * 3

    def test_values_populated_for_matching_family(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_avici_3way_dataframe

        runs = _build_thesis_runs(tmp_path)
        df = build_avici_3way_dataframe(runs)

        sub = df[(df["family"] == _FAMILY_KEY) & (df["metric"] == "auc")]
        assert len(sub) == 1
        row = sub.iloc[0]
        # Paper value from JSON reference.
        assert row["bcnp_paper_value"] == 0.89
        # Thesis and source values from our fixture.
        assert row["thesis_value"] == 0.85
        assert row["source_value"] == 0.85


# ── tables tests ──────────────────────────────────────────────────────


class TestTableGeneration:
    """Smoke tests that LaTeX table generators write files without error."""

    def test_cross_model_table_writes_file(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_cross_model_dataframe
        from causal_meta.analysis.paper_comparison.tables import \
            generate_cross_model_table

        runs = _build_thesis_runs(tmp_path)
        df = build_cross_model_dataframe(runs)
        out = tmp_path / "plausibility_auc.tex"
        generate_cross_model_table(df, metric="auc", output_path=out)
        assert out.exists()
        content = out.read_text()
        assert r"\begin{tabular}" in content
        assert r"\end{tabular}" in content
        assert r"\textit{Paper}" in content
        assert r"\textit{Ours}" in content

    def test_cross_model_table_all_metrics(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_cross_model_dataframe
        from causal_meta.analysis.paper_comparison.tables import \
            generate_cross_model_table

        runs = _build_thesis_runs(tmp_path)
        df = build_cross_model_dataframe(runs)
        for metric in ("auc", "e_shd", "e_edgef1"):
            out = tmp_path / f"plausibility_{metric}.tex"
            generate_cross_model_table(df, metric=metric, output_path=out)
            assert out.exists()

    def test_hyperparam_table_writes_file(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_hyperparam_comparison
        from causal_meta.analysis.paper_comparison.tables import \
            generate_hyperparam_table

        cfgs = _build_configs(tmp_path)
        hp = build_hyperparam_comparison(cfgs)
        for model, hp_df in hp.items():
            out = tmp_path / f"hp_{model}.tex"
            generate_hyperparam_table(hp_df, model=model, output_path=out)
            assert out.exists()
            content = out.read_text()
            assert r"\begin{tabular}" in content

    def test_source_notes_table_writes_file(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.tables import \
            generate_source_notes_table

        out = tmp_path / "source_notes.tex"
        generate_source_notes_table(out)
        assert out.exists()
        content = out.read_text()
        assert r"\begin{tabularx}" in content
        assert "AviCi" in content or "BCNP" in content

    def test_avici_3way_table_writes_file(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_avici_3way_dataframe
        from causal_meta.analysis.paper_comparison.tables import \
            generate_avici_3way_table

        runs = _build_thesis_runs(tmp_path)
        df = build_avici_3way_dataframe(runs)
        out = tmp_path / "avici_3way_auc.tex"
        generate_avici_3way_table(df, metric="auc", output_path=out)
        assert out.exists()
        content = out.read_text()
        assert r"\begin{tabular}" in content
        assert r"\end{tabular}" in content
        assert "BCNP paper re-impl." in content

    def test_avici_3way_table_all_metrics(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison.comparison import \
            build_avici_3way_dataframe
        from causal_meta.analysis.paper_comparison.tables import \
            generate_avici_3way_table

        runs = _build_thesis_runs(tmp_path)
        df = build_avici_3way_dataframe(runs)
        for metric in ("auc", "e_shd", "e_edgef1"):
            out = tmp_path / f"avici_3way_{metric}.tex"
            generate_avici_3way_table(df, metric=metric, output_path=out)
            assert out.exists()


# ── Integration test ──────────────────────────────────────────────────


class TestGeneratePaperComparison:
    """Smoke test for the top-level ``generate_paper_comparison`` entry point."""

    def test_end_to_end_creates_artifacts(self, tmp_path: Path) -> None:
        from causal_meta.analysis.paper_comparison import \
            generate_paper_comparison

        runs = _build_thesis_runs(tmp_path)
        thesis = tmp_path / "thesis"
        cfgs = _build_configs(tmp_path)

        generate_paper_comparison(
            thesis_runs_root=runs,
            thesis_root=thesis,
            configs_root=cfgs,
        )

        appendix_dir = thesis / "graphics" / "F_ReproducibilityArtifacts"
        assert appendix_dir.is_dir()

        # Expect: 3 cross-model + 3 AVICI 3-way + 4 HP + 1 notes = 11 files.
        tex_files = sorted(appendix_dir.glob("*.tex"))
        assert (
            len(tex_files) == 11
        ), f"Expected 11 .tex files, got: {[f.name for f in tex_files]}"

        expected_stems = {
            "plausibility_auc",
            "plausibility_e_shd",
            "plausibility_e_edgef1",
            "avici_3way_auc",
            "avici_3way_e_shd",
            "avici_3way_e_edgef1",
            "hp_comparison_avici",
            "hp_comparison_bcnp",
            "hp_comparison_dibs",
            "hp_comparison_bayesdag",
            "source_paper_notes",
        }
        actual_stems = {f.stem for f in tex_files}
        assert actual_stems == expected_stems
