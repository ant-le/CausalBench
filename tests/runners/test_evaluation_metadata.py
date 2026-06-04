"""Tests for Phase C (family_metadata) and Phase D (distances) in evaluation.py."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.datasets.generators.configs import (
    ErdosRenyiConfig,
    FamilyConfig,
    GPMechanismConfig,
    LinearMechanismConfig,
    MLPMechanismConfig,
    PeriodicMechanismConfig,
    SBMConfig,
    ScaleFreeConfig,
)
from causal_meta.runners.tasks.evaluation import (
    _build_family_metadata,
    _extract_graph_type,
    _extract_mech_type,
    _extract_sparsity_param,
)


# ── _extract_graph_type ──────────────────────────────────────────────────


class TestExtractGraphType:
    def test_erdos_renyi_config(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(sparsity=0.05),
            mech_cfg=LinearMechanismConfig(),
        )
        assert "er" in _extract_graph_type(fc).lower()

    def test_scale_free_config(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ScaleFreeConfig(m=2),
            mech_cfg=LinearMechanismConfig(),
        )
        assert "sf" in _extract_graph_type(fc).lower()

    def test_sbm_config(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=SBMConfig(n_blocks=4, p_intra=0.5, p_inter=0.01),
            mech_cfg=LinearMechanismConfig(),
        )
        assert "sbm" in _extract_graph_type(fc).lower()

    def test_unknown_config_returns_class_heuristic_or_unknown(self) -> None:
        """An arbitrary object with no 'type' attr falls back to class name or 'unknown'."""
        mock_graph = MagicMock(spec=[])
        mock_graph.__class__ = type("FooBarConfig", (), {})
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=mock_graph,
            mech_cfg=LinearMechanismConfig(),
        )
        result = _extract_graph_type(fc)
        assert isinstance(result, str)


# ── _extract_mech_type ───────────────────────────────────────────────────


class TestExtractMechType:
    def test_linear(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(),
            mech_cfg=LinearMechanismConfig(),
        )
        assert "linear" in _extract_mech_type(fc).lower()

    def test_mlp(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(),
            mech_cfg=MLPMechanismConfig(),
        )
        assert "mlp" in _extract_mech_type(fc).lower()

    def test_gp(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(),
            mech_cfg=GPMechanismConfig(),
        )
        assert "gp" in _extract_mech_type(fc).lower()

    def test_periodic(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(),
            mech_cfg=PeriodicMechanismConfig(),
        )
        assert "periodic" in _extract_mech_type(fc).lower()


# ── _extract_sparsity_param ──────────────────────────────────────────────


class TestExtractSparsityParam:
    def test_er_with_sparsity(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(sparsity=0.05),
            mech_cfg=LinearMechanismConfig(),
        )
        assert _extract_sparsity_param(fc) == pytest.approx(0.05)

    def test_er_with_edge_prob(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(edge_prob=0.3),
            mech_cfg=LinearMechanismConfig(),
        )
        assert _extract_sparsity_param(fc) == pytest.approx(0.3)

    def test_scale_free_returns_m(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ScaleFreeConfig(m=3),
            mech_cfg=LinearMechanismConfig(),
        )
        assert _extract_sparsity_param(fc) == pytest.approx(3.0)

    def test_sbm_returns_none(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=SBMConfig(n_blocks=4, p_intra=0.5, p_inter=0.01),
            mech_cfg=LinearMechanismConfig(),
        )
        assert _extract_sparsity_param(fc) is None


# ── _build_family_metadata ───────────────────────────────────────────────


class TestBuildFamilyMetadata:
    def test_empty_input(self) -> None:
        assert _build_family_metadata({}) == {}

    def test_single_family(self) -> None:
        cfgs = {
            "id_linear_er20": FamilyConfig(
                name="id_linear_er20",
                n_nodes=20,
                graph_cfg=ErdosRenyiConfig(sparsity=0.0526),
                mech_cfg=LinearMechanismConfig(),
            )
        }
        result = _build_family_metadata(cfgs)
        assert "id_linear_er20" in result
        entry = result["id_linear_er20"]
        assert entry["n_nodes"] == 20
        assert "er" in entry["graph_type"].lower()
        assert "linear" in entry["mech_type"].lower()
        assert entry["sparsity_param"] == pytest.approx(0.0526)

    def test_multiple_families(self) -> None:
        cfgs = {
            "id_linear_sf2": FamilyConfig(
                name="sf",
                n_nodes=20,
                graph_cfg=ScaleFreeConfig(m=2),
                mech_cfg=LinearMechanismConfig(),
            ),
            "ood_graph_sbm_linear": FamilyConfig(
                name="sbm",
                n_nodes=20,
                graph_cfg=SBMConfig(n_blocks=4, p_intra=0.5, p_inter=0.01),
                mech_cfg=LinearMechanismConfig(),
            ),
        }
        result = _build_family_metadata(cfgs)
        assert len(result) == 2
        # Keys are now cfg.name, not the original dict keys.
        assert result["sf"]["sparsity_param"] == pytest.approx(2.0)
        assert "sparsity_param" not in result["sbm"]


# ── CausalMetaModule.family_distances (was _compute_distances) ───────────


class TestFamilyDistances:
    """Test that CausalMetaModule._compute_all_distances() produces the expected structure."""

    def _make_module_with_families(
        self,
        train_family: Any,
        test_families: dict[str, Any],
    ) -> CausalMetaModule:
        """Create a CausalMetaModule with mocked config and pre-set families."""
        mock_config = MagicMock()
        dm = CausalMetaModule(mock_config)
        dm.train_family = train_family
        dm.test_families = test_families
        return dm

    def test_no_train_family_returns_empty(self) -> None:
        dm = self._make_module_with_families(None, {"fam_a": MagicMock()})
        result = dm._compute_all_distances()
        assert result == {}

    def test_no_test_families_returns_empty(self) -> None:
        dm = self._make_module_with_families(MagicMock(), {})
        result = dm._compute_all_distances()
        assert result == {}

    @patch("causal_meta.datasets.data_module.compute_family_distance")
    def test_with_train_and_test_families(self, mock_cfd: MagicMock) -> None:
        mock_cfd.return_value = 0.42
        dm = self._make_module_with_families(
            MagicMock(),
            {"id_linear_er20": MagicMock(), "ood_mech_periodic_er40": MagicMock()},
        )
        result = dm._compute_all_distances()
        assert "id_linear_er20" in result
        assert "ood_mech_periodic_er40" in result
        # Each entry should contain all 3 distance metrics
        for name in ("id_linear_er20", "ood_mech_periodic_er40"):
            assert "spectral" in result[name]
            assert "kl_degree" in result[name]
            assert "mechanism" in result[name]
            assert result[name]["spectral"] == pytest.approx(0.42)

    @patch("causal_meta.datasets.data_module.compute_family_distance")
    def test_graceful_fallback_on_failure(self, mock_cfd: MagicMock) -> None:
        """If compute_family_distance raises, the metric falls back to NaN."""
        mock_cfd.side_effect = RuntimeError("boom")
        dm = self._make_module_with_families(MagicMock(), {"fam_a": MagicMock()})
        result = dm._compute_all_distances()
        assert "fam_a" in result
        for key in ("spectral", "kl_degree", "mechanism"):
            assert math.isnan(result["fam_a"][key])
