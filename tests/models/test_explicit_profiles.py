from __future__ import annotations

from types import SimpleNamespace

import pytest

from causal_meta.models.bayesdag.model import BayesDAGModel
from causal_meta.models.dibs.model import DiBSModel
from causal_meta.runners.utils.explicit_profiles import (
    apply_explicit_profile,
    compute_fallback_keys,
    infer_explicit_profile,
    nearest_paper_node_bucket,
    strip_graph_suffix,
    strip_node_bucket_suffix,
)


# ── strip_graph_suffix ───────────────────────────────────────────────


def test_strip_graph_suffix_known() -> None:
    assert strip_graph_suffix("gpcde_er40") == "gpcde"
    assert strip_graph_suffix("neuralnet_sbm") == "neuralnet"
    assert strip_graph_suffix("linear_sf3") == "linear"


def test_strip_graph_suffix_no_suffix() -> None:
    assert strip_graph_suffix("linear") == "linear"
    assert strip_graph_suffix("logistic") == "logistic"


def test_strip_node_bucket_suffix_known() -> None:
    assert strip_node_bucket_suffix("linear_er40_d20") == "linear_er40"
    assert strip_node_bucket_suffix("gpcde_d50") == "gpcde"


def test_nearest_paper_node_bucket() -> None:
    assert nearest_paper_node_bucket(10) == "d20"
    assert nearest_paper_node_bucket(30) == "d20"
    assert nearest_paper_node_bucket(35) == "d50"
    assert nearest_paper_node_bucket(60) == "d50"


# ── compute_fallback_keys ────────────────────────────────────────────


def test_fallback_keys_id_mechanism_no_suffix() -> None:
    assert compute_fallback_keys("linear") == ["linear"]


def test_fallback_keys_id_mechanism_with_suffix() -> None:
    assert compute_fallback_keys("gpcde_er40_d20") == [
        "gpcde_er40_d20",
        "gpcde_d20",
        "gpcde_er40",
        "gpcde",
    ]


def test_fallback_keys_ood_mechanism_no_suffix() -> None:
    assert compute_fallback_keys("logistic") == ["logistic", "gpcde"]


def test_fallback_keys_ood_mechanism_with_suffix() -> None:
    assert compute_fallback_keys("logistic_er40_d20") == [
        "logistic_er40_d20",
        "logistic_d20",
        "logistic_er40",
        "logistic",
        "gpcde_er40_d20",
        "gpcde_d20",
        "gpcde_er40",
        "gpcde",
    ]


def test_fallback_keys_ood_mechanism_ood_graph() -> None:
    assert compute_fallback_keys("periodic_sbm_d50") == [
        "periodic_sbm_d50",
        "periodic_d50",
        "periodic_sbm",
        "periodic",
        "gpcde_sbm_d50",
        "gpcde_d50",
        "gpcde_sbm",
        "gpcde",
    ]


# ── infer_explicit_profile ───────────────────────────────────────────


def test_infer_explicit_profile_from_dataset_name() -> None:
    assert infer_explicit_profile("id_linear_er20_d20_n500", None) == "linear_d20"
    assert (
        infer_explicit_profile("id_neuralnet_er40_d20_n500", None)
        == "neuralnet_er40_d20"
    )
    assert infer_explicit_profile("id_gpcde_er60_d60_n500", None) == "gpcde_er60_d50"


def test_infer_explicit_profile_ood_mechanism_preserves_name() -> None:
    """OOD mechanisms are no longer remapped to gpcde."""
    assert (
        infer_explicit_profile("ood_mech_periodic_er20_d20_n500", None)
        == "periodic_d20"
    )
    assert (
        infer_explicit_profile("ood_mech_periodic_er40_d20_n500", None)
        == "periodic_er40_d20"
    )
    assert (
        infer_explicit_profile("ood_mech_logistic_map_er60_d20_n500", None)
        == "logistic_er60_d20"
    )
    assert (
        infer_explicit_profile("ood_mech_pnl_tanh_sf2_d20_n500", None) == "pnl_sf2_d20"
    )
    assert (
        infer_explicit_profile("ood_mech_square_sf3_d20_n500", None) == "square_sf3_d20"
    )


def test_infer_explicit_profile_ood_both() -> None:
    """OOD graph + OOD mechanism stress tests."""
    assert (
        infer_explicit_profile("ood_both_sbm_periodic_d20_n500", None)
        == "periodic_sbm_d20"
    )
    assert (
        infer_explicit_profile("ood_both_grg_logistic_map_d60_n50", None)
        == "logistic_grg_d50"
    )
    assert infer_explicit_profile("ood_both_ws_pnl_tanh_d60_n50", None) == "pnl_ws_d50"


def test_infer_explicit_profile_from_family_mechanism_fallback() -> None:
    """When dataset name doesn't match, fall back to family object."""

    class GPMechanismFactory:
        pass

    family = SimpleNamespace(mechanism_factory=GPMechanismFactory())
    # Name has no known mechanism keyword but "periodic" is in there — will match
    # the keyword check first, so use a name with no keyword at all:
    assert (
        infer_explicit_profile("unknown_dataset_er40_d20", family) == "gpcde_er40_d20"
    )


# ── apply_explicit_profile ───────────────────────────────────────────


def test_apply_explicit_profile_calls_setter() -> None:
    class DummyModel:
        def __init__(self) -> None:
            self.profile = None

        def set_inference_profile(self, profile: str | None) -> None:
            self.profile = profile

    model = DummyModel()
    assert apply_explicit_profile(model, "linear") is True
    assert model.profile == "linear"


# ── DiBS profile fallback ────────────────────────────────────────────


def test_dibs_profile_override_updates_hyperparameters() -> None:
    model = DiBSModel(
        num_nodes=5,
        mode="nonlinear",
        steps=1000,
        use_marginal=False,
        alpha=0.02,
        gamma_z=5.0,
        gamma_theta=1000.0,
        n_particles=32,
        profile_overrides={
            "linear_d20": {
                "mode": "linear",
                "steps": 3000,
                "use_marginal": True,
                "alpha": 0.2,
                "gamma_z": 5.0,
                "gamma_theta": 500.0,
                "n_particles": 64,
            }
        },
    )

    model.set_inference_profile("linear_d20")

    assert model.mode == "linear"
    assert model.steps == 3000
    assert model.use_marginal is True
    assert model.alpha == 0.2
    assert model.gamma_z == 5.0
    assert model.gamma_theta == 500.0
    assert model.n_particles == 64


def test_dibs_fallback_ood_mechanism_uses_mechanism_profile() -> None:
    """logistic_er40 should fall back to 'logistic' profile."""
    model = DiBSModel(
        num_nodes=5,
        alpha=0.05,
        gamma_z=5.0,
        profile_overrides={
            "logistic": {
                "mode": "nonlinear",
                "alpha": 0.08,
                "gamma_z": 15.0,
            },
            "gpcde_er40": {
                "mode": "nonlinear",
                "alpha": 0.05,
                "gamma_z": 10.0,
            },
        },
    )

    model.set_inference_profile("logistic_er40_d20")

    # Should use 'logistic' profile (step 2), NOT 'gpcde_er40' (step 3)
    assert model.alpha == 0.08
    assert model.gamma_z == 15.0


def test_dibs_fallback_ood_mechanism_falls_to_id_when_no_mechanism_profile() -> None:
    """If no mechanism profile exists, fall to the nearest paper-backed ID profile."""
    model = DiBSModel(
        num_nodes=5,
        alpha=0.05,
        gamma_z=5.0,
        profile_overrides={
            "gpcde_d20": {
                "mode": "nonlinear",
                "alpha": 0.05,
                "gamma_z": 10.0,
            },
        },
    )

    model.set_inference_profile("periodic_er40_d20")

    # No periodic profile → falls to gpcde_d20 before graph-specific density hints.
    assert model.gamma_z == 10.0


def test_dibs_set_num_nodes_updates_cache_sensitive_state() -> None:
    model = DiBSModel(num_nodes=5)
    model._target_cache = object()

    model.set_num_nodes(20)

    assert model.num_nodes == 20
    assert model._target_cache is None


# ── BayesDAG profile fallback ────────────────────────────────────────


def test_bayesdag_profile_override_updates_hyperparameters() -> None:
    model = BayesDAGModel(
        num_nodes=5,
        variant="nonlinear",
        lambda_sparse=10.0,
        num_chains=10,
        scale_noise=0.001,
        scale_noise_p=0.01,
        profile_overrides={
            "gpcde": {
                "variant": "nonlinear",
                "lambda_sparse": 1.0,
                "num_chains": 10,
                "scale_noise": 0.1,
                "scale_noise_p": 0.001,
            }
        },
    )

    model.set_inference_profile("gpcde")

    assert model.variant == "nonlinear"
    assert model.lambda_sparse == 1.0
    assert model.num_chains == 10
    assert model.scale_noise == 0.1
    assert model.scale_noise_p == 0.001


def test_bayesdag_fallback_ood_mechanism_uses_density_profile() -> None:
    """logistic_er40 should fall through to gpcde_er40 in BayesDAG."""
    model = BayesDAGModel(
        num_nodes=5,
        lambda_sparse=300.0,
        profile_overrides={
            "gpcde_er40": {
                "variant": "nonlinear",
                "lambda_sparse": 150.0,
                "num_chains": 10,
                "scale_noise": 0.01,
                "scale_noise_p": 0.1,
            },
        },
    )

    model.set_inference_profile("logistic_er40_d20")

    # No 'logistic_er40' or 'logistic' → falls to 'gpcde_er40'
    assert model.lambda_sparse == 150.0


def test_bayesdag_set_num_nodes_updates_active_shape() -> None:
    model = BayesDAGModel(num_nodes=5)

    model.set_num_nodes(20)

    assert model.num_nodes == 20
