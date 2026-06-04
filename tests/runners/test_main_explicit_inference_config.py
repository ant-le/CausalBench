from __future__ import annotations

from omegaconf import OmegaConf

from causal_meta.main import _resolve_explicit_model_inference_params


def test_resolve_explicit_model_inference_params_filters_to_allowlist() -> None:
    cfg = OmegaConf.create(
        {
            "inference": {
                "dibs": {
                    "steps": 50,
                    "n_particles": 8,
                    "profile_overrides": {"_global": {"n_particles": 8}},
                    "cache_n_samples": None,
                    "unknown": 123,
                }
            }
        }
    )

    resolved = _resolve_explicit_model_inference_params(cfg, "dibs")

    assert resolved == {
        "steps": 50,
        "n_particles": 8,
        "profile_overrides": {"_global": {"n_particles": 8}},
    }


def test_resolve_explicit_model_inference_params_supports_legacy_path() -> None:
    cfg = OmegaConf.create(
        {
            "inference": {
                "explicit": {
                    "bayesdag": {
                        "max_epochs": 5,
                        "num_chains": 2,
                        "batch_size": 512,
                    }
                }
            }
        }
    )

    resolved = _resolve_explicit_model_inference_params(cfg, "bayesdag")

    assert resolved == {
        "max_epochs": 5,
        "num_chains": 2,
        "batch_size": 512,
    }
