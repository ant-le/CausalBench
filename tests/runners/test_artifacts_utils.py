from __future__ import annotations

import torch
from omegaconf import OmegaConf

from causal_meta.runners.utils.artifacts import (
    atomic_torch_save, cache_settings, cache_suffix, find_inference_artifact,
    prepare_graph_samples_for_cache, torch_load)


def test_prepare_graph_samples_for_cache_uint8_and_truncation() -> None:
    samples = torch.tensor(
        [
            [
                [[0.0, 0.9], [0.2, 0.0]],
                [[0.0, 0.7], [0.6, 0.0]],
                [[0.0, 0.1], [0.8, 0.0]],
            ]
        ]
    )

    cached = prepare_graph_samples_for_cache(samples, dtype="uint8", max_samples=2)
    assert cached.dtype == torch.uint8
    assert cached.shape == (1, 2, 2, 2)
    assert torch.equal(cached[0, 0], torch.tensor([[0, 1], [0, 0]], dtype=torch.uint8))


def test_atomic_torch_save_and_torch_load_plain_and_gzip(tmp_path) -> None:
    payload = {"x": torch.tensor([1.0, 2.0]), "name": "artifact"}

    plain_path = tmp_path / "plain.pt"
    gzip_path = tmp_path / "gzip.pt.gz"

    atomic_torch_save(payload, plain_path)
    atomic_torch_save(payload, gzip_path)

    loaded_plain = torch_load(plain_path)
    loaded_gzip = torch_load(gzip_path)

    assert torch.equal(loaded_plain["x"], payload["x"])
    assert torch.equal(loaded_gzip["x"], payload["x"])
    assert loaded_plain["name"] == "artifact"
    assert loaded_gzip["name"] == "artifact"


def test_find_inference_artifact_supports_new_and_shared_layouts(tmp_path) -> None:
    inference_root = tmp_path / "inference"
    dataset_name = "dummy"
    model_name = "avici"
    seed = 123

    # New per-run layout: inference/<dataset>/seed_*.pt
    new_layout = inference_root / dataset_name
    new_layout.mkdir(parents=True, exist_ok=True)
    new_path = new_layout / "seed_123.pt"
    new_path.write_bytes(b"x")

    found_new = find_inference_artifact(
        inference_root,
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,
        prefer_compress=False,
        use_model_subdir=False,
    )
    assert found_new == new_path

    # Shared cache layout: inference/<model>/<dataset>/seed_*.pt.gz
    shared_layout = inference_root / model_name / dataset_name
    shared_layout.mkdir(parents=True, exist_ok=True)
    shared_path = shared_layout / "seed_123.pt.gz"
    shared_path.write_bytes(b"x")

    found_shared = find_inference_artifact(
        inference_root,
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,
        prefer_compress=True,
        use_model_subdir=True,
    )
    assert found_shared == shared_path


def test_cache_settings_and_suffix_from_config() -> None:
    cfg = OmegaConf.create(
        {
            "inference": {
                "cache_compress": True,
                "cache_dtype": "uint8",
                "cache_n_samples": 5,
            }
        }
    )

    compress, dtype, max_samples = cache_settings(cfg)
    assert compress is True
    assert dtype == "uint8"
    assert max_samples == 5
    assert cache_suffix(compress=compress) == ".pt.gz"
