from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from causal_meta.models.dibs.model import DiBSModel

log = logging.getLogger(__name__)


def _validate_data(data: np.ndarray, *, num_nodes: int) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(
            "DiBS external input data must have shape (samples, variables)."
        )
    if int(data.shape[1]) != int(num_nodes):
        raise ValueError(
            "DiBS external input node count does not match configured num_nodes."
        )
    if not np.isfinite(data).all():
        raise ValueError("DiBS external input contains NaN or inf values.")
    return data.astype(np.float32, copy=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    input_path = Path(args.input)

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    input_payload = np.load(input_path)
    data = _validate_data(input_payload["data"], num_nodes=int(config["num_nodes"]))

    # Log JAX backend info early so GPU vs CPU is immediately visible.
    import jax  # noqa: E402 — deferred to keep startup fast when unused

    jax_devices = jax.devices()
    log.info(
        "JAX backend: platform=%s, devices=%s",
        jax_devices[0].platform if jax_devices else "none",
        [str(d) for d in jax_devices],
    )

    log.info(
        "DiBS external bootstrap: samples=%d, num_nodes=%d, num_samples=%d, mode=%s, steps=%d, n_particles=%s",
        int(data.shape[0]),
        int(data.shape[1]),
        int(config["num_samples"]),
        str(config["mode"]),
        int(config["steps"]),
        config.get("n_particles"),
    )

    graph_samples = DiBSModel.sample_numpy_array(
        data=data,
        num_nodes=int(config["num_nodes"]),
        num_samples=int(config["num_samples"]),
        mode=str(config["mode"]),
        steps=int(config["steps"]),
        seed=int(config["seed"]),
        use_marginal=bool(config["use_marginal"]),
        xla_preallocate=bool(config["xla_preallocate"]),
        alpha=config.get("alpha"),
        gamma_z=config.get("gamma_z"),
        gamma_theta=config.get("gamma_theta"),
        n_particles=config.get("n_particles"),
    )

    np.savez(config["output"], graph_samples=graph_samples)


if __name__ == "__main__":
    main()
