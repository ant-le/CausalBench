from __future__ import annotations

import argparse
import json
import logging
import os
import time
from types import MethodType
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from causica.datasets.dataset import Dataset
from causica.datasets.variables import Variables
from causica.models.bayesdag.bayesdag_linear import BayesDAGLinear
from causica.models.bayesdag.bayesdag_nonlinear import BayesDAGNonLinear
from causica.preprocessing.data_processor import DataProcessor

log = logging.getLogger(__name__)


def _build_dataset(
    train_data: np.ndarray,
    train_mask: np.ndarray,
    variables: Variables,
    seed: int,
) -> Dataset:
    num_nodes = train_data.shape[1]
    graph_args: Dict[str, Any] = {
        "num_variables": int(num_nodes),
        "exp_edges": float("nan"),
        "exp_edges_per_node": float("nan"),
        "graph_type": "unknown",
        "seed": int(seed),
    }

    return Dataset(
        train_data=train_data,
        train_mask=train_mask,
        val_data=None,
        val_mask=None,
        test_data=train_data,
        test_mask=train_mask,
        variables=variables,
        data_split=None,
        held_out_interventions=None,
        true_posterior=None,
        graph_args=graph_args,
    )


def _variables_dict_from_data(data: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    variables = []
    for idx in range(data.shape[1]):
        col = data[:, idx]
        col_mask = mask[:, idx].astype(bool)
        if col_mask.any():
            col_min = float(col[col_mask].min())
            col_max = float(col[col_mask].max())
        else:
            col_min = 0.0
            col_max = 0.0
        variables.append(
            {
                "name": f"Column {idx}",
                "type": "continuous",
                "lower": col_min,
                "upper": col_max,
                "query": True,
                "target": False,
                "always_observed": True,
            }
        )
    return {
        "variables": variables,
        "auxiliary_variables": [],
        "used_cols": list(range(data.shape[1])),
    }


def _validate_data(data: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(data).all():
        raise ValueError("BayesDAG input data contains NaN or inf values.")
    if not np.isfinite(mask).all():
        raise ValueError("BayesDAG input mask contains NaN or inf values.")
    if data.ndim != 2:
        raise ValueError("BayesDAG input data must have shape (samples, variables).")
    if mask.shape != data.shape:
        raise ValueError("BayesDAG input mask must match data shape.")
    return data, mask


def _log_data_stats(data: np.ndarray, mask: np.ndarray) -> None:
    observed = mask.astype(bool)
    if observed.any():
        data_obs = data[observed]
        log.info(
            "BayesDAG data stats: n=%d, d=%d, min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
            data.shape[0],
            data.shape[1],
            float(data_obs.min()),
            float(data_obs.max()),
            float(data_obs.mean()),
            float(data_obs.std()),
        )
    else:
        log.warning("BayesDAG data mask has no observed entries.")


def _resolve_device(requested_device: Optional[str] = None) -> torch.device:
    if requested_device is not None:
        normalized = str(requested_device).strip().lower()
        if normalized == "cuda":
            normalized = "cuda:0"

        try:
            device = torch.device(normalized)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Invalid BayesDAG device request: {requested_device!r}"
            ) from exc

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "BayesDAG requested CUDA device but CUDA is not available."
                )
            device_count = torch.cuda.device_count()
            index = 0 if device.index is None else int(device.index)
            if index < 0 or index >= device_count:
                raise RuntimeError(
                    "BayesDAG requested CUDA device index out of range: "
                    f"{index} (device_count={device_count})."
                )
            return torch.device(f"cuda:{index}")

        if device.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "BayesDAG requested MPS device but MPS is not available."
                )
            return torch.device("mps")

        if device.type == "cpu":
            return torch.device("cpu")

        raise ValueError(
            "BayesDAG device must be one of 'cpu', 'mps', 'cuda', or 'cuda:<idx>'."
        )

    if torch.cuda.is_available():
        preferred_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        preferred_device = torch.device("mps")
    else:
        preferred_device = torch.device("cpu")

    return preferred_device


def _build_model(
    config: Dict[str, Any],
    variables: Variables,
    device: torch.device,
) -> Any:
    variant = config["variant"]
    model_cls = BayesDAGLinear if variant == "linear" else BayesDAGNonLinear

    kwargs = {
        "model_id": f"bayesdag_{variant}",
        "variables": variables,
        "save_dir": config["save_dir"],
        "device": device,
        "lambda_sparse": config["lambda_sparse"],
        "num_chains": config["num_chains"],
        "sinkhorn_n_iter": config["sinkhorn_n_iter"],
        "scale_noise": config["scale_noise"],
        "scale_noise_p": config["scale_noise_p"],
        "VI_norm": config["vi_norm"],
        "input_perm": config["input_perm"],
        "sparse_init": config["sparse_init"],
    }
    if variant == "nonlinear":
        kwargs["norm_layers"] = config["norm_layers"]
        kwargs["res_connection"] = config["res_connection"]

    return model_cls(**kwargs)


def _patch_model_for_unknown_graph_inference(model: Any) -> None:
    """Disable ground-truth graph dependencies for fair explicit inference."""

    def _process_dataset_without_ground_truth(
        inner_self: Any,
        dataset: Any,
        train_config_dict: Dict[str, Any] | None = None,
        variables: Variables | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if train_config_dict is None:
            train_config_dict = {}
        if variables is None:
            variables = inner_self.variables

        inner_self.data_processor = DataProcessor(
            variables,
            unit_scale_continuous=False,
            standardize_data_mean=train_config_dict.get("standardize_data_mean", False),
            standardize_data_std=train_config_dict.get("standardize_data_std", False),
        )
        # Process raw arrays directly instead of calling
        # data_processor.process_dataset(), which reconstructs a Dataset
        # object without forwarding graph_args and crashes in the
        # upstream causica Dataset.__init__.
        data, mask = dataset.train_data_and_mask
        data, mask = inner_self.data_processor.process_data_and_masks(data, mask)
        return data.astype(np.float32), mask

    model.process_dataset = MethodType(_process_dataset_without_ground_truth, model)
    model.evaluate_metrics = lambda *args, **kwargs: None


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    input_data = np.load(args.input)
    data = input_data["data"].astype(np.float32, copy=False)
    mask = input_data["mask"].astype(np.float32, copy=False)
    _validate_data(data, mask)
    _log_data_stats(data, mask)

    variables = Variables.create_from_data_and_dict(
        data, mask, _variables_dict_from_data(data, mask)
    )
    dataset = _build_dataset(data, mask, variables, seed=int(config["seed"]))

    requested_device_raw = config.get("device")
    requested_device = (
        str(requested_device_raw) if requested_device_raw is not None else None
    )
    resolved_device = _resolve_device(requested_device)
    cuda_index = (
        (0 if resolved_device.index is None else int(resolved_device.index))
        if resolved_device.type == "cuda"
        else None
    )
    cuda_name = (
        torch.cuda.get_device_name(cuda_index) if cuda_index is not None else None
    )
    log.info(
        "BayesDAG external bootstrap: requested_device=%s, resolved_device=%s, "
        "cuda_available=%s, cuda_device_count=%d, cuda_device_name=%s, "
        "cuda_visible_devices=%s",
        requested_device,
        resolved_device,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        cuda_name,
        os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    )

    model = _build_model(config["model"], variables, device=resolved_device)
    _patch_model_for_unknown_graph_inference(model)
    if bool(config.get("train", {}).get("skip_evaluation", False)):
        model.evaluate_metrics = lambda *args, **kwargs: None

    start_train = time.perf_counter()
    model.run_train(dataset, train_config_dict=config["train"])
    train_s = time.perf_counter() - start_train

    start_sample = time.perf_counter()
    graph_samples, _ = model.get_adj_matrix_tensor(samples=int(config["num_samples"]))
    sample_s = time.perf_counter() - start_sample

    log.info(
        "BayesDAG timing: train=%.2fs, sample=%.2fs, total=%.2fs",
        train_s,
        sample_s,
        train_s + sample_s,
    )
    np.savez(config["output"], graph_samples=graph_samples.detach().cpu().numpy())


if __name__ == "__main__":
    main()
