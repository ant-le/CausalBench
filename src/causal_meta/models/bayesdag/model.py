from __future__ import annotations

import json
import os
import subprocess
import tempfile
from types import MethodType
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import logging

import numpy as np
import torch

from causal_meta.models.base import BaseModel
from causal_meta.models.factory import register_model
from causal_meta.runners.utils.explicit_profiles import compute_fallback_keys

log = logging.getLogger(__name__)


@register_model("bayesdag")
class BayesDAGModel(BaseModel):
    """
    Wrapper for BayesDAG (gradient-based posterior inference for DAGs).

    The model performs explicit inference per dataset instance and returns
    posterior graph samples compatible with the inference cache pipeline.
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        variant: str = "nonlinear",
        lambda_sparse: float = 1.0,
        num_chains: int = 10,
        sinkhorn_n_iter: int = 3000,
        scale_noise: float = 0.1,
        scale_noise_p: float = 1.0,
        batch_size: int = 64,
        max_epochs: int = 100,
        standardize_data_mean: bool = False,
        standardize_data_std: bool = False,
        save_dir: Optional[str] = None,
        sparse_init: bool = False,
        input_perm: bool = False,
        vi_norm: bool = False,
        norm_layers: bool = False,
        res_connection: bool = False,
        external_python: Optional[str] = None,
        external_timeout_s: int = 3600,
        device: str = "auto",
        skip_evaluation: bool = True,
        profile_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = kwargs
        self.num_nodes = num_nodes
        self.variant = variant
        self.lambda_sparse = lambda_sparse
        self.num_chains = num_chains
        self.sinkhorn_n_iter = sinkhorn_n_iter
        self.scale_noise = scale_noise
        self.scale_noise_p = scale_noise_p
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.standardize_data_mean = standardize_data_mean
        self.standardize_data_std = standardize_data_std
        self.save_dir = save_dir or os.path.join(os.getcwd(), "bayesdag_output")
        self.sparse_init = sparse_init
        self.input_perm = input_perm
        self.vi_norm = vi_norm
        self.norm_layers = norm_layers
        self.res_connection = res_connection
        self.external_python = external_python
        self.external_process = bool(external_python)
        self.external_timeout_s = external_timeout_s
        self.device = device
        self.skip_evaluation = skip_evaluation
        self._base_profile = {
            "variant": variant,
            "lambda_sparse": lambda_sparse,
            "num_chains": num_chains,
            "scale_noise": scale_noise,
            "scale_noise_p": scale_noise_p,
        }
        self._profile_overrides: dict[str, dict[str, Any]] = {}
        if profile_overrides is not None:
            for name, values in profile_overrides.items():
                self._profile_overrides[str(name).lower()] = dict(values)
        self._global_override = self._profile_overrides.get("_global", {})
        self._active_profile: str | None = None

    def set_inference_profile(self, profile: str | None) -> None:
        """Apply named BayesDAG profile overrides for explicit comparisons.

        Applies global ``_global`` override first, then resolves the best
        matching specific profile via :func:`compute_fallback_keys` and
        merges it on top.

        Args:
            profile: Profile identifier or ``None`` for defaults.
        """
        profile_key = str(profile).lower() if profile is not None else "default"

        # Resolve best-matching profile via fallback chain
        specific_override: dict[str, Any] = {}
        for key in compute_fallback_keys(profile_key):
            if key in self._profile_overrides:
                specific_override = self._profile_overrides[key]
                break

        # Start with global override, then apply specific profile
        combined_override = dict(self._global_override)
        combined_override.update(specific_override)

        self.variant = str(
            combined_override.get("variant", self._base_profile["variant"])
        )
        self.lambda_sparse = float(
            combined_override.get("lambda_sparse", self._base_profile["lambda_sparse"])
        )
        self.num_chains = int(
            combined_override.get("num_chains", self._base_profile["num_chains"])
        )
        self.scale_noise = float(
            combined_override.get("scale_noise", self._base_profile["scale_noise"])
        )
        self.scale_noise_p = float(
            combined_override.get("scale_noise_p", self._base_profile["scale_noise_p"])
        )
        self._active_profile = profile_key

    def set_num_nodes(self, num_nodes: int) -> None:
        """Update the active node count for the next explicit inference call."""
        resolved = int(num_nodes)
        if resolved <= 0:
            raise ValueError("num_nodes must be positive.")
        self.num_nodes = resolved

    @property
    def needs_pretraining(self) -> bool:
        return False

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Forward pass placeholder for explicit inference models.

        Args:
            x: Input data tensor of shape (Batch, Samples, Variables).
            mask: Unused — accepted for interface compatibility.

        Raises:
            RuntimeError: Always raised since BayesDAG is sampled via `sample`.
        """
        raise RuntimeError("BayesDAGModel does not implement forward(); use sample().")

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample adjacency matrices from the BayesDAG posterior.

        Args:
            x: Input data tensor of shape (Batch, Samples, Variables).
            num_samples: Number of graph samples to generate per batch element.
            mask: Unused — accepted for interface compatibility.

        Returns:
            Sampled adjacency matrices of shape (Batch, num_samples, Variables, Variables).
        """
        if self.external_python:
            return self._sample_external(x, num_samples)

        Dataset, Variables, model_cls, DataProcessor = self._require_causica()

        if x.ndim != 3:
            raise ValueError("Input data must have shape (Batch, Samples, Variables).")

        batch_size, _, num_nodes = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(
                "Input data node count does not match configured num_nodes."
            )

        resolved_device = self._device_to_string(x.device)

        log.info(
            "BayesDAG in-process sample: mode=in_process, device=%s, "
            "resolved_device=%s, batch_size=%d, "
            "num_nodes=%d, num_samples=%d, variant=%s, "
            "cuda_available=%s, cuda_device_count=%d",
            x.device,
            resolved_device,
            batch_size,
            num_nodes,
            num_samples,
            self.variant,
            torch.cuda.is_available(),
            torch.cuda.device_count(),
        )

        train_config = {
            "batch_size": int(self.batch_size),
            "max_epochs": int(self.max_epochs),
            "standardize_data_mean": bool(self.standardize_data_mean),
            "standardize_data_std": bool(self.standardize_data_std),
            "skip_evaluation": bool(self.skip_evaluation),
        }

        samples_per_batch = []
        for batch_idx in range(batch_size):
            x_np = x[batch_idx].detach().cpu().numpy().astype(np.float32, copy=False)
            mask_np = np.ones_like(x_np, dtype=np.float32)
            variables = Variables.create_from_data_and_dict(x_np, mask_np, None)

            dataset = self._build_dataset(
                Dataset=Dataset,
                train_data=x_np,
                train_mask=mask_np,
                variables=variables,
                seed=batch_idx,
            )

            model = self._build_model(
                model_cls=model_cls,
                variables=variables,
                device=x.device,
            )
            self._patch_model_for_unknown_graph_inference(
                model=model,
                DataProcessor=DataProcessor,
            )
            model.run_train(dataset, train_config_dict=train_config)

            graph_samples, _ = model.get_adj_matrix_tensor(samples=int(num_samples))
            graphs_t = graph_samples.to(device=x.device, dtype=torch.float32)
            samples_per_batch.append(graphs_t)

        return torch.stack(samples_per_batch, dim=0)

    def calculate_loss(
        self, output: Any, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Loss placeholder for explicit inference models.

        Args:
            output: Model output (unused).
            target: Ground truth adjacency matrix (unused).

        Raises:
            RuntimeError: Always raised since BayesDAG does not expose a training loss here.
        """
        _ = output
        _ = target
        _ = kwargs
        raise RuntimeError("BayesDAGModel does not implement calculate_loss().")

    def _build_model(self, model_cls: Any, variables: Any, device: torch.device) -> Any:
        kwargs = {
            "model_id": f"bayesdag_{self.variant}",
            "variables": variables,
            "save_dir": self.save_dir,
            "device": device,
            "lambda_sparse": self.lambda_sparse,
            "num_chains": self.num_chains,
            "sinkhorn_n_iter": self.sinkhorn_n_iter,
            "scale_noise": self.scale_noise,
            "scale_noise_p": self.scale_noise_p,
            "VI_norm": self.vi_norm,
            "input_perm": self.input_perm,
            "sparse_init": self.sparse_init,
        }
        if self.variant == "nonlinear":
            kwargs["norm_layers"] = self.norm_layers
            kwargs["res_connection"] = self.res_connection
        return model_cls(**kwargs)

    def _device_to_string(self, device: torch.device) -> str:
        if device.type == "cuda":
            index = device.index
            if index is None:
                index = torch.cuda.current_device() if torch.cuda.is_available() else 0
            return f"cuda:{index}"
        return device.type

    def _sample_external(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Input data must have shape (Batch, Samples, Variables).")

        batch_size, _, num_nodes = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(
                "Input data node count does not match configured num_nodes."
            )

        python_path = self._resolve_external_python()
        script_path = Path(__file__).resolve().with_name("external_infer.py")
        requested_device = self._device_to_string(x.device)

        log.info(
            "BayesDAG external sample: mode=external, python=%s, batch_size=%d, "
            "num_nodes=%d, num_samples=%d, variant=%s, parent_device=%s, "
            "requested_device=%s, timeout=%ds, cuda_available=%s, "
            "cuda_device_count=%d, cuda_visible_devices=%s",
            python_path,
            batch_size,
            num_nodes,
            num_samples,
            self.variant,
            x.device,
            requested_device,
            self.external_timeout_s,
            torch.cuda.is_available(),
            torch.cuda.device_count(),
            os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        )

        train_config = {
            "batch_size": int(self.batch_size),
            "max_epochs": int(self.max_epochs),
            "standardize_data_mean": bool(self.standardize_data_mean),
            "standardize_data_std": bool(self.standardize_data_std),
            "skip_evaluation": bool(self.skip_evaluation),
        }
        model_config = {
            "variant": self.variant,
            "lambda_sparse": float(self.lambda_sparse),
            "num_chains": int(self.num_chains),
            "sinkhorn_n_iter": int(self.sinkhorn_n_iter),
            "scale_noise": float(self.scale_noise),
            "scale_noise_p": float(self.scale_noise_p),
            "save_dir": self.save_dir,
            "sparse_init": bool(self.sparse_init),
            "input_perm": bool(self.input_perm),
            "vi_norm": bool(self.vi_norm),
            "norm_layers": bool(self.norm_layers),
            "res_connection": bool(self.res_connection),
        }

        samples_per_batch = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for batch_idx in range(batch_size):
                x_np = (
                    x[batch_idx].detach().cpu().numpy().astype(np.float32, copy=False)
                )
                mask_np = np.ones_like(x_np, dtype=np.float32)

                input_path = tmp_path / f"input_{batch_idx}.npz"
                output_path = tmp_path / f"output_{batch_idx}.npz"
                config_path = tmp_path / f"config_{batch_idx}.json"

                np.savez(input_path, data=x_np, mask=mask_np)
                config_payload = {
                    "model": model_config,
                    "train": train_config,
                    "num_samples": int(num_samples),
                    "seed": int(batch_idx),
                    "device": requested_device,
                    "output": str(output_path),
                }
                config_path.write_text(json.dumps(config_payload))

                cmd = [
                    python_path,
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--input",
                    str(input_path),
                ]
                env = os.environ.copy()
                if requested_device == "mps" or self.device in {"auto", "mps"}:
                    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                subprocess.run(
                    cmd,
                    check=True,
                    timeout=self.external_timeout_s,
                    env=env,
                )

                result = np.load(output_path)
                graph_samples = result["graph_samples"]
                graphs_t = torch.from_numpy(graph_samples).to(
                    device=x.device, dtype=torch.float32
                )
                samples_per_batch.append(graphs_t)

        return torch.stack(samples_per_batch, dim=0)

    def _patch_model_for_unknown_graph_inference(
        self,
        *,
        model: Any,
        DataProcessor: Any,
    ) -> None:
        """Disable ground-truth graph dependencies for fair explicit inference."""

        def _process_dataset_without_ground_truth(
            inner_self: Any,
            dataset: Any,
            train_config_dict: Dict[str, Any] | None = None,
            variables: Any | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            if train_config_dict is None:
                train_config_dict = {}
            if variables is None:
                variables = inner_self.variables

            inner_self.data_processor = DataProcessor(
                variables,
                unit_scale_continuous=False,
                standardize_data_mean=train_config_dict.get(
                    "standardize_data_mean", False
                ),
                standardize_data_std=train_config_dict.get(
                    "standardize_data_std", False
                ),
            )
            processed_dataset = inner_self.data_processor.process_dataset(dataset)
            data, mask = processed_dataset.train_data_and_mask
            data = data.astype(np.float32)
            return data, mask

        model.process_dataset = MethodType(_process_dataset_without_ground_truth, model)
        model.evaluate_metrics = lambda *args, **kwargs: None

    def _build_dataset(
        self,
        *,
        Dataset: Any,
        train_data: np.ndarray,
        train_mask: np.ndarray,
        variables: Any,
        seed: int,
    ) -> Any:
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

    def _require_causica(self) -> Tuple[Any, Any, Any]:
        if self.variant not in {"linear", "nonlinear"}:
            raise ValueError("BayesDAG variant must be 'linear' or 'nonlinear'.")

        try:
            from causica.datasets.dataset import Dataset  # type: ignore[import-not-found]
            from causica.datasets.variables import Variables  # type: ignore[import-not-found]
            from causica.models.bayesdag.bayesdag_linear import BayesDAGLinear  # type: ignore[import-not-found]
            from causica.models.bayesdag.bayesdag_nonlinear import BayesDAGNonLinear  # type: ignore[import-not-found]
            from causica.preprocessing.data_processor import DataProcessor  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError(
                "BayesDAG requires the 'causica' package from Project-BayesDAG. "
                "Install with `pip install 'git+https://github.com/microsoft/Project-BayesDAG.git#subdirectory=src'` "
                "and use a Python version compatible with that package (>=3.8,<3.10)."
            ) from exc

        model_cls = BayesDAGLinear if self.variant == "linear" else BayesDAGNonLinear
        return Dataset, Variables, model_cls, DataProcessor

    def _resolve_external_python(self) -> str:
        if not self.external_python:
            raise RuntimeError(
                "BayesDAG external inference requires external_python to be set."
            )
        python_path = Path(os.path.expandvars(self.external_python)).expanduser()
        if not python_path.is_absolute():
            python_path = (Path.cwd() / python_path).resolve()
        if not python_path.exists():
            raise FileNotFoundError(
                f"Resolved BayesDAG external_python does not exist: {python_path}"
            )
        return str(python_path)
