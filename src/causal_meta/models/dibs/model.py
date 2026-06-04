from __future__ import annotations

import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple
from typing import cast

import numpy as np
import torch

from causal_meta.models.base import BaseModel
from causal_meta.models.factory import register_model
from causal_meta.runners.utils.explicit_profiles import compute_fallback_keys

log = logging.getLogger(__name__)


@register_model("dibs")
class DiBSModel(BaseModel):
    """
    Wrapper for DiBS (Differentiable Bayesian Structure Learning).

    This model runs explicit posterior inference per dataset instance and
    returns graph samples compatible with the inference cache pipeline.
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        mode: str = "nonlinear",
        steps: int = 1000,
        seed: int = 0,
        use_marginal: bool = False,
        xla_preallocate: bool = False,
        external_process: bool = False,
        external_python: Optional[str] = None,
        external_timeout_s: int = 3600,
        alpha: float | None = None,
        gamma_z: float | None = None,
        gamma_theta: float | None = None,
        n_particles: int | None = None,
        profile_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = kwargs
        self.num_nodes = num_nodes
        self.mode = mode
        self.steps = steps
        self.seed = seed
        self.use_marginal = use_marginal
        self.xla_preallocate = xla_preallocate
        self.external_process = external_process
        self.external_python = external_python
        self.external_timeout_s = external_timeout_s
        self.alpha = alpha
        self.gamma_z = gamma_z
        self.gamma_theta = gamma_theta
        self.n_particles = n_particles

        self._base_profile = {
            "mode": mode,
            "steps": steps,
            "use_marginal": use_marginal,
            "alpha": alpha,
            "gamma_z": gamma_z,
            "gamma_theta": gamma_theta,
            "n_particles": n_particles,
        }
        self._profile_overrides: dict[str, dict[str, Any]] = {}
        if profile_overrides is not None:
            for name, values in profile_overrides.items():
                self._profile_overrides[str(name).lower()] = dict(values)
        self._global_override = self._profile_overrides.get("_global", {})
        self._active_profile: str | None = None
        self._rng_key = None
        self._external_seed_counter = 0
        self._target_cache = None

    def set_inference_profile(self, profile: str | None) -> None:
        """Apply a named DiBS profile override for explicit comparisons.

        Applies global ``_global`` override first, then resolves the best
        matching specific profile via :func:`compute_fallback_keys` and
        merges it on top.

        Args:
            profile: Profile identifier (for example ``"linear"``) or ``None``.
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

        prev_state = (
            self.mode,
            self.use_marginal,
            self.alpha,
            self.gamma_z,
            self.gamma_theta,
            self.n_particles,
        )

        self.mode = str(combined_override.get("mode", self._base_profile["mode"]))
        self.steps = int(combined_override.get("steps", self._base_profile["steps"]))
        self.use_marginal = bool(
            combined_override.get("use_marginal", self._base_profile["use_marginal"])
        )
        self.alpha = self._optional_float(
            combined_override.get("alpha", self._base_profile["alpha"])
        )
        self.gamma_z = self._optional_float(
            combined_override.get("gamma_z", self._base_profile["gamma_z"])
        )
        self.gamma_theta = self._optional_float(
            combined_override.get("gamma_theta", self._base_profile["gamma_theta"])
        )
        self.n_particles = self._optional_int(
            combined_override.get("n_particles", self._base_profile["n_particles"])
        )

        next_state = (
            self.mode,
            self.use_marginal,
            self.alpha,
            self.gamma_z,
            self.gamma_theta,
            self.n_particles,
        )
        if next_state != prev_state:
            self._target_cache = None
        self._active_profile = profile_key

    def set_num_nodes(self, num_nodes: int) -> None:
        """Update the active node count for the next explicit inference call."""
        resolved = int(num_nodes)
        if resolved <= 0:
            raise ValueError("num_nodes must be positive.")
        if resolved != self.num_nodes:
            self.num_nodes = resolved
            self._target_cache = None

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
            RuntimeError: Always raised since DiBS is sampled via `sample`.
        """
        raise RuntimeError("DiBSModel does not implement forward(); use sample().")

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample adjacency matrices from the DiBS posterior.

        Args:
            x: Input data tensor of shape (Batch, Samples, Variables).
            num_samples: Number of graph samples to generate per batch element.
            mask: Unused — accepted for interface compatibility.

        Returns:
            Sampled adjacency matrices of shape (Batch, num_samples, Variables, Variables).
        """
        _ = mask
        batch_size, num_nodes = self._validate_sample_input(x)

        if self.external_process:
            return self._sample_external(x, num_samples, batch_size=batch_size)

        return self._sample_in_process(x, num_samples, batch_size=batch_size)

    def _sample_in_process(
        self,
        x: torch.Tensor,
        num_samples: int,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        jax, jnp, dibs_cls, make_target = self._require_dibs()
        num_nodes = self.num_nodes

        jax_platforms = sorted({d.platform for d in jax.devices()})
        log.info(
            "DiBS sample: mode=in_process, jax_platforms=%s, jax_version=%s, "
            "batch_size=%d, num_nodes=%d, num_samples=%d, "
            "mode=%s, steps=%d, n_particles=%s",
            jax_platforms,
            jax.__version__,
            batch_size,
            num_nodes,
            num_samples,
            self.mode,
            self.steps,
            self.n_particles,
        )

        if self._rng_key is None:
            self._rng_key = jax.random.PRNGKey(self.seed)

        samples_per_batch = []
        for batch_idx in range(batch_size):
            self._rng_key, model_key, sample_key = jax.random.split(self._rng_key, 3)
            graph_model, likelihood_model = self._get_target_models(
                make_target=make_target, key=model_key
            )

            x_jax = jnp.asarray(
                x[batch_idx].detach().cpu().numpy().astype(np.float32, copy=False)
            )

            graphs_t = self._sample_from_dibs(
                x_jax=x_jax,
                dibs_cls=dibs_cls,
                graph_model=graph_model,
                likelihood_model=likelihood_model,
                sample_key=sample_key,
                n_particles=self._resolved_n_particles(num_samples),
                steps=int(self.steps),
                num_samples=int(num_samples),
                match_seed=self.seed + batch_idx,
                device=x.device,
                inference_kwargs=self._build_inference_kwargs(
                    alpha=self.alpha,
                    gamma_z=self.gamma_z,
                    gamma_theta=self.gamma_theta,
                    use_marginal=self.use_marginal,
                ),
            )
            samples_per_batch.append(graphs_t)

        return torch.stack(samples_per_batch, dim=0)

    def _sample_external(
        self,
        x: torch.Tensor,
        num_samples: int,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        python_path = self._resolve_external_python()
        script_path = Path(__file__).resolve().with_name("external_infer.py")

        log.info(
            "DiBS sample: mode=external, python=%s, batch_size=%d, num_nodes=%d, "
            "num_samples=%d, profile=%s, mode_name=%s, steps=%d, n_particles=%s, timeout=%ds",
            python_path,
            batch_size,
            self.num_nodes,
            num_samples,
            self._active_profile,
            self.mode,
            self.steps,
            self.n_particles,
            self.external_timeout_s,
        )

        samples_per_batch = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for batch_idx in range(batch_size):
                input_path = tmp_path / f"input_{batch_idx}.npz"
                output_path = tmp_path / f"output_{batch_idx}.npz"
                config_path = tmp_path / f"config_{batch_idx}.json"
                x_np = (
                    x[batch_idx].detach().cpu().numpy().astype(np.float32, copy=False)
                )
                batch_seed = self.seed + self._external_seed_counter + batch_idx

                np.savez(input_path, data=x_np)
                config_payload = self._build_external_config(
                    output_path=output_path,
                    num_samples=int(num_samples),
                    seed=int(batch_seed),
                )
                config_path.write_text(json.dumps(config_payload), encoding="utf-8")

                cmd = [
                    python_path,
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--input",
                    str(input_path),
                ]

                env = os.environ.copy()
                # Ensure JAX in the subprocess initialises the CUDA/GPU
                # backend instead of silently falling back to CPU.
                env.setdefault("JAX_PLATFORMS", "cuda,cpu")
                # Allow a dedicated DiBS/JAX environment to import this checkout
                # without requiring causal_meta to be installed into that env.
                src_root = Path(__file__).resolve().parents[3]
                existing_pythonpath = env.get("PYTHONPATH")
                env["PYTHONPATH"] = (
                    f"{src_root}{os.pathsep}{existing_pythonpath}"
                    if existing_pythonpath
                    else str(src_root)
                )

                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        timeout=self.external_timeout_s,
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        f"DiBS external inference could not launch Python executable: {python_path}"
                    ) from exc
                except subprocess.CalledProcessError as exc:
                    detail_blocks = []
                    for stream_name, stream_value in (
                        ("stdout", exc.stdout),
                        ("stderr", exc.stderr),
                    ):
                        stream_text = (stream_value or "").strip()
                        if stream_text:
                            if len(stream_text) > 4000:
                                stream_text = stream_text[-4000:]
                            detail_blocks.append(f"{stream_name}:\n{stream_text}")
                    detail_suffix = (
                        "\n\n" + "\n\n".join(detail_blocks) if detail_blocks else ""
                    )
                    raise RuntimeError(
                        "DiBS external inference failed using "
                        f"{python_path} (exit code {exc.returncode}). Ensure the selected "
                        "Python environment has 'dibs-lib', JAX, and access to the "
                        f"causal_meta package.{detail_suffix}"
                    ) from exc
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        "DiBS external inference timed out after "
                        f"{self.external_timeout_s} seconds."
                    ) from exc

                result = np.load(output_path)
                graph_samples = result["graph_samples"]
                graphs_t = torch.from_numpy(graph_samples.copy()).to(
                    device=x.device,
                    dtype=torch.float32,
                )
                samples_per_batch.append(graphs_t)

        self._external_seed_counter += batch_size
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
            RuntimeError: Always raised since DiBS does not expose a training loss here.
        """
        _ = output
        _ = target
        _ = kwargs
        raise RuntimeError("DiBSModel does not implement calculate_loss().")

    def _get_target_models(
        self, make_target: Callable[..., Tuple[Any, Any]], key: Any
    ) -> Tuple[Any, Any]:
        if self._target_cache is None:
            if self.mode not in {"linear", "nonlinear"}:
                raise ValueError("DiBS mode must be 'linear' or 'nonlinear'.")
            result = make_target(
                key=key,
                n_vars=self.num_nodes,
                **self._build_target_kwargs(make_target),
            )
            graph_model, likelihood_model = self._extract_target_models(result)
            self._target_cache = (graph_model, likelihood_model)
        return self._target_cache

    def _build_external_config(
        self,
        *,
        output_path: Path,
        num_samples: int,
        seed: int,
    ) -> dict[str, Any]:
        return {
            "num_nodes": int(self.num_nodes),
            "mode": str(self.mode),
            "steps": int(self.steps),
            "seed": int(seed),
            "use_marginal": bool(self.use_marginal),
            "xla_preallocate": bool(self.xla_preallocate),
            "alpha": self.alpha,
            "gamma_z": self.gamma_z,
            "gamma_theta": self.gamma_theta,
            "n_particles": self.n_particles,
            "num_samples": int(num_samples),
            "output": str(output_path),
        }

    def _resolve_external_python(self) -> str:
        if not self.external_python:
            return sys.executable
        python_path = Path(os.path.expandvars(self.external_python)).expanduser()
        if not python_path.is_absolute():
            python_path = (Path.cwd() / python_path).resolve()
        if not python_path.exists():
            raise FileNotFoundError(
                f"Resolved DiBS external_python does not exist: {python_path}"
            )
        return str(python_path)

    def _resolved_n_particles(self, num_samples: int) -> int:
        return self._resolve_n_particles_value(self.n_particles, num_samples)

    def _validate_sample_input(self, x: torch.Tensor) -> tuple[int, int]:
        if x.ndim != 3:
            raise ValueError("Input data must have shape (Batch, Samples, Variables).")

        batch_size, _, num_nodes = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(
                "Input data node count does not match configured num_nodes."
            )
        return int(batch_size), int(num_nodes)

    @staticmethod
    def _build_inference_kwargs(
        *,
        alpha: float | None,
        gamma_z: float | None,
        gamma_theta: float | None,
        use_marginal: bool,
    ) -> dict[str, Any]:
        """Build kwargs for the ``JointDiBS`` / ``MarginalDiBS`` constructor.

        Maps config-level inference parameters to the library's constructor
        keyword arguments:

        - ``alpha``       → ``alpha_linear``
        - ``gamma_z``     → ``kernel_param["h_latent"]`` (Joint) or
                            ``kernel_param["h"]`` (Marginal)
        - ``gamma_theta`` → ``kernel_param["h_theta"]`` (Joint only)
        """
        kwargs: dict[str, Any] = {}
        if alpha is not None:
            kwargs["alpha_linear"] = float(alpha)

        kernel_param: dict[str, float] = {}
        if use_marginal:
            if gamma_z is not None:
                kernel_param["h"] = float(gamma_z)
        else:
            if gamma_z is not None:
                kernel_param["h_latent"] = float(gamma_z)
            if gamma_theta is not None:
                kernel_param["h_theta"] = float(gamma_theta)

        if kernel_param:
            kwargs["kernel_param"] = kernel_param

        return kwargs

    @staticmethod
    def _sample_from_dibs(
        *,
        x_jax: Any,
        dibs_cls: Any,
        graph_model: Any,
        likelihood_model: Any,
        sample_key: Any,
        n_particles: int,
        steps: int,
        num_samples: int,
        match_seed: int,
        device: torch.device,
        inference_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        dibs = dibs_cls(
            x=x_jax,
            interv_mask=None,
            graph_model=graph_model,
            likelihood_model=likelihood_model,
            **(inference_kwargs or {}),
        )

        sample_result = dibs.sample(
            key=sample_key,
            n_particles=int(n_particles),
            steps=int(steps),
        )

        if isinstance(sample_result, tuple):
            graphs = sample_result[0]
        else:
            graphs = sample_result

        graphs_np = np.asarray(graphs)
        graphs_t = torch.from_numpy(graphs_np.copy()).to(device=device)
        graphs_t = (graphs_t > 0.5).to(dtype=torch.float32)
        return DiBSModel._match_num_samples(
            graphs_t,
            target_samples=int(num_samples),
            seed=int(match_seed),
        )

    @classmethod
    def sample_numpy_array(
        cls,
        *,
        data: np.ndarray,
        num_nodes: int,
        num_samples: int,
        mode: str,
        steps: int,
        seed: int,
        use_marginal: bool,
        xla_preallocate: bool,
        alpha: float | None,
        gamma_z: float | None,
        gamma_theta: float | None,
        n_particles: int | None,
    ) -> np.ndarray:
        jax, jnp, dibs_cls, make_target = cls._require_dibs_static(
            mode=mode,
            use_marginal=use_marginal,
            xla_preallocate=xla_preallocate,
        )
        key = jax.random.PRNGKey(int(seed))
        key, model_key, sample_key = jax.random.split(key, 3)
        target_result = make_target(
            key=model_key,
            n_vars=int(num_nodes),
            **cls._build_target_kwargs_for(
                make_target,
                alpha=alpha,
                gamma_z=gamma_z,
                gamma_theta=gamma_theta,
            ),
        )
        graph_model, likelihood_model = cls._extract_target_models(target_result)
        x_jax = jnp.asarray(np.asarray(data, dtype=np.float32, copy=False))
        graphs_t = cls._sample_from_dibs(
            x_jax=x_jax,
            dibs_cls=dibs_cls,
            graph_model=graph_model,
            likelihood_model=likelihood_model,
            sample_key=sample_key,
            n_particles=cls._resolve_n_particles_value(n_particles, num_samples),
            steps=int(steps),
            num_samples=int(num_samples),
            match_seed=int(seed),
            device=torch.device("cpu"),
            inference_kwargs=cls._build_inference_kwargs(
                alpha=alpha,
                gamma_z=gamma_z,
                gamma_theta=gamma_theta,
                use_marginal=use_marginal,
            ),
        )
        return graphs_t.cpu().numpy()

    def _build_target_kwargs(
        self,
        make_target: Callable[..., Any],
    ) -> dict[str, float]:
        """Build optional target kwargs supported by the installed DiBS version."""
        return self._build_target_kwargs_for(
            make_target,
            alpha=self.alpha,
            gamma_z=self.gamma_z,
            gamma_theta=self.gamma_theta,
        )

    @classmethod
    def _build_target_kwargs_for(
        cls,
        make_target: Callable[..., Any],
        *,
        alpha: float | None,
        gamma_z: float | None,
        gamma_theta: float | None,
    ) -> dict[str, float]:
        params = set(inspect.signature(make_target).parameters.keys())
        kwargs: dict[str, float] = {}

        cls._set_first_supported_param(
            kwargs,
            params,
            candidates=("alpha_linear", "alpha"),
            value=alpha,
        )
        cls._set_first_supported_param(
            kwargs,
            params,
            candidates=("gamma_z", "gamma_latent"),
            value=gamma_z,
        )
        cls._set_first_supported_param(
            kwargs,
            params,
            candidates=("gamma_theta", "gamma"),
            value=gamma_theta,
        )

        return kwargs

    @staticmethod
    def _extract_target_models(result: Any) -> tuple[Any, Any]:
        result_tuple = cast(tuple[Any, ...], result)
        if len(result_tuple) == 3:
            _, graph_model, likelihood_model = result_tuple
        else:
            graph_model, likelihood_model = result_tuple
        return graph_model, likelihood_model

    @staticmethod
    def _set_first_supported_param(
        kwargs: dict[str, float],
        supported: set[str],
        *,
        candidates: tuple[str, ...],
        value: float | None,
    ) -> None:
        if value is None:
            return
        for name in candidates:
            if name in supported:
                kwargs[name] = float(value)
                return

    @staticmethod
    def _match_num_samples(
        graphs_t: torch.Tensor,
        *,
        target_samples: int,
        seed: int,
    ) -> torch.Tensor:
        """Ensure DiBS output has exactly ``target_samples`` graph draws."""
        if target_samples < 1:
            raise ValueError("target_samples must be >= 1.")
        available = int(graphs_t.shape[0])
        if available == target_samples:
            return graphs_t
        if available > target_samples:
            generator = torch.Generator(device=graphs_t.device)
            generator.manual_seed(int(seed))
            indices = torch.randperm(
                available,
                device=graphs_t.device,
                generator=generator,
            )[:target_samples]
            return graphs_t.index_select(0, indices)

        # Upsample with replacement (randperm only has `available` elements,
        # so it cannot produce `target_samples > available`).
        generator = torch.Generator(device=graphs_t.device)
        generator.manual_seed(int(seed))
        indices = torch.randint(
            low=0,
            high=available,
            size=(target_samples,),
            device=graphs_t.device,
            generator=generator,
        )
        return graphs_t.index_select(0, indices)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _resolve_n_particles_value(n_particles: int | None, num_samples: int) -> int:
        resolved = int(n_particles) if n_particles is not None else int(num_samples)
        if resolved < 1:
            raise ValueError("DiBS n_particles must be >= 1.")
        return resolved

    def _require_dibs(self) -> Tuple[Any, Any, Any, Callable[..., Any]]:
        return self._require_dibs_static(
            mode=self.mode,
            use_marginal=self.use_marginal,
            xla_preallocate=self.xla_preallocate,
        )

    @staticmethod
    def _require_dibs_static(
        *,
        mode: str,
        use_marginal: bool,
        xla_preallocate: bool,
    ) -> Tuple[Any, Any, Any, Callable[..., Any]]:
        if not xla_preallocate:
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        try:
            import jax
            import jax.numpy as jnp
            from dibs.inference import JointDiBS, MarginalDiBS
            from dibs.target import (
                make_linear_gaussian_equivalent_model,
                make_linear_gaussian_model,
                make_nonlinear_gaussian_model,
            )
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError(
                "DiBS requires the 'dibs-lib' package and JAX. "
                "Install with `pip install dibs-lib` and ensure a compatible JAX backend."
            ) from exc

        dibs_cls = MarginalDiBS if use_marginal else JointDiBS
        make_target = DiBSModel._select_target_factory(
            mode=mode,
            use_marginal=use_marginal,
            make_linear_gaussian_model=make_linear_gaussian_model,
            make_linear_gaussian_equivalent_model=make_linear_gaussian_equivalent_model,
            make_nonlinear_gaussian_model=make_nonlinear_gaussian_model,
        )

        return jax, jnp, dibs_cls, make_target

    @staticmethod
    def _select_target_factory(
        *,
        mode: str,
        use_marginal: bool,
        make_linear_gaussian_model: Callable[..., Any],
        make_linear_gaussian_equivalent_model: Callable[..., Any],
        make_nonlinear_gaussian_model: Callable[..., Any],
    ) -> Callable[..., Any]:
        if mode == "linear":
            if use_marginal:
                return make_linear_gaussian_equivalent_model
            return make_linear_gaussian_model
        if mode == "nonlinear":
            if use_marginal:
                raise ValueError(
                    "DiBS marginal inference is only supported for linear Gaussian targets with a BGe score."
                )
            return make_nonlinear_gaussian_model
        raise ValueError("DiBS mode must be 'linear' or 'nonlinear'.")
