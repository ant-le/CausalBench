from __future__ import annotations

import gzip
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import logging

log = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _cache_cfg(cfg: Any) -> Mapping[str, Any]:
    # Support DictConfig-like objects as well as plain dicts.
    inference = getattr(cfg, "inference", None)
    if inference is None:
        return {}
    if isinstance(inference, Mapping):
        return inference
    return dict(inference)  # type: ignore[arg-type]


def cache_suffix(*, compress: bool) -> str:
    return ".pt.gz" if compress else ".pt"


def cache_settings(cfg: Any) -> tuple[bool, str, int | None]:
    inf = _cache_cfg(cfg)
    compress = bool(inf.get("cache_compress", False))
    dtype = str(inf.get("cache_dtype", "float32")).lower()
    max_samples_raw = inf.get("cache_n_samples", None)
    max_samples = int(max_samples_raw) if max_samples_raw is not None else None
    if max_samples is not None and max_samples < 1:
        max_samples = None
    return compress, dtype, max_samples


def prepare_graph_samples_for_cache(
    samples: torch.Tensor, *, dtype: str, max_samples: int | None
) -> torch.Tensor:
    # samples: (B, K, N, N)
    if samples.ndim != 4:
        raise ValueError("Expected graph samples of shape (B, K, N, N).")

    if max_samples is not None:
        samples = samples[:, :max_samples]

    dtype_norm = dtype.lower()
    if dtype_norm in {"float32", "fp32"}:
        return samples.to(dtype=torch.float32)
    if dtype_norm in {"float16", "fp16"}:
        return samples.to(dtype=torch.float16)
    if dtype_norm in {"bfloat16", "bf16"}:
        return samples.to(dtype=torch.bfloat16)
    if dtype_norm in {"uint8", "u8"}:
        return (samples > 0.5).to(dtype=torch.uint8)
    if dtype_norm in {"bool", "boolean"}:
        return (samples > 0.5).to(dtype=torch.bool)

    raise ValueError(
        f"Unsupported inference.cache_dtype='{dtype}'. "
        "Use one of: float32, float16, bfloat16, uint8, bool."
    )


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{path}.tmp.{os.getpid()}")
    try:
        if path.suffix == ".gz":
            buffer = io.BytesIO()
            torch.save(obj, buffer)
            buffer.seek(0)
            with gzip.open(tmp_path, "wb") as f:
                f.write(buffer.getbuffer())
        else:
            torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def torch_load(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            buffer = io.BytesIO(f.read())
            return torch.load(buffer, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_output_dir(cfg: Any, output_dir: Path | str | None = None) -> Path:
    """Resolve the base output directory for a run.

    Args:
        cfg: Experiment configuration (DictConfig or dict-like).
        output_dir: Optional explicit output directory override.

    Returns:
        Base output directory for the run.
    """
    if output_dir is not None:
        return Path(output_dir)

    inference_cfg = _cache_cfg(cfg)
    override = inference_cfg.get("output_dir", None)
    if override:
        return Path(str(override))

    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        return Path(os.getcwd())


def get_model_name(cfg: Any, model: Any | None = None) -> str:
    """Derive a stable model identifier for artifact names.

    Args:
        cfg: Experiment configuration (DictConfig or dict-like).
        model: Optional model instance for fallback identification.

    Returns:
        Model identifier used for artifact names.
    """
    model_cfg = getattr(cfg, "model", None)
    if isinstance(model_cfg, Mapping):
        model_id = model_cfg.get("id", None)
        if model_id is not None:
            return str(model_id)
        model_type = model_cfg.get("type", None)
    else:
        model_id = getattr(model_cfg, "id", None)
        if model_id is not None:
            return str(model_id)
        model_type = getattr(model_cfg, "type", None)
    if model_type is not None:
        return str(model_type)

    if model is not None:
        from causal_meta.models.factory import MODEL_REGISTRY

        for name, cls in MODEL_REGISTRY.items():
            try:
                if isinstance(model, cls):
                    return str(name)
            except Exception:
                continue

    return "model"


def find_inference_artifact(
    inference_root: Path,
    *,
    dataset_name: str,
    model_name: str,
    seed: int,
    prefer_compress: bool,
    use_model_subdir: bool,
) -> Path | None:
    # New layout (per-run artifacts): inference/<dataset>/seed_*.pt
    # Old/shared layout (persistent cache): inference/<model>/<dataset>/seed_*.pt
    base = inference_root / model_name if use_model_subdir else inference_root
    preferred = (
        base / dataset_name / f"seed_{seed}{cache_suffix(compress=prefer_compress)}"
    )
    if preferred.exists():
        return preferred
    fallback = (
        base / dataset_name / f"seed_{seed}{cache_suffix(compress=not prefer_compress)}"
    )
    if fallback.exists():
        return fallback
    return None
