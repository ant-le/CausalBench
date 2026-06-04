import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_meta.models.bayesdag.model import BayesDAGModel
from causal_meta.models.dibs.model import DiBSModel


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_dibs_wrapper_dependency_contract() -> None:
    model = DiBSModel(num_nodes=3)
    x = torch.zeros(1, 4, 3)

    if _module_available("dibs") and _module_available("jax"):
        # Installed path: import contract should succeed without running expensive sampling.
        jax, jnp, dibs_cls, make_target = model._require_dibs()
        assert jax is not None
        assert jnp is not None
        assert dibs_cls is not None
        assert callable(make_target)
    else:
        # Missing dependency path: sample should fail with actionable message.
        with pytest.raises(RuntimeError, match="dibs-lib"):
            _ = model.sample(x)


def test_dibs_select_target_factory_uses_bge_for_linear_marginal() -> None:
    linear = object()
    linear_bge = object()
    nonlinear = object()

    selected = DiBSModel._select_target_factory(
        mode="linear",
        use_marginal=True,
        make_linear_gaussian_model=linear,
        make_linear_gaussian_equivalent_model=linear_bge,
        make_nonlinear_gaussian_model=nonlinear,
    )

    assert selected is linear_bge


def test_dibs_select_target_factory_rejects_nonlinear_marginal() -> None:
    with pytest.raises(ValueError, match="only supported for linear Gaussian"):
        DiBSModel._select_target_factory(
            mode="nonlinear",
            use_marginal=True,
            make_linear_gaussian_model=object(),
            make_linear_gaussian_equivalent_model=object(),
            make_nonlinear_gaussian_model=object(),
        )


def test_dibs_external_python_expands_user_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_dir = tmp_path / "home"
    python_path = home_dir / ".venv-dibs" / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("#!/usr/bin/env python3\n")

    monkeypatch.setenv("HOME", str(home_dir))

    model = DiBSModel(
        num_nodes=3,
        external_process=True,
        external_python="~/.venv-dibs/bin/python",
    )

    assert model._resolve_external_python() == str(python_path.resolve())


def test_dibs_external_python_missing_path_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist" / "python"
    model = DiBSModel(
        num_nodes=3,
        external_process=True,
        external_python=str(missing_path),
    )

    with pytest.raises(FileNotFoundError, match="external_python does not exist"):
        _ = model._resolve_external_python()


def test_dibs_external_process_writes_and_reads_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DiBSModel(
        num_nodes=3,
        mode="linear",
        steps=123,
        n_particles=7,
        external_process=True,
    )
    x = torch.zeros(1, 4, 3)

    calls: list[dict[str, object]] = []
    expected_src_root = str((Path(__file__).resolve().parents[2] / "src").resolve())

    def _fake_run(cmd, check, timeout, env, capture_output, text):
        _ = check
        _ = capture_output
        _ = text
        config_index = cmd.index("--config") + 1
        input_index = cmd.index("--input") + 1
        with open(cmd[config_index], "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        input_payload = np.load(cmd[input_index])
        output_path = Path(payload["output"])
        graph_samples = np.ones(
            (int(payload["num_samples"]), 3, 3),
            dtype=np.float32,
        )
        np.savez(output_path, graph_samples=graph_samples)
        calls.append(
            {
                "timeout": timeout,
                "payload": payload,
                "input_shape": tuple(input_payload["data"].shape),
                "python": cmd[0],
                "pythonpath": env["PYTHONPATH"],
            }
        )

    monkeypatch.setattr("causal_meta.models.dibs.model.subprocess.run", _fake_run)

    samples = model.sample(x, num_samples=5)

    assert tuple(samples.shape) == (1, 5, 3, 3)
    assert torch.equal(samples, torch.ones_like(samples))
    assert len(calls) == 1
    payload = calls[0]["payload"]
    assert payload["mode"] == "linear"
    assert payload["steps"] == 123
    assert payload["n_particles"] == 7
    assert payload["num_nodes"] == 3
    assert calls[0]["input_shape"] == (4, 3)
    assert str(calls[0]["pythonpath"]).split(":", maxsplit=1)[0] == expected_src_root


def test_dibs_external_process_surfaces_subprocess_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DiBSModel(
        num_nodes=3,
        external_process=True,
    )
    x = torch.zeros(1, 4, 3)

    def _fake_run(cmd, check, timeout, env, capture_output, text):
        _ = cmd
        _ = check
        _ = timeout
        _ = env
        _ = capture_output
        _ = text
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            stderr="ModuleNotFoundError: No module named 'dibs'",
        )

    monkeypatch.setattr("causal_meta.models.dibs.model.subprocess.run", _fake_run)

    with pytest.raises(RuntimeError, match="ModuleNotFoundError") as exc_info:
        _ = model.sample(x)

    assert "causal_meta package" in str(exc_info.value)


def test_bayesdag_wrapper_requires_dependency() -> None:
    if _module_available("causica"):
        pytest.skip("causica is installed; skipping missing-dependency check.")

    model = BayesDAGModel(num_nodes=3)
    x = torch.zeros(1, 4, 3)

    with pytest.raises(RuntimeError, match="Project-BayesDAG"):
        _ = model.sample(x)


def test_bayesdag_external_python_expands_user_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_dir = tmp_path / "home"
    python_path = home_dir / ".venv-bayesdag" / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("#!/usr/bin/env python3\n")

    monkeypatch.setenv("HOME", str(home_dir))

    model = BayesDAGModel(num_nodes=3, external_python="~/.venv-bayesdag/bin/python")

    assert model._resolve_external_python() == str(python_path.resolve())


def test_bayesdag_external_python_missing_path_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist" / "python"
    model = BayesDAGModel(num_nodes=3, external_python=str(missing_path))

    with pytest.raises(FileNotFoundError, match="external_python does not exist"):
        _ = model._resolve_external_python()
