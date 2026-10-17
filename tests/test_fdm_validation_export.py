"""Testes da exportação de resultados FDM-07.1."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.solver import solve


def load_export_function():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_fdm_validation.py"
    spec = importlib.util.spec_from_file_location("run_fdm_validation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_results


def test_export_results(tmp_path):
    config = SimulationConfig(
        reynolds=100.0,
        nx=8,
        ny=6,
        dt=0.001,
        max_iterations=3,
    )

    result = solve(
        config,
        backend="python",
        stop_on_steady=False,
        record_interval=1,
    )

    export_results = load_export_function()
    export_results(
        result,
        config,
        tmp_path,
        elapsed=0.5,
        settings={"backend": "python"},
    )

    with np.load(tmp_path / "fields.npz") as data:
        assert data["u"].shape == (6, 8)
        assert data["v"].shape == (6, 8)
        assert data["pressure"].shape == (6, 8)
        assert data["u_faces"].shape == (6, 9)
        assert data["v_faces"].shape == (7, 8)

        np.testing.assert_array_equal(data["u"], result.state.u)
        np.testing.assert_array_equal(data["pressure"], result.state.pressure)
        np.testing.assert_array_equal(data["x"], result.grid.x)
        np.testing.assert_array_equal(data["y"], result.grid.y)

    with (tmp_path / "history.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == len(result.history)
    assert int(rows[-1]["step"]) == result.steps
    assert "divergence_rms" in rows[-1]

    with (tmp_path / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["configuration"]["reynolds"] == 100.0
    assert metadata["result"]["steps"] == 3
    assert metadata["result"]["converged"] is False

    # Verifica os arquivos dos perfis centrais.
    u_path = tmp_path / "centerline_u.csv"
    v_path = tmp_path / "centerline_v.csv"

    assert u_path.exists()
    assert v_path.exists()

    with u_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        u_rows = list(csv.DictReader(file))

    with v_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        v_rows = list(csv.DictReader(file))

    assert len(u_rows) == config.ny
    assert len(v_rows) == config.nx

    assert list(u_rows[0]) == ["y", "u"]
    assert list(v_rows[0]) == ["x", "v"]

    np.testing.assert_allclose(
        [float(row["y"]) for row in u_rows],
        result.grid.y,
    )

    np.testing.assert_allclose(
        [float(row["x"]) for row in v_rows],
        result.grid.x,
    )


def test_plot_results(tmp_path):
    import importlib.util
    from pathlib import Path

    config = SimulationConfig(
        reynolds=100.0,
        nx=8,
        ny=8,
        dt=0.001,
        max_iterations=10,
    )

    result = solve(
        config,
        stop_on_steady=False,
        record_interval=1,
    )

    export_results = load_export_function()
    export_results(
        result,
        config,
        tmp_path,
        elapsed=0.5,
        settings={"backend": "python"},
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_fdm_validation.py"

    spec = importlib.util.spec_from_file_location("plot_fdm_validation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.plot_results(tmp_path)

    for name in (
        "velocity_magnitude",
        "streamlines",
        "velocity_vectors",
        "steady_convergence",
    ):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0
        assert (tmp_path / "figures" / f"{name}.pdf").stat().st_size > 0


def test_consolidated_parser_defaults():
    """Confere os parâmetros padrão do novo executor."""

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_fdm_validation.py"

    spec = importlib.util.spec_from_file_location(
        "run_fdm_validation_parser",
        script,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args([])

    assert args.reynolds == 100.0
    assert args.nx == 32
    assert args.ny == 32
    assert args.dt == 0.001
    assert args.steps == 2000

    assert args.backend == "numba"
    assert args.pressure_tolerance == 1e-9
    assert args.pressure_max_iterations == 20000
    assert args.residual_check_interval == 20

    assert args.steady_tolerance == 1e-5
    assert args.steady_consecutive_steps == 20
    assert args.min_steps == 100
    assert args.record_interval == 20


def test_consolidated_metadata(tmp_path):
    """Confere a rastreabilidade das opções de execução."""

    config = SimulationConfig(
        reynolds=100.0,
        nx=8,
        ny=8,
        dt=0.001,
        max_iterations=3,
    )

    settings = {
        "backend": "numba",
        "pressure_tolerance": 1e-9,
        "pressure_max_iterations": 20000,
        "residual_check_interval": 20,
        "record_interval": 1,
        "stop_on_steady": False,
        "steady_consecutive_steps": 20,
        "min_steps": 100,
    }

    result = solve(
        config,
        backend="numba",
        pressure_tolerance=1e-9,
        pressure_max_iterations=20000,
        residual_check_interval=20,
        record_interval=1,
        stop_on_steady=False,
    )

    export_results = load_export_function()

    export_results(
        result,
        config,
        tmp_path,
        elapsed=0.5,
        settings=settings,
    )

    with (tmp_path / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["execution_settings"] == settings

    assert metadata["result"]["steps"] == 3

    assert metadata["result"]["stop_reason"] == "max_steps"

    assert metadata["result"]["converged"] is False

    assert metadata["result"]["seconds_per_step"] == pytest.approx(0.5 / 3)

    assert (tmp_path / "fields.npz").exists()
    assert (tmp_path / "history.csv").exists()
