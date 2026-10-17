"""Testes da exportação de resultados FDM-07.1."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

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
