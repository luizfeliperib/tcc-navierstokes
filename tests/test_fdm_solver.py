"""Testes do controlador de simulacao FDM-06.7."""

from dataclasses import replace

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.solver import solve


def test_fixed_number_of_steps():
    config = SimulationConfig(
        nx=8,
        ny=8,
        dt=0.001,
        max_iterations=10,
    )

    result = solve(
        config,
        stop_on_steady=False,
        record_interval=3,
    )

    assert result.steps == 10
    assert result.final_time == pytest.approx(0.01)
    assert not result.converged
    assert result.stop_reason == "max_steps"
    assert [r.step for r in result.history] == [1, 3, 6, 9, 10]

    assert result.state.u.shape == (8, 8)
    assert result.state.u_faces.shape == (8, 9)
    assert result.state.v_faces.shape == (9, 8)


def test_history_and_diagnostics_are_finite():
    config = SimulationConfig(
        nx=8,
        ny=8,
        max_iterations=5,
    )

    result = solve(config, stop_on_steady=False)

    assert result.history[-1].step == 5

    for record in result.history:
        assert np.isfinite(record.steady_residual)
        assert np.isfinite(record.poisson_residual)
        assert record.diagnostics.divergence_rms < 1e-8


def test_invalid_record_interval():
    config = SimulationConfig(nx=8, ny=8)

    with pytest.raises(ValueError, match="record_interval"):
        solve(config, record_interval=0)


def test_invalid_max_steps():
    config = replace(
        SimulationConfig(nx=8, ny=8),
        max_iterations=0,
    )

    with pytest.raises(ValueError, match="max_iterations"):
        solve(config)


def test_numba_matches_python():
    pytest.importorskip("numba")

    config = SimulationConfig(
        nx=8,
        ny=8,
        max_iterations=10,
    )

    python_result = solve(
        config,
        backend="python",
        stop_on_steady=False,
    )
    numba_result = solve(
        config,
        backend="numba",
        stop_on_steady=False,
    )

    for name in ("u", "v", "u_faces", "v_faces", "pressure"):
        np.testing.assert_allclose(
            getattr(python_result.state, name),
            getattr(numba_result.state, name),
            rtol=1e-10,
            atol=1e-11,
        )


def test_steady_stop_requires_consecutive_steps():
    config = SimulationConfig(
        nx=8,
        ny=8,
        tolerance=1e6,
        max_iterations=20,
    )

    result = solve(
        config,
        stop_on_steady=True,
        min_steps=5,
        steady_consecutive_steps=3,
    )

    assert result.converged
    assert result.steps == 5
    assert result.stop_reason == "steady_tolerance"
    assert result.history[-1].step == 5
