
"""Verificacao final da integracao FDM-06.7."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.solver import initialize_state, solve
from navierstokes.fdm.time_step import advance_one_step


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_solver_matches_manual_time_stepping(backend):
    """Compara solve() com 100 passos executados manualmente."""

    if backend == "numba":
        pytest.importorskip("numba")

    config = SimulationConfig(
        reynolds=100.0,
        nx=8,
        ny=8,
        dt=0.001,
        max_iterations=100,
    )

    pressure_tolerance = 1e-9
    pressure_max_iterations = 10_000

    # Caminho 1: solver completo.
    simulation = solve(
        config,
        backend=backend,
        pressure_tolerance=pressure_tolerance,
        pressure_max_iterations=pressure_max_iterations,
        stop_on_steady=False,
        record_interval=10,
    )

    # Caminho 2: execucao manual dos mesmos 100 passos.
    grid = create_grid(config)
    state = initialize_state(grid)

    for _ in range(config.max_iterations):
        result = advance_one_step(
            state,
            grid,
            config,
            pressure_tolerance=pressure_tolerance,
            pressure_max_iterations=pressure_max_iterations,
            backend=backend,
        )
        state = result.state

    # Verificacao do controle temporal.
    assert simulation.steps == 100
    assert simulation.final_time == pytest.approx(0.1)
    assert simulation.stop_reason == "max_steps"
    assert not simulation.converged

    # Comparacao de todos os campos numericos.
    for field_name in (
        "u",
        "v",
        "u_faces",
        "v_faces",
        "pressure",
    ):
        actual = getattr(simulation.state, field_name)
        expected = getattr(state, field_name)

        np.testing.assert_array_equal(
            actual,
            expected,
            err_msg=f"Divergencia no campo {field_name}",
        )

    # Verificacao do historico.
    assert [record.step for record in simulation.history] == [
        1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    ]

    assert simulation.history[-1].time == pytest.approx(0.1)

    # Verifica a conservacao discreta de massa nas faces.
    assert (
        simulation.history[-1].diagnostics.divergence_rms
        < 1e-8
    )
