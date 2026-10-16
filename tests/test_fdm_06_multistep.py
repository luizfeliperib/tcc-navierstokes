"""FDM-06.6: testes integrados de múltiplos passos."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.diagnostics import calculate_diagnostics
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.time_step import FlowState, advance_one_step


def initial_state(grid):
    return FlowState(
        u=grid.zeros_scalar(),
        v=grid.zeros_scalar(),
        u_faces=grid.zeros_u_faces(),
        v_faces=grid.zeros_v_faces(),
        pressure=grid.zeros_scalar(),
    )


def run_steps(config, number_of_steps=20):
    """Executa passos e registra métricas de cada um."""
    grid = create_grid(config)
    state = initial_state(grid)
    records = []

    for step in range(1, number_of_steps + 1):
        result = advance_one_step(
            state,
            grid,
            config,
            pressure_tolerance=1e-9,
            pressure_max_iterations=10_000,
        )

        state = result.state
        diag = calculate_diagnostics(state, grid)

        records.append((step, diag, result.pressure_result))

    return grid, state, records


def test_multistep_mass_conservation():
    config = SimulationConfig(nx=8, ny=8, dt=0.001, reynolds=100.0)
    _, _, records = run_steps(config)

    for _, diag, pressure_result in records:
        assert pressure_result.converged
        assert diag.divergence_rms < 1e-9


def test_multistep_fields_remain_finite():
    config = SimulationConfig(nx=8, ny=8, dt=0.001, reynolds=100.0)
    _, state, records = run_steps(config)

    for field in (
        state.u,
        state.v,
        state.u_faces,
        state.v_faces,
        state.pressure,
    ):
        assert np.all(np.isfinite(field))

    for _, diag, _ in records:
        assert np.isfinite(diag.max_abs_u)
        assert np.isfinite(diag.max_abs_v)
        assert np.isfinite(diag.kinetic_energy_mean)


def test_multistep_lid_generates_motion():
    config = SimulationConfig(nx=8, ny=8, dt=0.001, reynolds=100.0)
    _, state, records = run_steps(config)

    assert np.max(state.u[-1, :]) > 0.0
    assert records[-1][1].kinetic_energy_mean > 0.0


def test_multistep_center_face_discrepancy_is_finite():
    config = SimulationConfig(nx=8, ny=8, dt=0.001, reynolds=100.0)
    _, _, records = run_steps(config)

    for _, diag, _ in records:
        assert np.isfinite(diag.center_face_u_rms)
        assert np.isfinite(diag.center_face_v_rms)

    # Não exigimos discrepância nula: são representações
    # discretas diferentes do mesmo campo de velocidade.


def test_multistep_pressure_converges():
    config = SimulationConfig(nx=8, ny=8, dt=0.001, reynolds=100.0)
    _, _, records = run_steps(config)

    for _, _, pressure_result in records:
        assert pressure_result.converged
        assert pressure_result.residual <= 1e-9


def test_diagnostics_reject_invalid_field():
    config = SimulationConfig(nx=8, ny=8)
    grid = create_grid(config)
    state = initial_state(grid)

    state.u[0, 0] = np.nan

    with pytest.raises(ValueError, match="u"):
        calculate_diagnostics(state, grid)
