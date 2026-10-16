"""Testes dos diagnósticos espaciais centro-face."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.consistency import calculate_consistency
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.time_step import FlowState


def make_state(grid):
    return FlowState(
        u=np.zeros(grid.scalar_shape),
        v=np.zeros(grid.scalar_shape),
        u_faces=np.zeros(grid.u_face_shape),
        v_faces=np.zeros(grid.v_face_shape),
        pressure=np.zeros(grid.scalar_shape),
    )


def test_zero_fields_have_zero_discrepancy():
    grid = create_grid(SimulationConfig(nx=8, ny=8))
    result = calculate_consistency(make_state(grid), grid)

    assert result.u_total == 0.0
    assert result.u_interior == 0.0
    assert result.u_boundary == 0.0
    assert result.v_total == 0.0
    assert result.v_interior == 0.0
    assert result.v_boundary == 0.0


def test_interior_discrepancy_is_separated():
    grid = create_grid(SimulationConfig(nx=8, ny=8))
    state = make_state(grid)
    state.u[2, 2] = 1.0

    result = calculate_consistency(state, grid)

    assert result.u_interior > 0.0
    assert result.u_boundary == 0.0
    assert result.v_total == 0.0


def test_boundary_discrepancy_is_separated():
    grid = create_grid(SimulationConfig(nx=8, ny=8))
    state = make_state(grid)
    state.v[0, 3] = 1.0

    result = calculate_consistency(state, grid)

    assert result.v_boundary > 0.0
    assert result.v_interior == 0.0
    assert result.u_total == 0.0


def test_total_rms_respects_region_weights():
    grid = create_grid(SimulationConfig(nx=8, ny=8))
    state = make_state(grid)
    state.u[2, 2] = 1.0
    state.u[0, 0] = 2.0

    result = calculate_consistency(state, grid)

    n_total = grid.nx * grid.ny
    n_interior = (grid.nx - 2) * (grid.ny - 2)
    n_boundary = n_total - n_interior

    expected_squared = (
        n_interior * result.u_interior**2 + n_boundary * result.u_boundary**2
    ) / n_total

    assert result.u_total**2 == pytest.approx(expected_squared)


def test_relative_discrepancy():
    grid = create_grid(SimulationConfig(nx=8, ny=8))
    state = make_state(grid)
    state.u[2, 2] = 1.0

    result = calculate_consistency(state, grid)

    assert result.u_relative_percent == pytest.approx(12.5)


def test_invalid_field_is_rejected():
    grid = create_grid(SimulationConfig(nx=8, ny=8))
    state = make_state(grid)
    state.u[0, 0] = np.nan

    with pytest.raises(ValueError):
        calculate_consistency(state, grid)


def test_grid_too_small_is_rejected():
    grid = create_grid(SimulationConfig(nx=2, ny=8))

    with pytest.raises(ValueError, match="nx >= 3"):
        calculate_consistency(make_state(grid), grid)
