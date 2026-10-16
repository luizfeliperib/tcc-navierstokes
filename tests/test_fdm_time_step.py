
"""Testes da integração temporal FDM-06.5."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.operators import divergence_faces
from navierstokes.fdm.time_step import (
    FlowState,
    advance_one_step,
)


def initial_state(grid):
    return FlowState(
        u=grid.zeros_scalar(),
        v=grid.zeros_scalar(),
        u_faces=grid.zeros_u_faces(),
        v_faces=grid.zeros_v_faces(),
        pressure=grid.zeros_scalar(),
    )


def test_one_step_from_rest_with_moving_lid():
    config = SimulationConfig(
        nx=8, ny=8, dt=0.001, reynolds=100.0
    )
    grid = create_grid(config)
    state = initial_state(grid)

    result = advance_one_step(
        state, grid, config,
        pressure_tolerance=1e-10,
    )

    assert result.pressure_result.converged
    assert np.all(np.isfinite(result.state.u))
    assert np.all(np.isfinite(result.state.v))

    # A tampa móvel deve produzir velocidade horizontal.
    assert np.max(result.state.u[-1, :]) > 0.0

    # A projeção deve conservar massa nas faces.
    div = divergence_faces(
        result.state.u_faces,
        result.state.v_faces,
        grid,
    )
    rms_div = np.sqrt(np.mean(div**2))
    assert rms_div < 1e-10


def test_one_step_preserves_input_state():
    config = SimulationConfig(
        nx=8, ny=8, dt=0.001
    )
    grid = create_grid(config)
    state = initial_state(grid)

    originals = [
        a.copy() for a in (
            state.u,
            state.v,
            state.u_faces,
            state.v_faces,
            state.pressure,
        )
    ]

    advance_one_step(state, grid, config)

    for actual, original in zip(
        (
            state.u,
            state.v,
            state.u_faces,
            state.v_faces,
            state.pressure,
        ),
        originals,
    ):
        np.testing.assert_array_equal(actual, original)


def test_one_step_rejects_unstable_timestep():
    config = SimulationConfig(
        nx=8, ny=8, dt=10.0, reynolds=100.0
    )
    grid = create_grid(config)

    with pytest.raises(ValueError, match="excede limite"):
        advance_one_step(initial_state(grid), grid, config)


def test_one_step_rejects_wrong_state_shape():
    config = SimulationConfig(nx=8, ny=8)
    grid = create_grid(config)
    state = initial_state(grid)

    invalid_state = FlowState(
        u=np.zeros((3, 3)),
        v=state.v,
        u_faces=state.u_faces,
        v_faces=state.v_faces,
        pressure=state.pressure,
    )

    with pytest.raises(ValueError, match="u"):
        advance_one_step(invalid_state, grid, config)


