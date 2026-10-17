"""Testes da reutilização dos coeficientes de Poisson."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.poisson import assemble_poisson_coefficients
from navierstokes.fdm.solver import initialize_state
from navierstokes.fdm.time_step import advance_one_step


@pytest.mark.parametrize("interval", [1, 10, 20])
def test_reused_coefficients_preserve_one_step(interval):
    config = SimulationConfig(
        reynolds=100,
        nx=12,
        ny=12,
        dt=0.001,
    )

    grid = create_grid(config)
    initial_state = initialize_state(grid)
    coefficients = assemble_poisson_coefficients(grid)

    kwargs = {
        "backend": "numba",
        "pressure_tolerance": 1e-9,
        "pressure_max_iterations": 10000,
        "residual_check_interval": interval,
    }

    original = advance_one_step(
        initial_state,
        grid,
        config,
        **kwargs,
    )

    optimized = advance_one_step(
        initial_state,
        grid,
        config,
        coefficients=coefficients,
        **kwargs,
    )

    for field in ("u", "v", "u_faces", "v_faces", "pressure"):
        np.testing.assert_allclose(
            getattr(original.state, field),
            getattr(optimized.state, field),
            rtol=0,
            atol=1e-12,
        )

    assert original.pressure_result.iterations == optimized.pressure_result.iterations

    assert optimized.pressure_result.residual <= 1e-9
