
"""Testa a equivalencia de passos temporais com Python e Numba."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.time_step import FlowState, advance_one_step


@pytest.mark.parametrize("size", [8, 16])
def test_integrated_python_numba_equivalence(size):
    pytest.importorskip("numba")

    config = SimulationConfig(
        nx=size,
        ny=size,
        reynolds=100.0,
        dt=0.001,
    )
    grid = create_grid(config)

    initial_state = FlowState(
        u=grid.zeros_scalar(),
        v=grid.zeros_scalar(),
        u_faces=grid.zeros_u_faces(),
        v_faces=grid.zeros_v_faces(),
        pressure=grid.zeros_scalar(),
    )

    states = {}

    for backend in ("python", "numba"):
        state = initial_state

        for _ in range(10):
            result = advance_one_step(
                state,
                grid,
                config,
                pressure_tolerance=1e-9,
                pressure_max_iterations=10_000,
                backend=backend,
            )
            state = result.state

        states[backend] = state

    for name in ("u", "v", "u_faces", "v_faces", "pressure"):
        np.testing.assert_allclose(
            getattr(states["python"], name),
            getattr(states["numba"], name),
            rtol=1e-9,
            atol=1e-11,
            err_msg=f"Diferenca no campo {name}",
        )
