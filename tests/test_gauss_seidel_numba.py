
"""Equivalencia numerica entre Gauss-Seidel Python e Numba."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.gauss_seidel import solve_poisson_gauss_seidel
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.poisson import assemble_poisson_coefficients


@pytest.mark.parametrize("size", [8, 16])
def test_python_and_numba_agree(size):
    pytest.importorskip("numba")

    grid = create_grid(SimulationConfig(nx=size, ny=size))
    coefficients = assemble_poisson_coefficients(grid)

    x, y = grid.cell_centers()

    rhs = np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    rhs -= np.mean(rhs)

    kwargs = {
        "tolerance": 1e-9,
        "max_iterations": 10_000,
    }

    python_result = solve_poisson_gauss_seidel(
        rhs, coefficients, backend="python", **kwargs
    )
    numba_result = solve_poisson_gauss_seidel(
        rhs, coefficients, backend="numba", **kwargs
    )

    assert python_result.converged
    assert numba_result.converged
    assert python_result.iterations == numba_result.iterations

    np.testing.assert_allclose(
        python_result.phi,
        numba_result.phi,
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        python_result.residual_history,
        numba_result.residual_history,
        rtol=1e-8,
        atol=1e-12,
    )
