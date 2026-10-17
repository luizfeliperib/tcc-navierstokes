
"""FDM-07.2.4: testes da frequência de avaliação do resíduo."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.gauss_seidel import solve_poisson_gauss_seidel
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.poisson import (
    apply_poisson_system,
    assemble_poisson_coefficients,
)


def make_problem(n=12):
    grid = create_grid(SimulationConfig(nx=n, ny=n))
    coeff = assemble_poisson_coefficients(grid)
    x, y = grid.cell_centers()
    phi_ref = np.cos(np.pi * x) * np.cos(np.pi * y)
    phi_ref -= np.mean(phi_ref)
    rhs = apply_poisson_system(phi_ref, coeff)
    return coeff, rhs


@pytest.mark.parametrize("interval", [1, 10, 20])
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_interval_preserves_accuracy(interval, backend):
    coeff, rhs = make_problem()

    result = solve_poisson_gauss_seidel(
        rhs,
        coeff,
        tolerance=1e-9,
        max_iterations=10000,
        backend=backend,
        residual_check_interval=interval,
    )

    true_residual = np.linalg.norm(
        rhs - apply_poisson_system(result.phi, coeff)
    ) / np.sqrt(rhs.size)

    assert result.converged
    assert true_residual <= 1e-9
    assert result.residual == pytest.approx(true_residual, abs=1e-12)
    assert abs(np.mean(result.phi)) < 1e-12

    # Histórico: resíduo inicial + verificações periódicas.
    assert len(result.residual_history) == 1 + result.iterations // interval
    assert result.iterations % interval == 0


@pytest.mark.parametrize("interval", [1, 10, 20])
def test_final_iteration_always_checks_residual(interval):
    coeff, rhs = make_problem()

    result = solve_poisson_gauss_seidel(
        rhs,
        coeff,
        tolerance=1e-14,
        max_iterations=7,
        backend="numba",
        residual_check_interval=interval,
    )

    assert result.iterations == 7
    assert len(result.residual_history) == 1 + 7 // interval + (
        7 % interval != 0
    )

    true_residual = np.linalg.norm(
        rhs - apply_poisson_system(result.phi, coeff)
    ) / np.sqrt(rhs.size)

    assert result.residual == pytest.approx(true_residual, abs=1e-12)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, np.nan])
def test_rejects_invalid_interval(invalid):
    coeff, rhs = make_problem()

    with pytest.raises(ValueError, match="residual_check_interval"):
        solve_poisson_gauss_seidel(
            rhs,
            coeff,
            residual_check_interval=invalid,
        )
