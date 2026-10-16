"""Testes de verificacao do solver de Poisson (FDM-04)."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.poisson import (
    apply_poisson_system,
    assemble_poisson_coefficients,
)
from navierstokes.fdm.verification import (
    calculate_errors,
    manufactured_solution,
    manufactured_source,
    observed_order,
    verify_poisson_solver,
)


@pytest.mark.parametrize(
    "nx,ny,lx,ly",
    [
        pytest.param(8, 8, 1.0, 1.0, id="square"),
        pytest.param(
            12, 8, 2.0, 1.0,
            marks=pytest.mark.extended,
            id="rectangle",
        ),
        pytest.param(
            16, 12, 1.0, 2.0,
            marks=pytest.mark.extended,
            id="anisotropic",
        ),
    ],
)
def test_manufactured_solution_has_zero_mean(
    nx, ny, lx, ly
):
    grid = create_grid(
        SimulationConfig(nx=nx, ny=ny, lx=lx, ly=ly)
    )

    exact = manufactured_solution(grid)
    rhs = manufactured_source(grid)

    assert exact.shape == grid.scalar_shape
    assert rhs.shape == grid.scalar_shape
    assert abs(np.mean(exact)) < 1e-12
    assert abs(np.mean(rhs)) < 1e-12


@pytest.mark.parametrize(
    "nx,ny,lx,ly",
    [
        pytest.param(8, 8, 1.0, 1.0, id="square"),
        pytest.param(
            12, 8, 2.0, 1.0,
            marks=pytest.mark.extended,
            id="rectangle",
        ),
    ],
)
def test_manufactured_solution_is_discrete_eigenmode(
    nx, ny, lx, ly
):
    """Verifica a identidade discreta independentemente do solver."""
    grid = create_grid(
        SimulationConfig(nx=nx, ny=ny, lx=lx, ly=ly)
    )

    coefficients = assemble_poisson_coefficients(grid)
    exact = manufactured_solution(grid)

    discrete_eigenvalue = (
        4.0 / grid.dx**2
        * np.sin(np.pi / (2 * nx)) ** 2
        + 4.0 / grid.dy**2
        * np.sin(np.pi / (2 * ny)) ** 2
    )

    np.testing.assert_allclose(
        apply_poisson_system(exact, coefficients),
        discrete_eigenvalue * exact,
        rtol=1e-12,
        atol=1e-12,
    )


def test_error_norms():
    exact = np.zeros((2, 2))
    numerical = np.array(
        [[1.0, -1.0], [0.0, 0.0]]
    )

    l2, linf = calculate_errors(numerical, exact)

    assert l2 == pytest.approx(np.sqrt(0.5))
    assert linf == pytest.approx(1.0)


def test_error_norms_reject_invalid_shape():
    with pytest.raises(ValueError):
        calculate_errors(
            np.zeros((2, 2)),
            np.zeros((3, 2)),
        )


def test_error_norms_reject_nonfinite_values():
    numerical = np.zeros((2, 2))
    numerical[0, 0] = np.nan

    with pytest.raises(ValueError):
        calculate_errors(
            numerical,
            np.zeros((2, 2)),
        )


def test_observed_order():
    assert observed_order(
        coarse_error=0.04,
        fine_error=0.01,
    ) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "coarse,fine,ratio",
    [
        (0.0, 0.01, 2.0),
        (0.04, 0.0, 2.0),
        (0.04, 0.01, 1.0),
        (np.nan, 0.01, 2.0),
    ],
)
def test_observed_order_rejects_invalid_values(
    coarse, fine, ratio
):
    with pytest.raises(ValueError):
        observed_order(coarse, fine, ratio)


def test_solver_matches_analytical_solution():
    result = verify_poisson_solver(
        nx=8,
        ny=8,
        tolerance=1e-10,
    )

    assert result.converged
    assert result.residual <= 1e-10
    assert result.algebraic_error <= 1e-10
    assert result.l2_error < 0.01
    assert result.linf_error < 0.02
    assert abs(result.rhs_mean) < 1e-12


def test_spatial_refinement_second_order():
    coarse = verify_poisson_solver(
        nx=8,
        ny=8,
        tolerance=1e-10,
    )

    fine = verify_poisson_solver(
        nx=16,
        ny=16,
        tolerance=1e-10,
    )

    assert fine.l2_error < coarse.l2_error
    assert fine.linf_error < coarse.linf_error

    order_l2 = observed_order(
        coarse.l2_error, fine.l2_error
    )

    order_linf = observed_order(
        coarse.linf_error, fine.linf_error
    )

    assert 1.8 < order_l2 < 2.2
    assert 1.8 < order_linf < 2.2


@pytest.mark.extended
def test_rectangular_domain_refinement():
    coarse = verify_poisson_solver(
        nx=8,
        ny=6,
        lx=2.0,
        ly=1.0,
        tolerance=1e-10,
    )

    fine = verify_poisson_solver(
        nx=16,
        ny=12,
        lx=2.0,
        ly=1.0,
        tolerance=1e-10,
    )

    assert coarse.dx != coarse.dy
    assert fine.dx != fine.dy

    assert fine.l2_error < coarse.l2_error
    assert fine.linf_error < coarse.linf_error

    order_l2 = observed_order(
        coarse.l2_error, fine.l2_error
    )

    assert 1.7 < order_l2 < 2.3


@pytest.mark.extended
def test_solver_fails_explicitly_when_not_converged():
    with pytest.raises(
        RuntimeError, match="nao convergiu"
    ):
        verify_poisson_solver(
            nx=8,
            ny=8,
            tolerance=1e-14,
            max_iterations=1,
        )
