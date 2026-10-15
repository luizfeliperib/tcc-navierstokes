
"""Testes do solver Gauss-Seidel para Poisson (FDM-03).

A = -L_h, com condicoes de Neumann homogeneas.
A suite rapida cobre os criterios essenciais.
Casos adicionais usam o marker 'extended'.
"""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.gauss_seidel import solve_poisson_gauss_seidel
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.poisson import (
    apply_poisson_system,
    assemble_poisson_coefficients,
)

GRID_CASES = [
    pytest.param(5, 5, 1.0, 1.0, id="base-5x5"),
    pytest.param(
        2, 2, 1.0, 1.0,
        marks=pytest.mark.extended,
        id="min-2x2",
    ),
    pytest.param(
        7, 4, 2.0, 1.0,
        marks=pytest.mark.extended,
        id="rect-7x4",
    ),
    pytest.param(
        12, 9, 1.0, 2.0,
        marks=pytest.mark.extended,
        id="rect-12x9",
    ),
]

SEEDS = [
    pytest.param(42, id="base-seed"),
    pytest.param(
        7, marks=pytest.mark.extended, id="seed-7"
    ),
    pytest.param(
        123, marks=pytest.mark.extended, id="seed-123"
    ),
]


def make_test_grid(nx, ny, lx, ly):
    """Cria uma malha cartesiana de teste."""
    return create_grid(
        SimulationConfig(nx=nx, ny=ny, lx=lx, ly=ly)
    )


def make_known_problem(grid, seed=42):
    """Constroi A(phi_ref) = rhs com solucao conhecida."""
    coefficients = assemble_poisson_coefficients(grid)

    phi_ref = np.random.default_rng(seed).normal(
        size=grid.scalar_shape
    )
    phi_ref -= np.mean(phi_ref)

    rhs = apply_poisson_system(phi_ref, coefficients)

    return coefficients, phi_ref, rhs


# ============================================================
# 1. RECUPERACAO DE SOLUCAO DISCRETA CONHECIDA
# ============================================================


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_recovers_known_discrete_solution(
    nx, ny, lx, ly, seed
):
    grid = make_test_grid(nx, ny, lx, ly)
    coefficients, phi_ref, rhs = make_known_problem(
        grid, seed
    )

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=1e-9,
        max_iterations=15_000,
    )

    assert result.converged
    assert result.iterations <= 15_000

    np.testing.assert_allclose(
        result.phi,
        phi_ref,
        rtol=0.0,
        atol=1e-7,
    )


# ============================================================
# 2. RESIDUO VERDADEIRO E HISTORICO
# ============================================================


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_reported_residual_matches_true_residual(
    nx, ny, lx, ly
):
    grid = make_test_grid(nx, ny, lx, ly)
    coefficients, _, rhs = make_known_problem(grid)

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=1e-8,
        max_iterations=15_000,
    )

    residual_field = (
        rhs - apply_poisson_system(
            result.phi, coefficients
        )
    )
    true_residual = (
        np.linalg.norm(residual_field) / np.sqrt(rhs.size)
    )

    assert result.converged
    assert result.residual == pytest.approx(
        true_residual, rel=1e-12, abs=1e-12
    )

    assert len(result.residual_history) == (
        result.iterations + 1
    )
    assert result.residual_history[-1] == pytest.approx(
        result.residual
    )
    assert result.residual <= 1e-8


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_residual_reduces_from_initial_value(
    nx, ny, lx, ly
):
    grid = make_test_grid(nx, ny, lx, ly)
    coefficients, _, rhs = make_known_problem(grid)

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=1e-8,
        max_iterations=15_000,
    )

    assert result.converged
    assert result.residual_history[0] > 0
    assert (
        result.residual_history[-1]
        < result.residual_history[0]
    )


# ============================================================
# 3. REFERENCIA DE MEDIA ZERO
# ============================================================


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_solution_has_zero_mean(nx, ny, lx, ly):
    grid = make_test_grid(nx, ny, lx, ly)
    coefficients, _, rhs = make_known_problem(grid)

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=1e-8,
        max_iterations=15_000,
    )

    assert result.converged
    assert np.mean(result.phi) == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_initial_constant_shift_does_not_change_solution(
    nx, ny, lx, ly
):
    grid = make_test_grid(nx, ny, lx, ly)
    coefficients, _, rhs = make_known_problem(grid)

    initial = np.random.default_rng(11).normal(
        size=grid.scalar_shape
    )

    first = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        initial_phi=initial,
        tolerance=1e-9,
        max_iterations=15_000,
    )

    second = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        initial_phi=initial + 100.0,
        tolerance=1e-9,
        max_iterations=15_000,
    )

    assert first.converged
    assert second.converged

    np.testing.assert_allclose(
        first.phi,
        second.phi,
        rtol=0.0,
        atol=1e-8,
    )


# ============================================================
# 4. COMPATIBILIDADE DO TERMO-FONTE
# ============================================================


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_rejects_incompatible_rhs(nx, ny, lx, ly):
    grid = make_test_grid(nx, ny, lx, ly)
    coefficients = assemble_poisson_coefficients(grid)

    rhs = np.ones(grid.scalar_shape)

    with pytest.raises(
        ValueError, match="incompat"
    ):
        solve_poisson_gauss_seidel(rhs, coefficients)


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_accepts_compatible_rhs(nx, ny, lx, ly):
    grid = make_test_grid(nx, ny, lx, ly)
    coefficients, _, rhs = make_known_problem(grid)

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=1e-8,
        max_iterations=15_000,
    )

    assert result.converged


# ============================================================
# 5. SOLUCAO TRIVIAL E CONVERGENCIA IMEDIATA
# ============================================================


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_zero_rhs_converges_without_iterations(
    nx, ny, lx, ly
):
    grid = make_test_grid(nx, ny, lx, ly)
    coefficients = assemble_poisson_coefficients(grid)

    rhs = np.zeros(grid.scalar_shape)

    result = solve_poisson_gauss_seidel(
        rhs, coefficients
    )

    assert result.converged
    assert result.iterations == 0
    assert result.residual == pytest.approx(0.0)
    assert len(result.residual_history) == 1

    np.testing.assert_array_equal(
        result.phi, np.zeros(grid.scalar_shape)
    )


# ============================================================
# 6. LIMITE DE ITERACOES
# ============================================================


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_reports_nonconvergence_at_iteration_limit(
    nx, ny, lx, ly
):
    grid = make_test_grid(nx, ny, lx, ly)
    coefficients, _, rhs = make_known_problem(grid)

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=1e-14,
        max_iterations=1,
    )

    assert not result.converged
    assert result.iterations == 1
    assert len(result.residual_history) == 2
    assert result.residual > 1e-14


# ============================================================
# 7. VALIDACAO DE ENTRADAS
# ============================================================


def test_rejects_invalid_rhs_shape():
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients = assemble_poisson_coefficients(grid)

    with pytest.raises(ValueError, match="Shape"):
        solve_poisson_gauss_seidel(
            np.zeros((4, 5)), coefficients
        )


def test_rejects_nonfinite_rhs():
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients = assemble_poisson_coefficients(grid)

    rhs = np.zeros(grid.scalar_shape)
    rhs[0, 0] = np.nan

    with pytest.raises(ValueError, match="finit"):
        solve_poisson_gauss_seidel(rhs, coefficients)


@pytest.mark.parametrize(
    "tolerance",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(
            -1e-5, marks=pytest.mark.extended,
            id="negative",
        ),
        pytest.param(
            np.nan, marks=pytest.mark.extended,
            id="nan",
        ),
    ],
)
def test_rejects_invalid_tolerance(tolerance):
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients = assemble_poisson_coefficients(grid)

    with pytest.raises(ValueError):
        solve_poisson_gauss_seidel(
            np.zeros(grid.scalar_shape),
            coefficients,
            tolerance=tolerance,
        )


@pytest.mark.parametrize(
    "max_iterations",
    [
        pytest.param(0, id="zero"),
        pytest.param(
            -1, marks=pytest.mark.extended,
            id="negative",
        ),
        pytest.param(
            1.5, marks=pytest.mark.extended,
            id="fractional",
        ),
    ],
)
def test_rejects_invalid_max_iterations(
    max_iterations
):
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients = assemble_poisson_coefficients(grid)

    with pytest.raises(ValueError):
        solve_poisson_gauss_seidel(
            np.zeros(grid.scalar_shape),
            coefficients,
            max_iterations=max_iterations,
        )


def test_rejects_invalid_initial_phi_shape():
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients = assemble_poisson_coefficients(grid)

    with pytest.raises(ValueError, match="initial_phi"):
        solve_poisson_gauss_seidel(
            np.zeros(grid.scalar_shape),
            coefficients,
            initial_phi=np.zeros((4, 5)),
        )


def test_rejects_nonfinite_initial_phi():
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients = assemble_poisson_coefficients(grid)

    initial = np.zeros(grid.scalar_shape)
    initial[0, 0] = np.inf

    with pytest.raises(ValueError, match="initial_phi"):
        solve_poisson_gauss_seidel(
            np.zeros(grid.scalar_shape),
            coefficients,
            initial_phi=initial,
        )


@pytest.mark.parametrize(
    "compatibility_tolerance",
    [
        pytest.param(-1.0, id="negative"),
        pytest.param(
            np.nan, marks=pytest.mark.extended,
            id="nan",
        ),
    ],
)
def test_rejects_invalid_compatibility_tolerance(
    compatibility_tolerance
):
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients = assemble_poisson_coefficients(grid)

    with pytest.raises(ValueError):
        solve_poisson_gauss_seidel(
            np.zeros(grid.scalar_shape),
            coefficients,
            compatibility_tolerance=compatibility_tolerance,
        )


# ============================================================
# 8. ENTRADAS NAO DEVEM SER MODIFICADAS
# ============================================================


def test_solver_does_not_modify_input_arrays():
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients, _, rhs = make_known_problem(grid)

    initial = np.random.default_rng(22).normal(
        size=grid.scalar_shape
    )

    rhs_before = rhs.copy()
    initial_before = initial.copy()

    solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        initial_phi=initial,
        tolerance=1e-8,
        max_iterations=15_000,
    )

    np.testing.assert_array_equal(rhs, rhs_before)
    np.testing.assert_array_equal(
        initial, initial_before
    )


# ============================================================
# 9. VALIDACOES ADICIONAIS DO FDM-03
# ============================================================


@pytest.mark.parametrize(
    "invalid_iterations",
    [
        pytest.param(True, id="bool-true"),
        pytest.param(
            False,
            marks=pytest.mark.extended,
            id="bool-false",
        ),
        pytest.param(
            np.bool_(True),
            marks=pytest.mark.extended,
            id="numpy-bool",
        ),
    ],
)
def test_rejects_boolean_max_iterations(invalid_iterations):
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients = assemble_poisson_coefficients(grid)

    with pytest.raises(ValueError, match="max_iterations"):
        solve_poisson_gauss_seidel(
            np.zeros(grid.scalar_shape),
            coefficients,
            max_iterations=invalid_iterations,
        )


def test_rejects_nonfinite_initial_residual():
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients = assemble_poisson_coefficients(grid)

    # Valores finitos cuja norma euclidiana transborda.
    rhs = np.zeros(grid.scalar_shape)
    rhs[0, 0] = 1e200
    rhs[0, 1] = -1e200

    with pytest.raises(
        FloatingPointError,
        match="Residuo inicial nao finito",
    ):
        solve_poisson_gauss_seidel(
            rhs,
            coefficients,
        )


def test_approximately_compatible_rhs_has_residual_floor():
    """Uma media residual nao nula limita a convergencia."""
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients, _, compatible_rhs = make_known_problem(grid)

    offset = 1e-7
    rhs = compatible_rhs + offset

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=1e-9,
        max_iterations=500,
        compatibility_tolerance=1e-5,
    )

    # Como A(phi) possui media zero, o residuo nao pode
    # ter RMS inferior ao modulo da media de rhs.
    assert not result.converged
    assert result.iterations == 500
    assert result.residual >= abs(np.mean(rhs)) - 1e-12
    assert result.residual > 1e-9


@pytest.mark.extended
def test_approximately_compatible_rhs_converges_if_tolerance_allows():
    """Convergencia e possivel acima do piso de compatibilidade."""
    grid = make_test_grid(5, 5, 1.0, 1.0)
    coefficients, _, compatible_rhs = make_known_problem(grid)

    rhs = compatible_rhs + 1e-7

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=1e-6,
        max_iterations=5_000,
        compatibility_tolerance=1e-5,
    )

    assert result.converged
    assert result.residual <= 1e-6
    assert abs(np.mean(result.phi)) < 1e-12

