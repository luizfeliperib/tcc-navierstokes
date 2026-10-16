"""Testes do sistema discreto de Poisson — FDM-02.

Convencao: A = -L_h = -D_h G_h.
Contorno: Neumann homogenea nas quatro paredes.

Casos essenciais executam na suite rapida.
Casos adicionais usam o marker 'extended'.
"""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.operators import (
    gradient_faces,
    laplacian_neumann,
)
from navierstokes.fdm.poisson import (
    apply_poisson_system,
    assemble_poisson_coefficients,
)

GRID_CASES = [
    pytest.param(8, 6, 2.0, 1.0, id="base-8x6"),
    pytest.param(
        2,
        2,
        1.0,
        1.0,
        marks=pytest.mark.extended,
        id="min-2x2",
    ),
    pytest.param(
        3,
        5,
        1.0,
        2.0,
        marks=pytest.mark.extended,
        id="rect-3x5",
    ),
    pytest.param(
        16,
        12,
        1.0,
        1.0,
        marks=pytest.mark.extended,
        id="large-16x12",
    ),
]

SEEDS = [
    pytest.param(42, id="base-seed"),
    pytest.param(7, marks=pytest.mark.extended, id="seed-7"),
    pytest.param(123, marks=pytest.mark.extended, id="seed-123"),
]

CONSTANTS = [
    pytest.param(7.0, id="base-constant"),
    pytest.param(0.0, marks=pytest.mark.extended, id="zero"),
    pytest.param(1.0, marks=pytest.mark.extended, id="one"),
    pytest.param(-3.0, marks=pytest.mark.extended, id="negative"),
]


def make_test_grid(nx, ny, lx, ly):
    """Cria uma malha cartesiana para os testes."""
    return create_grid(SimulationConfig(nx=nx, ny=ny, lx=lx, ly=ly))


# ============================================================
# 1. EQUIVALENCIA COM O LAPLACIANO DO FDM-01
# ============================================================


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_assembled_system_matches_existing_laplacian(nx, ny, lx, ly, seed):
    """Verifica A(phi) = -L_h(phi)."""
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    phi = np.random.default_rng(seed).normal(size=grid.scalar_shape)

    np.testing.assert_allclose(
        apply_poisson_system(phi, c),
        -laplacian_neumann(phi, grid),
        rtol=1e-12,
        atol=1e-12,
    )


# ============================================================
# 2. COEFICIENTES: INTERIOR, BORDAS E CANTOS
# ============================================================


def test_diagonal_coefficients_at_interior_edges_and_corners():
    """Confere diagonais para dx = dy = 1."""
    grid = make_test_grid(5, 5, 5.0, 5.0)
    c = assemble_poisson_coefficients(grid)

    assert c.center[2, 2] == pytest.approx(4.0)
    assert c.center[0, 2] == pytest.approx(3.0)
    assert c.center[0, 0] == pytest.approx(2.0)


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_boundary_directional_coefficients_are_zero(nx, ny, lx, ly):
    """Nao ha conexoes para fora do dominio."""
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    np.testing.assert_array_equal(c.west[:, 0], 0.0)
    np.testing.assert_array_equal(c.east[:, -1], 0.0)
    np.testing.assert_array_equal(c.south[0, :], 0.0)
    np.testing.assert_array_equal(c.north[-1, :], 0.0)


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_interior_directional_coefficients(nx, ny, lx, ly):
    """Confere coeficientes das conexoes internas."""
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    np.testing.assert_allclose(c.east[:, :-1], 1.0 / grid.dx**2)
    np.testing.assert_allclose(c.west[:, 1:], 1.0 / grid.dx**2)
    np.testing.assert_allclose(c.north[:-1, :], 1.0 / grid.dy**2)
    np.testing.assert_allclose(c.south[1:, :], 1.0 / grid.dy**2)


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_diagonal_equals_sum_of_neighbor_coefficients(nx, ny, lx, ly):
    """Diagonal igual a soma dos coeficientes vizinhos."""
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    expected = c.east + c.west + c.north + c.south

    np.testing.assert_allclose(
        c.center,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


# ============================================================
# 3. SIMETRIA
# ============================================================


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_poisson_coefficients_are_symmetric(nx, ny, lx, ly):
    """Conexoes entre celulas vizinhas sao reciprocas."""
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    np.testing.assert_array_equal(
        c.east[:, :-1],
        c.west[:, 1:],
    )
    np.testing.assert_array_equal(
        c.north[:-1, :],
        c.south[1:, :],
    )


# ============================================================
# 4. SOMA DAS LINHAS E CONSERVACAO GLOBAL
# ============================================================


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_poisson_row_sums_are_zero(nx, ny, lx, ly):
    """Verifica A(1) = 0 pela soma dos coeficientes."""
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    row_sum = c.center - c.east - c.west - c.north - c.south

    np.testing.assert_allclose(
        row_sum,
        np.zeros(grid.scalar_shape),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_poisson_global_conservation(nx, ny, lx, ly, seed):
    """A integral discreta de A(phi) deve ser zero."""
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    phi = np.random.default_rng(seed).normal(size=grid.scalar_shape)
    a_phi = apply_poisson_system(phi, c)

    integral = np.sum(a_phi) * grid.cell_area

    np.testing.assert_allclose(
        integral,
        0.0,
        rtol=0.0,
        atol=1e-11,
    )


# ============================================================
# 5. SINGULARIDADE E INVARIANCIA POR CONSTANTE
# ============================================================


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
@pytest.mark.parametrize("constant", CONSTANTS)
def test_constant_field_is_in_nullspace(nx, ny, lx, ly, constant):
    """Campos constantes pertencem ao nucleo de A."""
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    phi = np.full(grid.scalar_shape, constant)

    np.testing.assert_allclose(
        apply_poisson_system(phi, c),
        0.0,
        rtol=0.0,
        atol=1e-11,
    )


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_poisson_constant_shift_invariance(nx, ny, lx, ly):
    """Adicionar constante nao modifica A(phi)."""
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    phi = np.random.default_rng(99).normal(size=grid.scalar_shape)

    original = apply_poisson_system(phi, c)
    shifted = apply_poisson_system(phi + 3.0, c)

    np.testing.assert_allclose(
        shifted,
        original,
        rtol=1e-12,
        atol=1e-11,
    )


# ============================================================
# 6. IDENTIDADE DE ENERGIA DISCRETA
# ============================================================


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_poisson_discrete_energy_identity(nx, ny, lx, ly, seed):
    """Verifica <phi,A(phi)>_h = ||G_h(phi)||_h^2."""
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    phi = np.random.default_rng(seed).normal(size=grid.scalar_shape)

    a_phi = apply_poisson_system(phi, c)

    energy_operator = np.sum(phi * a_phi) * grid.cell_area

    gx, gy = gradient_faces(phi, grid)

    energy_gradient = (np.sum(gx**2) + np.sum(gy**2)) * grid.cell_area

    np.testing.assert_allclose(
        energy_operator,
        energy_gradient,
        rtol=1e-12,
        atol=1e-11,
    )

    assert energy_operator >= -1e-11


# ============================================================
# 7. VALIDACAO DE ENTRADA
# ============================================================


def test_poisson_system_rejects_invalid_phi_shape():
    """Rejeita campos com shape incorreto."""
    grid = make_test_grid(8, 6, 1.0, 1.0)
    c = assemble_poisson_coefficients(grid)

    invalid_phi = np.zeros((8, 6))

    with pytest.raises(ValueError):
        apply_poisson_system(invalid_phi, c)
