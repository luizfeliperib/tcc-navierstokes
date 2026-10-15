"""Testes da montagem do sistema discreto de Poisson (FDM-02).

Convenção: A = -L_h = -D_h G_h.
Contorno: Neumann homogênea nas quatro paredes.

Os casos básicos são executados por padrão. As combinações adicionais
recebem o marker 'extended' e são executadas na suíte completa.
"""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.operators import gradient_faces, laplacian_neumann
from navierstokes.fdm.poisson import (
    apply_poisson_system,
    assemble_poisson_coefficients,
)

# A primeira geometria é a referência da suíte rápida.
# As demais são preservadas na suíte completa.
GRID_CASES = [
    pytest.param(8, 6, 2.0, 1.0, id="base-8x6"),
    pytest.param(2, 2, 1.0, 1.0, marks=pytest.mark.extended, id="min-2x2"),
    pytest.param(3, 5, 1.0, 2.0, marks=pytest.mark.extended, id="rect-3x5"),
    pytest.param(16, 12, 1.0, 1.0, marks=pytest.mark.extended, id="large-16x12"),
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
    """Cria malha de teste com dimensões configuráveis."""
    return create_grid(SimulationConfig(nx=nx, ny=ny, lx=lx, ly=ly))


# 1. Equivalência com o Laplaciano do FDM-01.
@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_assembled_system_matches_existing_laplacian(nx, ny, lx, ly, seed):
    grid = make_test_grid(nx, ny, lx, ly)
    coefficients = assemble_poisson_coefficients(grid)
    phi = np.random.default_rng(seed).normal(size=grid.scalar_shape)

    np.testing.assert_allclose(
        apply_poisson_system(phi, coefficients),
        -laplacian_neumann(phi, grid),
        rtol=1e-12,
        atol=1e-12,
    )


# 2. Coeficientes de interior, bordas e cantos.
def test_diagonal_coefficients_at_interior_edges_and_corners():
    grid = make_test_grid(nx=5, ny=5, lx=5.0, ly=5.0)
    c = assemble_poisson_coefficients(grid)

    assert c.center[2, 2] == pytest.approx(4.0)
    assert c.center[0, 2] == pytest.approx(3.0)
    assert c.center[0, 0] == pytest.approx(2.0)


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_boundary_directional_coefficients_are_zero(nx, ny, lx, ly):
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    np.testing.assert_array_equal(c.west[:, 0], 0.0)
    np.testing.assert_array_equal(c.east[:, -1], 0.0)
    np.testing.assert_array_equal(c.south[0, :], 0.0)
    np.testing.assert_array_equal(c.north[-1, :], 0.0)


@pytest.mark.parametrize("nx,ny,lx,ly", GRID_CASES)
def test_interior_directional_coefficients(nx, ny, lx, ly):
    grid = make_test_grid(nx, ny, lx, ly)
    c = assemble_poisson_coefficients(grid)

    np.testing.assert_allclose(c.east[:, :-1], 1.0 / grid.dx**2)
    np.testing.assert_allclose(c.west[:, 1:], 1.0 / grid.dx**2)
    np.testing.assert_allclose(c.north[:-1, :], 1.0 / grid.dy**2)
    np.testing.assert_allclose(c.south[1:, :], 1.0 / grid.dy**2)

