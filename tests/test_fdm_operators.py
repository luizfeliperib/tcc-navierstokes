import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.operators import (
    divergence_faces,
    gradient_faces,
    laplacian_neumann,
    total_divergence,
)


@pytest.fixture
def grid():
    return create_grid(SimulationConfig(nx=8, ny=6, lx=2.0, ly=1.0))


def test_constant_field_has_zero_gradient(grid):
    phi = np.full(grid.scalar_shape, 7.0)

    gx, gy = gradient_faces(phi, grid)

    np.testing.assert_allclose(gx, 0.0)
    np.testing.assert_allclose(gy, 0.0)


def test_constant_field_has_zero_laplacian(grid):
    phi = np.full(grid.scalar_shape, 7.0)

    result = laplacian_neumann(phi, grid)

    np.testing.assert_allclose(result, 0.0)


def test_linear_field_gradient_on_internal_faces(grid):
    x, y = grid.cell_centers()
    phi = 3.0 * x - 2.0 * y + 5.0

    gx, gy = gradient_faces(phi, grid)

    np.testing.assert_allclose(gx[:, 1:-1], 3.0, atol=1e-12)
    np.testing.assert_allclose(gy[1:-1, :], -2.0, atol=1e-12)

    # Neumann homogênea nas faces externas.
    np.testing.assert_allclose(gx[:, 0], 0.0)
    np.testing.assert_allclose(gx[:, -1], 0.0)
    np.testing.assert_allclose(gy[0, :], 0.0)
    np.testing.assert_allclose(gy[-1, :], 0.0)


def test_laplacian_operator_identity(grid):
    rng = np.random.default_rng(42)
    phi = rng.normal(size=grid.scalar_shape)

    gx, gy = gradient_faces(phi, grid)

    expected = divergence_faces(gx, gy, grid)
    actual = laplacian_neumann(phi, grid)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_laplacian_neumann_compatibility(grid):
    rng = np.random.default_rng(123)
    phi = rng.normal(size=grid.scalar_shape)

    result = laplacian_neumann(phi, grid)

    np.testing.assert_allclose(np.sum(result), 0.0, atol=1e-11)


def test_corner_stencil():
    grid = create_grid(SimulationConfig(nx=2, ny=2, lx=2.0, ly=2.0))

    phi = np.array(
        [
            [3.0, 5.0],
            [7.0, 9.0],
        ]
    )

    result = laplacian_neumann(phi, grid)

    # Canto inferior esquerdo:
    # (5 - 3) / dx² + (7 - 3) / dy²
    # dx = dy = 1
    assert result[0, 0] == pytest.approx(6.0)


def test_closed_domain_global_conservation(grid):
    rng = np.random.default_rng(7)

    uf = rng.normal(size=grid.u_face_shape)
    vf = rng.normal(size=grid.v_face_shape)

    uf[:, 0] = 0.0
    uf[:, -1] = 0.0
    vf[0, :] = 0.0
    vf[-1, :] = 0.0

    integral = total_divergence(uf, vf, grid)

    assert integral == pytest.approx(0.0, abs=1e-12)


def test_invalid_field_shapes(grid):
    wrong_phi = np.zeros((3, 3))

    with pytest.raises(ValueError):
        gradient_faces(wrong_phi, grid)

    with pytest.raises(ValueError):
        laplacian_neumann(wrong_phi, grid)


def test_laplacian_interior_five_point_stencil():
    """Verifica o Laplaciano em uma célula interior."""

    grid = create_grid(
        SimulationConfig(
            nx=3,
            ny=3,
            lx=3.0,
            ly=3.0,
        )
    )

    phi = np.array(
        [
            [0.0, 2.0, 0.0],
            [4.0, 10.0, 6.0],
            [0.0, 8.0, 0.0],
        ]
    )

    result = laplacian_neumann(phi, grid)

    assert result[1, 1] == pytest.approx(-20.0)
