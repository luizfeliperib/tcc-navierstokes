import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.boundary import interpolate_impermeable_faces
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.operators import total_divergence


@pytest.fixture
def grid():
    return create_grid(
        SimulationConfig(
            nx=8, ny=6, lx=2.0, ly=1.0
        )
    )


def test_face_shapes(grid):
    u = grid.zeros_scalar()
    v = grid.zeros_scalar()

    uf, vf = interpolate_impermeable_faces(
        u, v, grid
    )

    assert uf.shape == grid.u_face_shape
    assert vf.shape == grid.v_face_shape


def test_impermeable_walls(grid):
    rng = np.random.default_rng(42)

    u = rng.normal(size=grid.scalar_shape)
    v = rng.normal(size=grid.scalar_shape)

    uf, vf = interpolate_impermeable_faces(
        u, v, grid
    )

    np.testing.assert_allclose(uf[:, 0], 0.0)
    np.testing.assert_allclose(uf[:, -1], 0.0)
    np.testing.assert_allclose(vf[0, :], 0.0)
    np.testing.assert_allclose(vf[-1, :], 0.0)


def test_internal_face_interpolation(grid):
    u = np.full(grid.scalar_shape, 4.0)
    v = np.full(grid.scalar_shape, -2.0)

    uf, vf = interpolate_impermeable_faces(
        u, v, grid
    )

    np.testing.assert_allclose(
        uf[:, 1:-1], 4.0
    )
    np.testing.assert_allclose(
        vf[1:-1, :], -2.0
    )


def test_global_conservation(grid):
    rng = np.random.default_rng(11)

    u = rng.normal(size=grid.scalar_shape)
    v = rng.normal(size=grid.scalar_shape)

    uf, vf = interpolate_impermeable_faces(
        u, v, grid
    )

    integral = total_divergence(uf, vf, grid)

    assert integral == pytest.approx(0.0, abs=1e-12)


def test_invalid_velocity_shape(grid):
    u = np.zeros((3, 3))
    v = grid.zeros_scalar()

    with pytest.raises(ValueError):
        interpolate_impermeable_faces(u, v, grid)
