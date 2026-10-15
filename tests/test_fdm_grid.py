import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid, initialize_fields


def test_default_grid_geometry():
    config = SimulationConfig()
    grid = create_grid(config)

    assert grid.nx == config.nx
    assert grid.ny == config.ny

    assert grid.dx == pytest.approx(config.lx / config.nx)
    assert grid.dy == pytest.approx(config.ly / config.ny)

    assert grid.x[0] == pytest.approx(grid.dx / 2)
    assert grid.y[0] == pytest.approx(grid.dy / 2)

    assert grid.x[-1] == pytest.approx(
        config.lx - grid.dx / 2
    )
    assert grid.y[-1] == pytest.approx(
        config.ly - grid.dy / 2
    )

    assert grid.cell_area == pytest.approx(
        grid.dx * grid.dy
    )
    
def test_grid_requires_at_least_two_points_per_direction():
    with pytest.raises(ValueError):
        create_grid(SimulationConfig(nx=1))

def test_cell_centered_geometry():
    config = SimulationConfig(
        nx=4, ny=2, lx=2.0, ly=1.0
    )

    grid = create_grid(config)

    assert grid.dx == pytest.approx(0.5)
    assert grid.dy == pytest.approx(0.5)

    np.testing.assert_allclose(
        grid.x, [0.25, 0.75, 1.25, 1.75]
    )

    np.testing.assert_allclose(
        grid.y, [0.25, 0.75]
    )


def test_grid_shapes():
    config = SimulationConfig(nx=50, ny=30)
    grid = create_grid(config)

    assert grid.scalar_shape == (30, 50)
    assert grid.u_face_shape == (30, 51)
    assert grid.v_face_shape == (31, 50)

    x, y = grid.cell_centers()

    assert x.shape == (30, 50)
    assert y.shape == (30, 50)


@pytest.mark.parametrize(
    "nx, ny",
    [
        (8, 6),
        (31, 21),
        (50, 30),
    ],
)
def test_initialize_fields(nx, ny):
    config = SimulationConfig(nx=nx, ny=ny)

    u, v, p = initialize_fields(config)

    expected_shape = (ny, nx)

    for field in (u, v, p):
        assert field.shape == expected_shape
        assert field.dtype == np.float64
        np.testing.assert_array_equal(
            field,
            np.zeros(expected_shape),
        )

@pytest.mark.parametrize(
    "nx, ny, lx, ly, expected_exception",
    [
        (1, 10, 1.0, 1.0, ValueError),
        (10, 1, 1.0, 1.0, ValueError),
        (10, 10, 0.0, 1.0, ValueError),
        (10, 10, 1.0, -1.0, ValueError),
        (10, 10, float("inf"), 1.0, ValueError),
        (10.5, 10, 1.0, 1.0, TypeError),
        (True, 10, 1.0, 1.0, TypeError),
        (10, 5.5, 1.0, 1.0, TypeError),
    ],
)
def test_invalid_grid(
    nx, ny, lx, ly, expected_exception
):
    config = SimulationConfig(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
    )

    with pytest.raises(expected_exception):
        create_grid(config)
