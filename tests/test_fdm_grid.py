import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid, initialize_fields


def test_default_grid_geometry():
    config = SimulationConfig()
    grid = create_grid(config)

    assert grid.dx == pytest.approx(0.02)
    assert grid.dy == pytest.approx(0.02)
    assert grid.x.shape == (51,)
    assert grid.y.shape == (51,)
    assert grid.x[0] == pytest.approx(0.0)
    assert grid.x[-1] == pytest.approx(1.0)
    assert grid.y[0] == pytest.approx(0.0)
    assert grid.y[-1] == pytest.approx(1.0)


def test_initialize_fields_uses_ny_nx_shape():
    config = SimulationConfig(nx=31, ny=21)
    u, v, p = initialize_fields(config)

    assert u.shape == (21, 31)
    assert v.shape == (21, 31)
    assert p.shape == (21, 31)
    assert np.all(u == 0.0)
    assert np.all(v == 0.0)
    assert np.all(p == 0.0)


def test_grid_requires_at_least_two_points_per_direction():
    with pytest.raises(ValueError):
        create_grid(SimulationConfig(nx=1))
