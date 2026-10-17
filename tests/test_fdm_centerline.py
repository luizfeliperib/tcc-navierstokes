"""Testes da extração dos perfis centrais FDM-07.4.3."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.centerline import (
    extract_centerline_profiles,
)
from navierstokes.fdm.grid import create_grid


@pytest.mark.parametrize(
    "nx, ny",
    [
        (8, 6),
        (7, 5),
        (32, 32),
    ],
)
def test_linear_interpolation(nx, ny):
    """Campos lineares devem ser interpolados exatamente."""

    config = SimulationConfig(
        nx=nx,
        ny=ny,
        lx=1.0,
        ly=1.0,
    )

    grid = create_grid(config)

    xx, yy = np.meshgrid(
        grid.x,
        grid.y,
        indexing="xy",
    )

    u = 2.0 * xx + 3.0 * yy
    v = 4.0 * xx - 2.0 * yy

    profiles = extract_centerline_profiles(
        grid,
        u,
        v,
    )

    expected_u = 1.0 + 3.0 * grid.y
    expected_v = 4.0 * grid.x - 1.0

    np.testing.assert_allclose(
        profiles["u_vertical"],
        expected_u,
        rtol=0,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        profiles["v_horizontal"],
        expected_v,
        rtol=0,
        atol=1e-14,
    )

    np.testing.assert_array_equal(
        profiles["x"],
        grid.x,
    )

    np.testing.assert_array_equal(
        profiles["y"],
        grid.y,
    )


def test_rejects_invalid_shapes():
    config = SimulationConfig(
        nx=8,
        ny=8,
    )

    grid = create_grid(config)

    u = np.zeros((8, 7))
    v = np.zeros((8, 8))

    with pytest.raises(ValueError):
        extract_centerline_profiles(
            grid,
            u,
            v,
        )


def test_rejects_nonfinite_values():
    config = SimulationConfig(
        nx=8,
        ny=8,
    )

    grid = create_grid(config)

    u = np.zeros((8, 8))
    v = np.zeros((8, 8))

    u[3, 4] = np.nan

    with pytest.raises(ValueError):
        extract_centerline_profiles(
            grid,
            u,
            v,
        )
