
import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.boundary import interpolate_impermeable_faces
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.projection import (
    reconstruct_predicted_face_fluxes,
)
from navierstokes.fdm.solver import solve


def test_reconstruction_matches_interpolation():
    config = SimulationConfig(
        reynolds=100,
        nx=8,
        ny=8,
        dt=0.001,
        max_iterations=1,
    )
    grid = create_grid(config)

    rng = np.random.default_rng(42)
    u_star = rng.normal(size=grid.scalar_shape)
    v_star = rng.normal(size=grid.scalar_shape)

    expected_u, expected_v = interpolate_impermeable_faces(
        u_star, v_star, grid
    )

    actual_u, actual_v = reconstruct_predicted_face_fluxes(
        u_star, v_star, grid
    )

    np.testing.assert_allclose(actual_u, expected_u)
    np.testing.assert_allclose(actual_v, expected_v)

    assert np.all(actual_u[:, 0] == 0.0)
    assert np.all(actual_u[:, -1] == 0.0)
    assert np.all(actual_v[0, :] == 0.0)
    assert np.all(actual_v[-1, :] == 0.0)


@pytest.mark.parametrize(
    "mode", ["incremental", "reconstructed"]
)
def test_both_modes_conserve_mass(mode):
    config = SimulationConfig(
        reynolds=100,
        nx=8,
        ny=8,
        dt=0.001,
        max_iterations=10,
    )

    result = solve(
        config,
        backend="python",
        pressure_tolerance=1e-8,
        pressure_max_iterations=20000,
        face_flux_mode=mode,
        stop_on_steady=False,
        record_interval=1,
    )

    assert result.steps == 10
    assert result.history[-1].diagnostics.divergence_rms < 1e-6


def test_invalid_face_flux_mode():
    config = SimulationConfig(
        reynolds=100,
        nx=8,
        ny=8,
        dt=0.001,
        max_iterations=1,
    )

    with pytest.raises(ValueError, match="face_flux_mode"):
        solve(config, face_flux_mode="invalid")
