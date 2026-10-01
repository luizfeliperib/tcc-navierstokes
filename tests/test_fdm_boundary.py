import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.boundary import apply_velocity_boundary_conditions
from navierstokes.fdm.grid import initialize_fields


def test_lid_driven_cavity_velocity_boundary_conditions():
    config = SimulationConfig(nx=11, ny=9, lid_velocity=1.5)
    u, v, _ = initialize_fields(config)

    u[:, :] = 7.0
    v[:, :] = -3.0
    apply_velocity_boundary_conditions(u, v, config)

    assert np.all(u[-1, :] == pytest.approx(config.lid_velocity))
    assert np.all(u[0, :] == pytest.approx(0.0))
    assert np.all(u[:-1, 0] == pytest.approx(0.0))
    assert np.all(u[:-1, -1] == pytest.approx(0.0))

    assert np.all(v[0, :] == pytest.approx(0.0))
    assert np.all(v[-1, :] == pytest.approx(0.0))
    assert np.all(v[:, 0] == pytest.approx(0.0))
    assert np.all(v[:, -1] == pytest.approx(0.0))

    # O interior não deve ser alterado pela aplicação das condições de contorno.
    assert np.all(u[1:-1, 1:-1] == pytest.approx(7.0))
    assert np.all(v[1:-1, 1:-1] == pytest.approx(-3.0))


def test_boundary_conditions_reject_wrong_field_shape():
    config = SimulationConfig(nx=11, ny=9)
    u = np.zeros((9, 10))
    v = np.zeros((9, 11))

    with pytest.raises(ValueError):
        apply_velocity_boundary_conditions(u, v, config)
