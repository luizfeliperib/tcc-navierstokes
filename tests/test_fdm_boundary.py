import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.boundary import apply_velocity_boundary_conditions
from navierstokes.fdm.grid import initialize_fields


def test_lid_driven_cavity_velocity_boundary_conditions():
    config = SimulationConfig(nx=11, ny=9, lid_velocity=1.5)

    u, v, _ = initialize_fields(config)

    u[:, :] = 0.3
    v[:, :] = -0.2

    u_original = u.copy()
    v_original = v.copy()

    u_ext, v_ext = apply_velocity_boundary_conditions(u, v, config)

    assert u_ext.shape == (11, 13)
    assert v_ext.shape == (11, 13)

    # Os campos originais não devem ser alterados.
    np.testing.assert_array_equal(u, u_original)
    np.testing.assert_array_equal(v, v_original)

    # Os valores centrais devem ser preservados.
    np.testing.assert_array_equal(u_ext[1:-1, 1:-1], u)
    np.testing.assert_array_equal(v_ext[1:-1, 1:-1], v)

    # Parede inferior: velocidade nula.
    np.testing.assert_allclose(
        0.5 * (u_ext[0, 1:-1] + u_ext[1, 1:-1]),
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        0.5 * (v_ext[0, 1:-1] + v_ext[1, 1:-1]),
        0.0,
        atol=1e-12,
    )

    # Tampa superior: velocidade tangencial prescrita.
    np.testing.assert_allclose(
        0.5 * (u_ext[-1, 1:-1] + u_ext[-2, 1:-1]),
        config.lid_velocity,
        atol=1e-12,
    )

    # Tampa superior: velocidade normal nula.
    np.testing.assert_allclose(
        0.5 * (v_ext[-1, 1:-1] + v_ext[-2, 1:-1]),
        0.0,
        atol=1e-12,
    )

    # Parede esquerda: não deslizamento.
    np.testing.assert_allclose(
        0.5 * (u_ext[1:-1, 0] + u_ext[1:-1, 1]),
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        0.5 * (v_ext[1:-1, 0] + v_ext[1:-1, 1]),
        0.0,
        atol=1e-12,
    )

    # Parede direita: não deslizamento.
    np.testing.assert_allclose(
        0.5 * (u_ext[1:-1, -1] + u_ext[1:-1, -2]),
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        0.5 * (v_ext[1:-1, -1] + v_ext[1:-1, -2]),
        0.0,
        atol=1e-12,
    )


def test_boundary_conditions_reject_wrong_field_shape():
    config = SimulationConfig(nx=11, ny=9)
    u = np.zeros((9, 10))
    v = np.zeros((9, 11))

    with pytest.raises(ValueError):
        apply_velocity_boundary_conditions(u, v, config)
