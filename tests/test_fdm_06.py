
"""Testes da construção incremental dos fluxos provisórios."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.gauss_seidel import GaussSeidelResult
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.operators import divergence_faces, gradient_faces
from navierstokes.fdm.projection import predict_face_fluxes, solve_projection_pressure


def test_projection_pressure_zero_fluxes():
    grid = create_grid(SimulationConfig(nx=6, ny=5))

    result = solve_projection_pressure(
        grid.zeros_u_faces(),
        grid.zeros_v_faces(),
        grid,
        density=1.0,
        dt=0.01,
    )

    assert isinstance(result, GaussSeidelResult)
    assert result.converged
    assert result.iterations == 0
    np.testing.assert_allclose(result.phi, 0.0, atol=1e-12)


def test_projection_pressure_manufactured_solution():
    grid = create_grid(SimulationConfig(nx=8, ny=6))
    x, y = grid.cell_centers()

    # Pressão manufaturada, definida até uma constante.
    exact = np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    exact -= np.mean(exact)

    density = 1.0
    dt = 0.01

    # Fluxos provisórios construídos a partir do gradiente
    # discreto da pressão manufaturada.
    gx, gy = gradient_faces(exact, grid)
    uf_star = (dt / density) * gx
    vf_star = (dt / density) * gy

    result = solve_projection_pressure(
        uf_star,
        vf_star,
        grid,
        density=density,
        dt=dt,
        tolerance=1e-10,
        max_iterations=10_000,
    )

    assert result.converged
    np.testing.assert_allclose(
        result.phi, exact, atol=1e-8, rtol=0
    )


def test_projection_pressure_rejects_invalid_inputs():
    grid = create_grid(SimulationConfig(nx=5, ny=4))
    uf = grid.zeros_u_faces()
    vf = grid.zeros_v_faces()

    with pytest.raises(ValueError, match="density"):
        solve_projection_pressure(
            uf, vf, grid, density=0.0, dt=0.01
        )

    with pytest.raises(ValueError, match="dt"):
        solve_projection_pressure(
            uf, vf, grid, density=1.0, dt=0.0
        )

    bad = uf.copy()
    bad[:, 0] = 1.0

    with pytest.raises(ValueError, match="fluxos normais"):
        solve_projection_pressure(
            bad, vf, grid, density=1.0, dt=0.01
        )

def test_zero_increment_preserves_existing_face_fluxes():
    grid = create_grid(SimulationConfig(nx=8, ny=6))
    rng = np.random.default_rng(42)

    u = rng.normal(size=grid.scalar_shape)
    v = rng.normal(size=grid.scalar_shape)

    uf = rng.normal(size=grid.u_face_shape)
    vf = rng.normal(size=grid.v_face_shape)
    uf[:, [0, -1]] = 0.0
    vf[[0, -1], :] = 0.0

    uf_star, vf_star = predict_face_fluxes(
        u, v, u.copy(), v.copy(), uf, vf, grid
    )

    np.testing.assert_array_equal(uf_star, uf)
    np.testing.assert_array_equal(vf_star, vf)


def test_increment_is_interpolated_to_internal_faces():
    grid = create_grid(SimulationConfig(nx=4, ny=3))

    u = grid.zeros_scalar()
    v = grid.zeros_scalar()
    u_star = np.tile(np.arange(4, dtype=float), (3, 1))
    v_star = np.tile(np.arange(3, dtype=float)[:, None], (1, 4))

    uf_star, vf_star = predict_face_fluxes(
        u, v, u_star, v_star,
        grid.zeros_u_faces(), grid.zeros_v_faces(), grid
    )

    np.testing.assert_allclose(
        uf_star[:, 1:-1],
        0.5 * (u_star[:, :-1] + u_star[:, 1:]),
    )
    np.testing.assert_allclose(
        vf_star[1:-1, :],
        0.5 * (v_star[:-1, :] + v_star[1:, :]),
    )

    np.testing.assert_array_equal(uf_star[:, [0, -1]], 0.0)
    np.testing.assert_array_equal(vf_star[[0, -1], :], 0.0)


def test_inputs_are_not_modified():
    grid = create_grid(SimulationConfig(nx=6, ny=5))
    rng = np.random.default_rng(123)

    fields = [
        rng.normal(size=grid.scalar_shape)
        for _ in range(4)
    ]
    fields.extend([
        grid.zeros_u_faces(),
        grid.zeros_v_faces(),
    ])
    originals = [field.copy() for field in fields]

    predict_face_fluxes(*fields, grid)

    for actual, original in zip(fields, originals):
        np.testing.assert_array_equal(actual, original)


def test_invalid_boundary_fluxes_are_rejected():
    grid = create_grid(SimulationConfig(nx=5, ny=4))
    u = grid.zeros_scalar()
    v = grid.zeros_scalar()
    uf = grid.zeros_u_faces()
    vf = grid.zeros_v_faces()

    uf[:, 0] = 1.0

    with pytest.raises(ValueError, match="fluxos normais"):
        predict_face_fluxes(u, v, u, v, uf, vf, grid)


def test_divergence_is_not_assumed_zero_before_projection():
    grid = create_grid(SimulationConfig(nx=8, ny=8))
    u = grid.zeros_scalar()
    v = grid.zeros_scalar()

    x, y = grid.cell_centers()
    u_star = x.copy()
    v_star = np.zeros_like(y)

    uf_star, vf_star = predict_face_fluxes(
        u, v, u_star, v_star,
        grid.zeros_u_faces(), grid.zeros_v_faces(), grid
    )

    div = divergence_faces(uf_star, vf_star, grid)

    assert np.max(np.abs(div)) > 1e-8
    assert abs(np.sum(div) * grid.cell_area) < 1e-12
