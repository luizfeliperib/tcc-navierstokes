
"""Testes da construção incremental dos fluxos provisórios."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.gauss_seidel import GaussSeidelResult
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.operators import divergence_faces, gradient_faces
from navierstokes.fdm.poisson import (
    apply_poisson_system,
    assemble_poisson_coefficients,
)
from navierstokes.fdm.projection import (
    correct_face_fluxes,
    predict_face_fluxes,
    solve_projection_pressure,
    correct_center_velocities,
    pressure_gradient_centers
)

def test_center_pressure_gradient_matches_central_difference():
    grid = create_grid(SimulationConfig(nx=8, ny=6))
    rng = np.random.default_rng(17)
    pressure = rng.normal(size=grid.scalar_shape)

    gx, gy = pressure_gradient_centers(pressure, grid)

    np.testing.assert_allclose(
        gx[:, 1:-1],
        (pressure[:, 2:] - pressure[:, :-2]) / (2 * grid.dx),
    )
    np.testing.assert_allclose(
        gy[1:-1, :],
        (pressure[2:, :] - pressure[:-2, :]) / (2 * grid.dy),
    )

def test_center_pressure_gradient_boundary_closure():
    grid = create_grid(SimulationConfig(nx=7, ny=5))
    rng = np.random.default_rng(18)
    pressure = rng.normal(size=grid.scalar_shape)

    gx, gy = pressure_gradient_centers(pressure, grid)

    np.testing.assert_allclose(
        gx[:, 0],
        (pressure[:, 1] - pressure[:, 0]) / (2 * grid.dx),
    )
    np.testing.assert_allclose(
        gx[:, -1],
        (pressure[:, -1] - pressure[:, -2]) / (2 * grid.dx),
    )
    np.testing.assert_allclose(
        gy[0, :],
        (pressure[1, :] - pressure[0, :]) / (2 * grid.dy),
    )
    np.testing.assert_allclose(
        gy[-1, :],
        (pressure[-1, :] - pressure[-2, :]) / (2 * grid.dy),
    )

def test_center_correction_with_constant_pressure():
    grid = create_grid(SimulationConfig(nx=8, ny=6))
    rng = np.random.default_rng(19)

    u_star = rng.normal(size=grid.scalar_shape)
    v_star = rng.normal(size=grid.scalar_shape)
    pressure = np.full(grid.scalar_shape, 7.0)

    u_new, v_new = correct_center_velocities(
        u_star, v_star, pressure, grid,
        density=1.0, dt=0.01,
    )

    np.testing.assert_array_equal(u_new, u_star)
    np.testing.assert_array_equal(v_new, v_star)

def test_center_correction_matches_pressure_gradient():
    grid = create_grid(SimulationConfig(nx=8, ny=6))
    x, y = grid.cell_centers()

    pressure = np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    gx, gy = pressure_gradient_centers(pressure, grid)

    density = 2.0
    dt = 0.02
    u_star = np.ones(grid.scalar_shape)
    v_star = np.full(grid.scalar_shape, -0.5)

    u_new, v_new = correct_center_velocities(
        u_star, v_star, pressure, grid,
        density=density, dt=dt,
    )

    np.testing.assert_allclose(
        u_new, u_star - (dt / density) * gx
    )
    np.testing.assert_allclose(
        v_new, v_star - (dt / density) * gy
    )

def test_center_correction_preserves_inputs():
    grid = create_grid(SimulationConfig(nx=6, ny=5))
    rng = np.random.default_rng(20)

    u_star = rng.normal(size=grid.scalar_shape)
    v_star = rng.normal(size=grid.scalar_shape)
    pressure = rng.normal(size=grid.scalar_shape)

    originals = [
        arr.copy() for arr in (u_star, v_star, pressure)
    ]

    correct_center_velocities(
        u_star, v_star, pressure, grid,
        density=1.0, dt=0.01,
    )

    for actual, original in zip(
        (u_star, v_star, pressure), originals
    ):
        np.testing.assert_array_equal(actual, original)

def test_center_correction_rejects_invalid_parameters():
    grid = create_grid(SimulationConfig(nx=5, ny=4))
    field = grid.zeros_scalar()

    with pytest.raises(ValueError, match="density"):
        correct_center_velocities(
            field, field, field, grid,
            density=0.0, dt=0.01,
        )

    with pytest.raises(ValueError, match="dt"):
        correct_center_velocities(
            field, field, field, grid,
            density=1.0, dt=0.0,
        )

def test_face_correction_preserves_zero_pressure_fluxes():
    grid = create_grid(SimulationConfig(nx=7, ny=5))
    rng = np.random.default_rng(11)

    uf = rng.normal(size=grid.u_face_shape)
    vf = rng.normal(size=grid.v_face_shape)
    uf[:, [0, -1]] = 0.0
    vf[[0, -1], :] = 0.0

    uf_new, vf_new = correct_face_fluxes(
        uf, vf, grid.zeros_scalar(), grid,
        density=1.0, dt=0.01,
    )

    np.testing.assert_array_equal(uf_new, uf)
    np.testing.assert_array_equal(vf_new, vf)


def test_face_correction_manufactured_pressure():
    grid = create_grid(SimulationConfig(nx=8, ny=6))
    x, y = grid.cell_centers()

    pressure = np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    gx, gy = gradient_faces(pressure, grid)

    density = 1.0
    dt = 0.01

    uf_star = (dt / density) * gx
    vf_star = (dt / density) * gy

    uf_new, vf_new = correct_face_fluxes(
        uf_star, vf_star, pressure, grid,
        density=density, dt=dt,
    )

    np.testing.assert_allclose(uf_new, 0.0, atol=1e-14)
    np.testing.assert_allclose(vf_new, 0.0, atol=1e-14)


def test_face_projection_divergence_matches_poisson_residual():
    grid = create_grid(SimulationConfig(nx=8, ny=6))
    x, y = grid.cell_centers()

    exact = np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    gx, gy = gradient_faces(exact, grid)

    density = 1.0
    dt = 0.01

    uf_star = (dt / density) * gx
    vf_star = (dt / density) * gy

    result = solve_projection_pressure(
        uf_star, vf_star, grid,
        density=density,
        dt=dt,
        tolerance=1e-10,
        max_iterations=10_000,
    )

    uf_new, vf_new = correct_face_fluxes(
        uf_star, vf_star, result.phi, grid,
        density=density, dt=dt,
    )

    divergence_new = divergence_faces(uf_new, vf_new, grid)

    coefficients = assemble_poisson_coefficients(grid)
    rhs = -(density / dt) * divergence_faces(
        uf_star, vf_star, grid
    )
    residual_field = rhs - apply_poisson_system(
        result.phi, coefficients
    )

    np.testing.assert_allclose(
        divergence_new,
        -(dt / density) * residual_field,
        atol=1e-12,
        rtol=1e-10,
    )

    rms_divergence = np.linalg.norm(divergence_new) / np.sqrt(
        divergence_new.size
    )

    assert rms_divergence <= (
        (dt / density) * result.residual + 1e-12
    )


def test_face_correction_preserves_inputs_and_walls():
    grid = create_grid(SimulationConfig(nx=7, ny=5))
    rng = np.random.default_rng(23)

    uf = rng.normal(size=grid.u_face_shape)
    vf = rng.normal(size=grid.v_face_shape)
    uf[:, [0, -1]] = 0.0
    vf[[0, -1], :] = 0.0
    pressure = rng.normal(size=grid.scalar_shape)

    originals = [arr.copy() for arr in (uf, vf, pressure)]

    uf_new, vf_new = correct_face_fluxes(
        uf, vf, pressure, grid,
        density=1.0, dt=0.01,
    )

    for actual, original in zip(
        (uf, vf, pressure), originals
    ):
        np.testing.assert_array_equal(actual, original)

    np.testing.assert_array_equal(uf_new[:, [0, -1]], 0.0)
    np.testing.assert_array_equal(vf_new[[0, -1], :], 0.0)


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
