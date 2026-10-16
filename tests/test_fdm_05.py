"""Testes FDM-05: propriedades discretas, fronteiras, Euler e estabilidade."""

from dataclasses import replace

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.operators import divergence_faces
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.predictor import (
    convection_upwind,
    stable_timestep_limit,
    velocity_laplacian,
    velocity_predictor,
)


def setup_case(nx=12, ny=10, dt=1e-4, lid_velocity=1.0):
    cfg = SimulationConfig(nx=nx, ny=ny, dt=dt, lid_velocity=lid_velocity)
    grid = create_grid(cfg)
    return cfg, grid


def test_zero_transport_and_inputs_unchanged():
    _, grid = setup_case()
    rng = np.random.default_rng(123)
    u = rng.standard_normal(grid.scalar_shape)
    v = rng.standard_normal(grid.scalar_shape)
    u0, v0 = u.copy(), v.copy()
    cu, cv = convection_upwind(u, v, grid.zeros_u_faces(),
                              grid.zeros_v_faces(), grid)
    np.testing.assert_array_equal(cu, 0)
    np.testing.assert_array_equal(cv, 0)
    np.testing.assert_array_equal(u, u0)
    np.testing.assert_array_equal(v, v0)


def test_convection_global_conservation_and_constant_field():
    _, grid = setup_case()
    rng = np.random.default_rng(12)
    u = rng.normal(size=grid.scalar_shape)
    v = rng.normal(size=grid.scalar_shape)
    uf = rng.normal(size=grid.u_face_shape)
    vf = rng.normal(size=grid.v_face_shape)
    uf[:, 0] = uf[:, -1] = 0
    vf[0, :] = vf[-1, :] = 0
    cu, cv = convection_upwind(u, v, uf, vf, grid)
    assert abs(np.sum(cu) * grid.cell_area) < 1e-12
    assert abs(np.sum(cv) * grid.cell_area) < 1e-12
    # Constantes só têm advecção zero se os fluxos transportadores forem
    # discretamente solenoidais. Fluxos arbitrários NÃO satisfazem isso.
    uc = np.full(grid.scalar_shape, 2.0)
    vc = np.full(grid.scalar_shape, -3.0)
    cu, cv = convection_upwind(uc, vc, grid.zeros_u_faces(),
                              grid.zeros_v_faces(), grid)
    np.testing.assert_array_equal(cu, 0)
    np.testing.assert_array_equal(cv, 0)


def test_laplacian_constant_zero_for_stationary_walls():
    cfg, grid = setup_case(lid_velocity=0)
    u = grid.zeros_scalar()
    v = grid.zeros_scalar()
    lu, lv = velocity_laplacian(u, v, grid, cfg)
    np.testing.assert_array_equal(lu, 0)
    np.testing.assert_array_equal(lv, 0)


def test_lid_drives_top_row_only_at_initial_rest():
    cfg, grid = setup_case()
    u = grid.zeros_scalar()
    v = grid.zeros_scalar()
    lu, lv = velocity_laplacian(u, v, grid, cfg)
    np.testing.assert_allclose(lu[-1, :], 2 * cfg.lid_velocity / grid.dy**2)
    np.testing.assert_array_equal(lu[:-1, :], 0)
    np.testing.assert_array_equal(lv, 0)


def test_laplacian_quadratic_interior_and_refinement():
    errors = []
    for n in (10, 20, 40):
        cfg, grid = setup_case(nx=n, ny=n)
        x, y = grid.cell_centers()
        # Função suave compatível com paredes estacionárias e tampa móvel:
        # verificamos o operador apenas no interior (sem fronteiras).
        u = np.sin(np.pi * x) * np.sin(np.pi * y)
        v = np.zeros_like(u)
        lu, _ = velocity_laplacian(u, v, grid, cfg)
        exact = -2 * np.pi**2 * u
        errors.append(np.max(np.abs(lu[1:-1, 1:-1] - exact[1:-1, 1:-1])))
    assert errors[1] < errors[0] / 3.5
    assert errors[2] < errors[1] / 3.5


def test_euler_predictor_matches_initial_lid_diffusion():
    cfg, grid = setup_case(dt=1e-5)
    u = grid.zeros_scalar()
    v = grid.zeros_scalar()
    u_star, v_star = velocity_predictor(
        u, v, grid.zeros_u_faces(), grid.zeros_v_faces(), grid, cfg
    )
    expected = grid.zeros_scalar()
    expected[-1, :] = cfg.dt * cfg.kinematic_viscosity * (
        2 * cfg.lid_velocity / grid.dy**2
    )
    np.testing.assert_allclose(u_star, expected)
    np.testing.assert_array_equal(v_star, 0)
    np.testing.assert_array_equal(u, 0)


def test_stability_validation():
    cfg, grid = setup_case()
    uf = grid.zeros_u_faces()
    vf = grid.zeros_v_faces()
    limit = stable_timestep_limit(uf, vf, grid, cfg.kinematic_viscosity)
    assert limit > 0
    u = grid.zeros_scalar()
    v = grid.zeros_scalar()
    with pytest.raises(ValueError, match="excede limite"):
        velocity_predictor(u, v, uf, vf, grid, replace(cfg, dt=limit * 2))
    with pytest.raises(ValueError, match="dt deve"):
        velocity_predictor(u, v, uf, vf, grid, replace(cfg, dt=0))
    with pytest.raises(ValueError, match="safety_factor"):
        stable_timestep_limit(uf, vf, grid, cfg.kinematic_viscosity, 1.2)
    with pytest.raises(ValueError, match="fluxos normais"):
        bad = uf.copy()
        bad[:, 0] = 1
        convection_upwind(u, v, bad, vf, grid)

def test_convection_preserves_constant_fields_with_nonzero_transport():
    _, grid = setup_case(nx=20, ny=16)

    # Função de corrente definida nos vértices da malha.
    xv = np.linspace(0.0, grid.nx * grid.dx, grid.nx + 1)
    yv = np.linspace(0.0, grid.ny * grid.dy, grid.ny + 1)
    x, y = np.meshgrid(xv, yv)

    psi = np.sin(np.pi * x / xv[-1]) * np.sin(np.pi * y / yv[-1])

    # Fluxos normais nas faces: divergência discreta exatamente nula,
    # salvo erros de arredondamento.
    u_faces = (psi[1:, :] - psi[:-1, :]) / grid.dy
    v_faces = -(psi[:, 1:] - psi[:, :-1]) / grid.dx

    # O transporte não é trivial.
    assert np.max(np.abs(u_faces)) > 0
    assert np.max(np.abs(v_faces)) > 0

    # Campo transportado constante.
    u = np.full(grid.scalar_shape, 2.0)
    v = np.full(grid.scalar_shape, -3.0)

    cu, cv = convection_upwind(u, v, u_faces, v_faces, grid)

    np.testing.assert_allclose(cu, 0.0, atol=1e-12)
    np.testing.assert_allclose(cv, 0.0, atol=1e-12)
    np.testing.assert_allclose(divergence_faces(u_faces, v_faces, grid),0.0,atol=1e-12,)

def test_convection_upwind_first_order_refinement():
    errors = []

    for n in (20, 40, 80, 160):
        _, grid = setup_case(nx=n, ny=n)

        # Coordenadas dos vértices.
        xv = np.linspace(0.0, 1.0, n + 1)
        yv = np.linspace(0.0, 1.0, n + 1)
        x_vertex, y_vertex = np.meshgrid(xv, yv)

        psi = (
            np.sin(np.pi * x_vertex)
            * np.sin(np.pi * y_vertex)
        )

        # Fluxos discretamente incompressíveis.
        u_faces = (psi[1:, :] - psi[:-1, :]) / grid.dy
        v_faces = -(psi[:, 1:] - psi[:, :-1]) / grid.dx

        # Campo transportado phi = psi avaliado nos centros.
        x, y = grid.cell_centers()
        phi = np.sin(np.pi * x) * np.sin(np.pi * y)

        # Usamos a primeira componente como escalar transportado.
        zeros = grid.zeros_scalar()

        conv_phi, _ = convection_upwind(
            phi, zeros, u_faces, v_faces, grid
        )

        # A solução analítica é identicamente zero.
        error = np.sqrt(np.mean(conv_phi**2))
        errors.append(error)

    # Para upwind de primeira ordem, ao dobrar a resolução,
    # esperamos aproximadamente reduzir o erro pela metade.
    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(2)
        for i in range(len(errors) - 1)
    ]

    assert all(0.8 < order < 1.2 for order in orders[-2:])
