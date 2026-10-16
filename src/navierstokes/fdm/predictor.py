"""FDM-05: convecção conservativa, difusão e preditor explícito da cavidade."""

import numpy as np
from numpy.typing import NDArray

from navierstokes.config import SimulationConfig
from navierstokes.fdm.boundary import apply_velocity_boundary_conditions
from navierstokes.fdm.grid import CartesianGrid
from navierstokes.fdm.operators import divergence_faces


def _check_field(field: NDArray, shape: tuple[int, int], name: str) -> None:
    if not isinstance(field, np.ndarray) or field.shape != shape:
        raise ValueError(f"{name}: esperado ndarray com shape {shape}")
    if not np.issubdtype(field.dtype, np.number) or not np.all(np.isfinite(field)):
        raise ValueError(f"{name}: valores devem ser numéricos e finitos")


def _check_geometry(grid: CartesianGrid, config: SimulationConfig) -> None:
    if (grid.nx, grid.ny) != (config.nx, config.ny):
        raise ValueError("grid e config têm dimensões incompatíveis")
    if not np.isclose(grid.dx * grid.nx, config.lx) or not np.isclose(
        grid.dy * grid.ny, config.ly
    ):
        raise ValueError("grid e config têm extensões incompatíveis")


def convection_upwind(
    u: NDArray,
    v: NDArray,
    u_faces: NDArray,
    v_faces: NDArray,
    grid: CartesianGrid,
) -> tuple[NDArray, NDArray]:
    """Divergência de (u*u, v*u) e (u*v, v*v), com upwind nas faces.

    Fluxos normais nas faces externas devem ser nulos (cavidade fechada).
    """
    for name, arr, shape in (
        ("u", u, grid.scalar_shape),
        ("v", v, grid.scalar_shape),
        ("u_faces", u_faces, grid.u_face_shape),
        ("v_faces", v_faces, grid.v_face_shape),
    ):
        _check_field(arr, shape, name)
    if not (
        np.allclose(u_faces[:, [0, -1]], 0, atol=1e-14, rtol=0)
        and np.allclose(v_faces[[0, -1], :], 0, atol=1e-14, rtol=0)
    ):
        raise ValueError("Cavidade impermeável exige fluxos normais externos nulos")

    # Fluxos de u: componente transportada upwind em faces verticais/horizontais.
    flux_ux = grid.zeros_u_faces()
    flux_uy = grid.zeros_v_faces()
    flux_vx = grid.zeros_u_faces()
    flux_vy = grid.zeros_v_faces()

    flux_ux[:, 1:-1] = u_faces[:, 1:-1] * np.where(
        u_faces[:, 1:-1] >= 0, u[:, :-1], u[:, 1:]
    )
    flux_vx[:, 1:-1] = u_faces[:, 1:-1] * np.where(
        u_faces[:, 1:-1] >= 0, v[:, :-1], v[:, 1:]
    )
    flux_uy[1:-1, :] = v_faces[1:-1, :] * np.where(
        v_faces[1:-1, :] >= 0, u[:-1, :], u[1:, :]
    )
    flux_vy[1:-1, :] = v_faces[1:-1, :] * np.where(
        v_faces[1:-1, :] >= 0, v[:-1, :], v[1:, :]
    )
    return (
        divergence_faces(flux_ux, flux_uy, grid),
        divergence_faces(flux_vx, flux_vy, grid),
    )


def velocity_laplacian(
    u: NDArray,
    v: NDArray,
    grid: CartesianGrid,
    config: SimulationConfig,
) -> tuple[NDArray, NDArray]:
    """Laplacianos centrais de 5 pontos com Dirichlet via ghost cells."""
    _check_geometry(grid, config)
    _check_field(u, grid.scalar_shape, "u")
    _check_field(v, grid.scalar_shape, "v")
    u_ext, v_ext = apply_velocity_boundary_conditions(u, v, config)

    def lap(a: NDArray) -> NDArray:
        core = a[1:-1, 1:-1]
        return (a[1:-1, 2:] - 2 * core + a[1:-1, :-2]) / grid.dx**2 + (
            a[2:, 1:-1] - 2 * core + a[:-2, 1:-1]
        ) / grid.dy**2

    return lap(u_ext), lap(v_ext)


def stable_timestep_limit(
    u_faces: NDArray,
    v_faces: NDArray,
    grid: CartesianGrid,
    viscosity: float,
    safety_factor: float = 0.8,
) -> float:
    """Limite suficiente conservador para advecção upwind + difusão explícita.

    Usa a soma dos máximos das velocidades normais das faces de cada célula.
    """
    _check_field(u_faces, grid.u_face_shape, "u_faces")
    _check_field(v_faces, grid.v_face_shape, "v_faces")
    if not np.isfinite(viscosity) or viscosity <= 0:
        raise ValueError("viscosity deve ser finita e positiva")
    if not np.isfinite(safety_factor) or not 0 < safety_factor <= 1:
        raise ValueError("safety_factor deve pertencer a (0, 1]")
    ux = np.maximum(np.abs(u_faces[:, :-1]), np.abs(u_faces[:, 1:]))
    vy = np.maximum(np.abs(v_faces[:-1, :]), np.abs(v_faces[1:, :]))
    advective_rate = np.max(ux / grid.dx + vy / grid.dy)
    diffusive_rate = 2 * viscosity * (1 / grid.dx**2 + 1 / grid.dy**2)
    return float(safety_factor / (advective_rate + diffusive_rate))


def velocity_predictor(
    u: NDArray,
    v: NDArray,
    u_faces: NDArray,
    v_faces: NDArray,
    grid: CartesianGrid,
    config: SimulationConfig,
) -> tuple[NDArray, NDArray]:
    """Preditor não incremental por Euler explícito; não projeta nem resolve p."""
    _check_geometry(grid, config)
    if not np.isfinite(config.dt) or config.dt <= 0:
        raise ValueError("dt deve ser finito e positivo")
    nu = config.kinematic_viscosity
    dt_max = stable_timestep_limit(u_faces, v_faces, grid, nu)
    if config.dt > dt_max * (1 + 1e-12):
        raise ValueError(
            f"dt={config.dt:.8g} excede limite estimado "
            f"dt_max={dt_max:.8g} (fator de segurança 0.8)"
        )
    cu, cv = convection_upwind(u, v, u_faces, v_faces, grid)
    lu, lv = velocity_laplacian(u, v, grid, config)
    return u + config.dt * (-cu + nu * lu), v + config.dt * (-cv + nu * lv)
