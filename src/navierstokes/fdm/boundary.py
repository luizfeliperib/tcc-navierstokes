"""Condições de contorno da cavidade com tampa móvel."""

import numpy as np
from numpy.typing import NDArray

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import CartesianGrid


def apply_velocity_boundary_conditions(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    config: SimulationConfig,
) -> None:
    """Aplica, in-place, as condições de não deslizamento e da tampa móvel."""
    expected_shape = (config.ny, config.nx)
    if u.shape != expected_shape or v.shape != expected_shape:
        raise ValueError(
            f"u e v devem possuir shape {expected_shape}; "
            f"recebidos {u.shape} e {v.shape}"
        )

    # Paredes inferior e laterais: condição de não deslizamento.
    u[0, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    # Tampa superior: velocidade horizontal prescrita.
    # A tampa prevalece nos dois cantos superiores.
    u[-1, :] = config.lid_velocity

    # Não há velocidade normal em nenhuma parede.
    v[0, :] = 0.0
    v[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0


def interpolate_impermeable_faces(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    grid: "CartesianGrid",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Interpola velocidades centrais para faces internas.

    Nas faces externas, impõe fluxo normal nulo.
    Aplicável a domínios retangulares fechados.
    """

    expected = grid.scalar_shape

    if u.shape != expected or v.shape != expected:
        raise ValueError(
            f"u e v devem possuir shape {expected}"
        )

    u_faces = grid.zeros_u_faces()
    v_faces = grid.zeros_v_faces()

    u_faces[:, 1:-1] = 0.5 * (
        u[:, :-1] + u[:, 1:]
    )

    v_faces[1:-1, :] = 0.5 * (
        v[:-1, :] + v[1:, :]
    )

    return u_faces, v_faces
