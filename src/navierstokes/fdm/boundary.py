"""Condições de contorno da cavidade com tampa móvel."""

import numpy as np
from numpy.typing import NDArray

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import CartesianGrid


def apply_velocity_boundary_conditions(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    config: SimulationConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Constrói campos com células fantasmas para a
    cavidade com tampa móvel em malha cell-centered.

    Os campos originais não são modificados.

    Retorna:
        u_ext, v_ext: arrays (ny + 2, nx + 2).
    """

    expected = (config.ny, config.nx)

    if u.shape != expected or v.shape != expected:
        raise ValueError(
            f"u e v devem possuir shape {expected}; recebidos {u.shape} e {v.shape}"
        )

    u_ext = np.zeros(
        (config.ny + 2, config.nx + 2),
        dtype=np.float64,
    )
    v_ext = np.zeros_like(u_ext)

    # Copia os valores dos centros das células.
    u_ext[1:-1, 1:-1] = u
    v_ext[1:-1, 1:-1] = v

    # Parede inferior: u = 0, v = 0.
    u_ext[0, 1:-1] = -u_ext[1, 1:-1]
    v_ext[0, 1:-1] = -v_ext[1, 1:-1]

    # Parede superior: u = U_tampa, v = 0.
    u_ext[-1, 1:-1] = 2.0 * config.lid_velocity - u_ext[-2, 1:-1]
    v_ext[-1, 1:-1] = -v_ext[-2, 1:-1]

    # Parede esquerda: u = 0, v = 0.
    u_ext[1:-1, 0] = -u_ext[1:-1, 1]
    v_ext[1:-1, 0] = -v_ext[1:-1, 1]

    # Parede direita: u = 0, v = 0.
    u_ext[1:-1, -1] = -u_ext[1:-1, -2]
    v_ext[1:-1, -1] = -v_ext[1:-1, -2]

    return u_ext, v_ext


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
        raise ValueError(f"u e v devem possuir shape {expected}")

    u_faces = grid.zeros_u_faces()
    v_faces = grid.zeros_v_faces()

    u_faces[:, 1:-1] = 0.5 * (u[:, :-1] + u[:, 1:])

    v_faces[1:-1, :] = 0.5 * (v[:-1, :] + v[1:, :])

    return u_faces, v_faces
