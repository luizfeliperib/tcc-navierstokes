
"""Operadores discretos para malha cartesiana cell-centered."""

import numpy as np
from numpy.typing import NDArray

from navierstokes.fdm.grid import CartesianGrid


def _check_shape(
    field: NDArray[np.float64],
    expected: tuple[int, int],
    name: str,
) -> None:
    if field.shape != expected:
        raise ValueError(
            f"{name}: esperado shape {expected}, "
            f"recebido {field.shape}"
        )


def divergence_faces(
    u_faces: NDArray[np.float64],
    v_faces: NDArray[np.float64],
    grid: CartesianGrid,
) -> NDArray[np.float64]:
    """Divergência conservativa calculada nos centros."""

    _check_shape(u_faces, grid.u_face_shape, "u_faces")
    _check_shape(v_faces, grid.v_face_shape, "v_faces")

    du_dx = (
        u_faces[:, 1:] - u_faces[:, :-1]
    ) / grid.dx

    dv_dy = (
        v_faces[1:, :] - v_faces[:-1, :]
    ) / grid.dy

    return du_dx + dv_dy


def gradient_faces(
    phi: NDArray[np.float64],
    grid: CartesianGrid,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gradiente nas faces com Neumann homogênea externa."""

    _check_shape(phi, grid.scalar_shape, "phi")

    gx = grid.zeros_u_faces()
    gy = grid.zeros_v_faces()

    gx[:, 1:-1] = (
        phi[:, 1:] - phi[:, :-1]
    ) / grid.dx

    gy[1:-1, :] = (
        phi[1:, :] - phi[:-1, :]
    ) / grid.dy

    return gx, gy


def laplacian_neumann(
    phi: NDArray[np.float64],
    grid: CartesianGrid,
) -> NDArray[np.float64]:
    """Laplaciano discreto L_h = D_h G_h."""

    gx, gy = gradient_faces(phi, grid)
    return divergence_faces(gx, gy, grid)


def total_divergence(
    u_faces: NDArray[np.float64],
    v_faces: NDArray[np.float64],
    grid: CartesianGrid,
) -> float:
    """Integral discreta da divergência sobre o domínio."""

    div = divergence_faces(u_faces, v_faces, grid)
    return float(np.sum(div) * grid.cell_area)
