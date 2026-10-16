
"""Varredura Gauss-Seidel compilada, preservando a ordem lexicografica."""

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True)
def gauss_seidel_sweep(
    phi: NDArray[np.float64],
    rhs: NDArray[np.float64],
    center: NDArray[np.float64],
    east: NDArray[np.float64],
    west: NDArray[np.float64],
    north: NDArray[np.float64],
    south: NDArray[np.float64],
) -> None:
    """Executa uma varredura in-place, da esquerda para a direita."""

    ny, nx = phi.shape

    for j in range(ny):
        for i in range(nx):
            neighbor_sum = 0.0

            if i + 1 < nx:
                neighbor_sum += east[j, i] * phi[j, i + 1]

            if i > 0:
                neighbor_sum += west[j, i] * phi[j, i - 1]

            if j + 1 < ny:
                neighbor_sum += north[j, i] * phi[j + 1, i]

            if j > 0:
                neighbor_sum += south[j, i] * phi[j - 1, i]

            phi[j, i] = (
                rhs[j, i] + neighbor_sum
            ) / center[j, i]
