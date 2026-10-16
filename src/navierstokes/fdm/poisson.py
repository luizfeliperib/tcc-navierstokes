"""Montagem do sistema discreto de Poisson com Neumann homogênea."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from navierstokes.fdm.grid import CartesianGrid


@dataclass(frozen=True, slots=True)
class PoissonCoefficients:
    """Coeficientes de A = -L_h, armazenados por direção."""

    center: NDArray[np.float64]
    east: NDArray[np.float64]
    west: NDArray[np.float64]
    north: NDArray[np.float64]
    south: NDArray[np.float64]


def assemble_poisson_coefficients(
    grid: CartesianGrid,
) -> PoissonCoefficients:
    """Monta A = -D_h G_h, com Neumann homogênea nas paredes."""

    shape = grid.scalar_shape
    inv_dx2 = 1.0 / grid.dx**2
    inv_dy2 = 1.0 / grid.dy**2

    east = np.zeros(shape, dtype=np.float64)
    west = np.zeros(shape, dtype=np.float64)
    north = np.zeros(shape, dtype=np.float64)
    south = np.zeros(shape, dtype=np.float64)

    east[:, :-1] = inv_dx2
    west[:, 1:] = inv_dx2
    north[:-1, :] = inv_dy2
    south[1:, :] = inv_dy2

    center = east + west + north + south

    return PoissonCoefficients(
        center=center,
        east=east,
        west=west,
        north=north,
        south=south,
    )


def apply_poisson_system(
    phi: NDArray[np.float64],
    coefficients: PoissonCoefficients,
) -> NDArray[np.float64]:
    """Aplica A phi = -L_h phi, sem montar matriz densa."""

    if phi.shape != coefficients.center.shape:
        raise ValueError("Shape de phi incompatível com o sistema")

    result = coefficients.center * phi

    result[:, :-1] -= coefficients.east[:, :-1] * phi[:, 1:]
    result[:, 1:] -= coefficients.west[:, 1:] * phi[:, :-1]
    result[:-1, :] -= coefficients.north[:-1, :] * phi[1:, :]
    result[1:, :] -= coefficients.south[1:, :] * phi[:-1, :]

    return result
