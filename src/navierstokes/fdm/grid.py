"""Malha cartesiana e inicialização dos campos do solver FDM."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from navierstokes.config import SimulationConfig


@dataclass(frozen=True, slots=True)
class CartesianGrid:
    """Malha cartesiana uniforme bidimensional."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    dx: float
    dy: float


def create_grid(config: SimulationConfig) -> CartesianGrid:
    """Cria uma malha cartesiana uniforme a partir da configuração."""
    if config.nx < 2 or config.ny < 2:
        raise ValueError("nx e ny devem ser maiores ou iguais a 2")
    if config.lx <= 0 or config.ly <= 0:
        raise ValueError("lx e ly devem ser positivos")

    dx = config.lx / (config.nx - 1)
    dy = config.ly / (config.ny - 1)
    x = np.linspace(0.0, config.lx, config.nx)
    y = np.linspace(0.0, config.ly, config.ny)

    return CartesianGrid(x=x, y=y, dx=dx, dy=dy)


def initialize_fields(
    config: SimulationConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Inicializa os campos u, v e p com zeros."""
    shape = (config.ny, config.nx)
    u = np.zeros(shape, dtype=np.float64)
    v = np.zeros(shape, dtype=np.float64)
    p = np.zeros(shape, dtype=np.float64)
    return u, v, p
