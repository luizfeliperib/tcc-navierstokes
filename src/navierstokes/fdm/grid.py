
"""Malha cartesiana centrada nas células para o solver FDM."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from navierstokes.config import SimulationConfig


@dataclass(frozen=True, slots=True)
class CartesianGrid:
    """Malha cartesiana uniforme bidimensional, cell-centered."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    dx: float
    dy: float

    @property
    def nx(self) -> int:
        return self.x.size

    @property
    def ny(self) -> int:
        return self.y.size

    @property
    def scalar_shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    @property
    def u_face_shape(self) -> tuple[int, int]:
        return self.ny, self.nx + 1

    @property
    def v_face_shape(self) -> tuple[int, int]:
        return self.ny + 1, self.nx

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    def cell_centers(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Retorna as coordenadas 2D dos centros."""
        return np.meshgrid(self.x, self.y, indexing="xy")

    def zeros_scalar(self) -> NDArray[np.float64]:
        return np.zeros(self.scalar_shape, dtype=np.float64)

    def zeros_u_faces(self) -> NDArray[np.float64]:
        return np.zeros(self.u_face_shape, dtype=np.float64)

    def zeros_v_faces(self) -> NDArray[np.float64]:
        return np.zeros(self.v_face_shape, dtype=np.float64)


def create_grid(config: SimulationConfig) -> CartesianGrid:
    """Cria uma malha cartesiana uniforme centrada nas células."""

    if (
        isinstance(config.nx, (bool, np.bool_))
        or not isinstance(config.nx, (int, np.integer))
        or isinstance(config.ny, (bool, np.bool_))
        or not isinstance(config.ny, (int, np.integer))
    ):
        raise TypeError("nx e ny devem ser inteiros")

    if config.nx < 2 or config.ny < 2:
        raise ValueError("nx e ny devem ser maiores ou iguais a 2")

    if not np.isfinite(config.lx) or not np.isfinite(config.ly):
        raise ValueError("lx e ly devem ser finitos")

    if config.lx <= 0 or config.ly <= 0:
        raise ValueError("lx e ly devem ser positivos")

    dx = config.lx / config.nx
    dy = config.ly / config.ny

    x = (np.arange(config.nx, dtype=np.float64) + 0.5) * dx
    y = (np.arange(config.ny, dtype=np.float64) + 0.5) * dy

    return CartesianGrid(x=x, y=y, dx=dx, dy=dy)


def initialize_fields(
    config: SimulationConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Inicializa os campos centrais u, v e p com zeros."""

    shape = (config.ny, config.nx)

    u = np.zeros(shape, dtype=np.float64)
    v = np.zeros(shape, dtype=np.float64)
    p = np.zeros(shape, dtype=np.float64)

    return u, v, p
