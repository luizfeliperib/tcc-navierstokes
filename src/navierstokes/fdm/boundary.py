"""Condições de contorno da cavidade com tampa móvel."""

import numpy as np
from numpy.typing import NDArray

from navierstokes.config import SimulationConfig


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
