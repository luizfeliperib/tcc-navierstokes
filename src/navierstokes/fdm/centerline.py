"""FDM-07.4.3: extração dos perfis centrais da cavidade."""

import numpy as np
from numpy.typing import NDArray

from navierstokes.fdm.grid import CartesianGrid


def extract_centerline_profiles(
    grid: CartesianGrid,
    u: NDArray,
    v: NDArray,
) -> dict[str, NDArray]:
    """Extrai os perfis centrais de velocidade.

    Perfil vertical:
        u(x=Lx/2, y)

    Perfil horizontal:
        v(x, y=Ly/2)

    As velocidades são interpoladas linearmente
    entre os centros das células.

    Não inclui pontos nas paredes.
    """

    if u.shape != grid.scalar_shape:
        raise ValueError("Dimensões de u incompatíveis com a malha.")

    if v.shape != grid.scalar_shape:
        raise ValueError("Dimensões de v incompatíveis com a malha.")

    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        raise ValueError("Campos de velocidade devem ser finitos.")

    # Comprimentos físicos do domínio.
    lx = grid.nx * grid.dx
    ly = grid.ny * grid.dy

    # Coordenadas geométricas das linhas centrais.
    x_center = lx / 2.0
    y_center = ly / 2.0

    # --------------------------------------------
    # Perfil vertical: u(x=Lx/2, y)
    # --------------------------------------------

    u_vertical = np.array(
        [
            np.interp(
                x_center,
                grid.x,
                u[j, :],
            )
            for j in range(grid.ny)
        ],
        dtype=np.float64,
    )

    # --------------------------------------------
    # Perfil horizontal: v(x, y=Ly/2)
    # --------------------------------------------

    v_horizontal = np.array(
        [
            np.interp(
                y_center,
                grid.y,
                v[:, i],
            )
            for i in range(grid.nx)
        ],
        dtype=np.float64,
    )

    return {
        "y": grid.y.copy(),
        "u_vertical": u_vertical,
        "x": grid.x.copy(),
        "v_horizontal": v_horizontal,
    }
