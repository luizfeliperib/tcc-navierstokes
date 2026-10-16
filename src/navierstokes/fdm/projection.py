"""FDM-06: construção e projeção dos fluxos nas faces."""

import numpy as np
from numpy.typing import NDArray

from navierstokes.fdm.boundary import interpolate_impermeable_faces
from navierstokes.fdm.grid import CartesianGrid
from navierstokes.fdm.predictor import _check_field


def predict_face_fluxes(
    u: NDArray,
    v: NDArray,
    u_star: NDArray,
    v_star: NDArray,
    u_faces: NDArray,
    v_faces: NDArray,
    grid: CartesianGrid,
) -> tuple[NDArray, NDArray]:
    """Atualiza incrementalmente os fluxos normais da cavidade fechada.

    Os fluxos de entrada representam o estado projetado no instante n.
    Os campos centrais provisórios são produzidos pelo preditor FDM-05.

    Não modifica os arrays recebidos nem realiza projeção de pressão.
    """
    for name, field, shape in (
        ("u", u, grid.scalar_shape),
        ("v", v, grid.scalar_shape),
        ("u_star", u_star, grid.scalar_shape),
        ("v_star", v_star, grid.scalar_shape),
        ("u_faces", u_faces, grid.u_face_shape),
        ("v_faces", v_faces, grid.v_face_shape),
    ):
        _check_field(field, shape, name)

    if not (
        np.allclose(u_faces[:, [0, -1]], 0.0, atol=1e-14, rtol=0)
        and np.allclose(v_faces[[0, -1], :], 0.0, atol=1e-14, rtol=0)
    ):
        raise ValueError(
            "Cavidade impermeável exige fluxos normais externos nulos"
        )

    delta_u = u_star - u
    delta_v = v_star - v

    delta_u_faces, delta_v_faces = interpolate_impermeable_faces(
        delta_u, delta_v, grid
    )

    u_faces_star = u_faces + delta_u_faces
    v_faces_star = v_faces + delta_v_faces

    # Impõe exatamente a impermeabilidade nas fronteiras.
    u_faces_star[:, 0] = 0.0
    u_faces_star[:, -1] = 0.0
    v_faces_star[0, :] = 0.0
    v_faces_star[-1, :] = 0.0

    return u_faces_star, v_faces_star
