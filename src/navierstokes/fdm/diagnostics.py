
"""FDM-06.6: diagnósticos de conservação e consistência."""

from dataclasses import dataclass

import numpy as np

from navierstokes.fdm.grid import CartesianGrid
from navierstokes.fdm.operators import divergence_faces
from navierstokes.fdm.predictor import _check_field
from navierstokes.fdm.time_step import FlowState


@dataclass(frozen=True, slots=True)
class FlowDiagnostics:
    divergence_rms: float
    center_face_u_rms: float
    center_face_v_rms: float
    max_abs_u: float
    max_abs_v: float
    kinetic_energy_mean: float


def calculate_diagnostics(
    state: FlowState,
    grid: CartesianGrid,
) -> FlowDiagnostics:
    """Calcula diagnósticos sem modificar o estado."""

    for name, field, shape in (
        ("u", state.u, grid.scalar_shape),
        ("v", state.v, grid.scalar_shape),
        ("u_faces", state.u_faces, grid.u_face_shape),
        ("v_faces", state.v_faces, grid.v_face_shape),
        ("pressure", state.pressure, grid.scalar_shape),
    ):
        _check_field(field, shape, name)

    divergence = divergence_faces(
        state.u_faces, state.v_faces, grid
    )

    # Interpolação apenas para diagnóstico.
    # Não substitui os fluxos projetados do estado.
    u_interpolated = 0.5 * (
        state.u_faces[:, :-1] + state.u_faces[:, 1:]
    )
    v_interpolated = 0.5 * (
        state.v_faces[:-1, :] + state.v_faces[1:, :]
    )

    return FlowDiagnostics(
        divergence_rms=float(
            np.sqrt(np.mean(divergence**2))
        ),
        center_face_u_rms=float(
            np.sqrt(np.mean((state.u - u_interpolated)**2))
        ),
        center_face_v_rms=float(
            np.sqrt(np.mean((state.v - v_interpolated)**2))
        ),
        max_abs_u=float(np.max(np.abs(state.u))),
        max_abs_v=float(np.max(np.abs(state.v))),
        kinetic_energy_mean=float(
            0.5 * np.mean(state.u**2 + state.v**2)
        ),
    )
