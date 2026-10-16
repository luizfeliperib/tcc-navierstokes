"""Diagnósticos espaciais de consistência centro-face para FDM."""

from dataclasses import dataclass

import numpy as np

from navierstokes.fdm.grid import CartesianGrid
from navierstokes.fdm.predictor import _check_field
from navierstokes.fdm.time_step import FlowState


@dataclass(frozen=True, slots=True)
class ConsistencyDiagnostics:
    u_total: float
    u_interior: float
    u_boundary: float
    v_total: float
    v_interior: float
    v_boundary: float
    u_relative_percent: float
    v_relative_percent: float


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def calculate_consistency(
    state: FlowState,
    grid: CartesianGrid,
) -> ConsistencyDiagnostics:
    """Compara velocidades centrais com médias de faces.

    Interior: exclui a primeira camada de células.
    Fronteira: inclui somente a primeira camada de células.
    """

    if grid.nx < 3 or grid.ny < 3:
        raise ValueError("O diagnóstico exige nx >= 3 e ny >= 3.")

    for name, field, shape in (
        ("u", state.u, grid.scalar_shape),
        ("v", state.v, grid.scalar_shape),
        ("u_faces", state.u_faces, grid.u_face_shape),
        ("v_faces", state.v_faces, grid.v_face_shape),
        ("pressure", state.pressure, grid.scalar_shape),
    ):
        _check_field(field, shape, name)

    u_from_faces = 0.5 * (state.u_faces[:, :-1] + state.u_faces[:, 1:])
    v_from_faces = 0.5 * (state.v_faces[:-1, :] + state.v_faces[1:, :])

    error_u = state.u - u_from_faces
    error_v = state.v - v_from_faces

    interior = np.zeros(grid.scalar_shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    boundary = ~interior

    u_total = _rms(error_u)
    v_total = _rms(error_v)

    max_u = float(np.max(np.abs(state.u)))
    max_v = float(np.max(np.abs(state.v)))

    u_relative = 100.0 * u_total / max_u if max_u > 0.0 else 0.0
    v_relative = 100.0 * v_total / max_v if max_v > 0.0 else 0.0

    return ConsistencyDiagnostics(
        u_total=u_total,
        u_interior=_rms(error_u[interior]),
        u_boundary=_rms(error_u[boundary]),
        v_total=v_total,
        v_interior=_rms(error_v[interior]),
        v_boundary=_rms(error_v[boundary]),
        u_relative_percent=u_relative,
        v_relative_percent=v_relative,
    )
