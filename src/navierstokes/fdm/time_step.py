"""FDM-06.5: integração de um passo temporal da cavidade."""

from dataclasses import dataclass
from typing import Literal

from numpy.typing import NDArray

from navierstokes.config import SimulationConfig
from navierstokes.fdm.gauss_seidel import GaussSeidelResult
from navierstokes.fdm.grid import CartesianGrid
from navierstokes.fdm.poisson import PoissonCoefficients
from navierstokes.fdm.predictor import (
    _check_field,
    _check_geometry,
    velocity_predictor,
)
from navierstokes.fdm.projection import (
    correct_center_velocities,
    correct_face_fluxes,
    predict_face_fluxes,
    solve_projection_pressure,
)


@dataclass(frozen=True, slots=True)
class FlowState:
    """Estado discreto no instante n."""

    u: NDArray
    v: NDArray
    u_faces: NDArray
    v_faces: NDArray
    pressure: NDArray


@dataclass(frozen=True, slots=True)
class TimeStepResult:
    """Estado atualizado e diagnóstico do solucionador de pressão."""

    state: FlowState
    pressure_result: GaussSeidelResult


def advance_one_step(
    state: FlowState,
    grid: CartesianGrid,
    config: SimulationConfig,
    *,
    pressure_tolerance: float = 1e-8,
    pressure_max_iterations: int = 10_000,
    backend: Literal["python", "numba"] = "python",
    residual_check_interval: int = 1,
    coefficients: PoissonCoefficients | None = None,
    omega: float = 1.0,
) -> TimeStepResult:
    """Executa um passo completo do método de projeção.

    Usa a pressão anterior como estimativa inicial do Poisson.
    Não modifica o estado recebido.
    """
    _check_geometry(grid, config)

    for name, field, shape in (
        ("u", state.u, grid.scalar_shape),
        ("v", state.v, grid.scalar_shape),
        ("u_faces", state.u_faces, grid.u_face_shape),
        ("v_faces", state.v_faces, grid.v_face_shape),
        ("pressure", state.pressure, grid.scalar_shape),
    ):
        _check_field(field, shape, name)

    # 1. Predição das velocidades nos centros.
    u_star, v_star = velocity_predictor(
        state.u,
        state.v,
        state.u_faces,
        state.v_faces,
        grid,
        config,
    )

    # 2. Construção incremental dos fluxos provisórios.
    uf_star, vf_star = predict_face_fluxes(
        state.u,
        state.v,
        u_star,
        v_star,
        state.u_faces,
        state.v_faces,
        grid,
    )

    # 3. Resolução da equação de Poisson.
    pressure_result = solve_projection_pressure(
        uf_star,
        vf_star,
        grid,
        density=config.density,
        dt=config.dt,
        tolerance=pressure_tolerance,
        max_iterations=pressure_max_iterations,
        initial_pressure=state.pressure,
        backend=backend,
        residual_check_interval=residual_check_interval,
        coefficients=coefficients,
        omega=omega,
    )

    pressure = pressure_result.phi

    # 4. Projeção dos fluxos nas faces.
    uf_new, vf_new = correct_face_fluxes(
        uf_star,
        vf_star,
        pressure,
        grid,
        density=config.density,
        dt=config.dt,
    )

    # 5. Correção das velocidades centrais.
    u_new, v_new = correct_center_velocities(
        u_star,
        v_star,
        pressure,
        grid,
        density=config.density,
        dt=config.dt,
    )

    new_state = FlowState(
        u=u_new,
        v=v_new,
        u_faces=uf_new,
        v_faces=vf_new,
        pressure=pressure.copy(),
    )

    return TimeStepResult(
        state=new_state,
        pressure_result=pressure_result,
    )
