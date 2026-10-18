"""FDM-06.7: controle da simulacao temporal da cavidade."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.diagnostics import (
    FlowDiagnostics,
    calculate_diagnostics,
)
from navierstokes.fdm.grid import CartesianGrid, create_grid
from navierstokes.fdm.poisson import assemble_poisson_coefficients
from navierstokes.fdm.time_step import FlowState, advance_one_step


@dataclass(frozen=True, slots=True)
class SimulationRecord:
    """Diagnosticos registrados em um instante da simulacao."""

    step: int
    time: float
    steady_residual: float
    poisson_residual: float
    poisson_iterations: int
    diagnostics: FlowDiagnostics


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Resultado de uma simulacao completa."""

    grid: CartesianGrid
    state: FlowState
    steps: int
    final_time: float
    converged: bool
    stop_reason: str
    steady_residual: float
    history: tuple[SimulationRecord, ...]


def initialize_state(grid: CartesianGrid) -> FlowState:
    """Inicializa o fluido em repouso."""
    return FlowState(
        u=grid.zeros_scalar(),
        v=grid.zeros_scalar(),
        u_faces=grid.zeros_u_faces(),
        v_faces=grid.zeros_v_faces(),
        pressure=grid.zeros_scalar(),
    )


def _rms(field: np.ndarray) -> float:
    return float(np.sqrt(np.mean(field**2)))


def _steady_residual(
    previous: FlowState,
    current: FlowState,
    dt: float,
) -> float:
    """Maior taxa RMS de variacao entre as componentes."""
    ru = _rms(current.u - previous.u) / dt
    rv = _rms(current.v - previous.v) / dt
    return max(ru, rv)


def solve(
    config: SimulationConfig,
    *,
    backend: Literal["python", "numba"] = "python",
    face_flux_mode: Literal["incremental", "reconstructed"] = "incremental",
    pressure_tolerance: float = 1e-9,
    pressure_max_iterations: int = 10_000,
    stop_on_steady: bool = True,
    steady_consecutive_steps: int = 20,
    min_steps: int = 100,
    record_interval: int = 10,
    residual_check_interval: int = 1,
    omega: float = 1.0,
) -> SimulationResult:
    """Executa a simulacao com passo temporal constante.

    config.max_iterations controla o numero maximo de passos.
    config.tolerance controla a parada estacionaria.
    """

    if (
        isinstance(config.max_iterations, (bool, np.bool_))
        or not isinstance(config.max_iterations, (int, np.integer))
        or config.max_iterations < 1
    ):
        raise ValueError("max_iterations deve ser inteiro positivo")

    for name, value, minimum in (
        ("steady_consecutive_steps", steady_consecutive_steps, 1),
        ("min_steps", min_steps, 0),
        ("record_interval", record_interval, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} deve ser inteiro >= {minimum}")

    if not np.isfinite(config.dt) or config.dt <= 0:
        raise ValueError("dt deve ser finito e positivo")

    if not np.isfinite(config.tolerance) or config.tolerance <= 0:
        raise ValueError("tolerance deve ser finita e positiva")

    if backend not in ("python", "numba"):
        raise ValueError("backend invalido")

    if (
        isinstance(omega, (bool, np.bool_))
        or not isinstance(
            omega,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(omega)
        or not (0.0 < omega < 2.0)
    ):
        raise ValueError("omega deve satisfazer 0 < omega < 2")

    if face_flux_mode not in ("incremental", "reconstructed"):
        raise ValueError("face_flux_mode deve ser 'incremental' ou 'reconstructed'")

    grid = create_grid(config)
    state = initialize_state(grid)

    # A matriz discreta permanece constante durante a simulação.
    coefficients = assemble_poisson_coefficients(grid)

    history: list[SimulationRecord] = []
    consecutive = 0
    converged = False
    steady_residual = float("inf")
    completed_steps = 0

    for step in range(1, config.max_iterations + 1):
        previous = state

        step_result = advance_one_step(
            state,
            grid,
            config,
            pressure_tolerance=pressure_tolerance,
            pressure_max_iterations=pressure_max_iterations,
            backend=backend,
            residual_check_interval=residual_check_interval,
            coefficients=coefficients,
            omega=omega,
            face_flux_mode=face_flux_mode,
        )

        state = step_result.state
        completed_steps = step

        steady_residual = _steady_residual(previous, state, config.dt)

        if not np.isfinite(steady_residual):
            raise FloatingPointError("Indicador estacionario nao finito")

        if steady_residual <= config.tolerance:
            consecutive += 1
        else:
            consecutive = 0

        reached_steady = (
            stop_on_steady
            and step >= min_steps
            and consecutive >= steady_consecutive_steps
        )

        should_record = (
            step == 1
            or step % record_interval == 0
            or step == config.max_iterations
            or reached_steady
        )

        if should_record:
            history.append(
                SimulationRecord(
                    step=step,
                    time=step * config.dt,
                    steady_residual=steady_residual,
                    poisson_residual=(step_result.pressure_result.residual),
                    poisson_iterations=(step_result.pressure_result.iterations),
                    diagnostics=calculate_diagnostics(state, grid),
                )
            )

        if reached_steady:
            converged = True
            break

    return SimulationResult(
        grid=grid,
        state=state,
        steps=completed_steps,
        final_time=completed_steps * config.dt,
        converged=converged,
        stop_reason=("steady_tolerance" if converged else "max_steps"),
        steady_residual=steady_residual,
        history=tuple(history),
    )
