"""FDM-06.6: estudo exploratorio de refinamento de malha."""

import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.consistency import calculate_consistency
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.operators import divergence_faces
from navierstokes.fdm.time_step import FlowState, advance_one_step

REYNOLDS = 100.0
DT = 0.001
NUMBER_OF_STEPS = 1000
GRID_SIZES = (32,)

PRESSURE_TOLERANCE = 1e-9
PRESSURE_MAX_ITERATIONS = 10_000

OUTPUT_DIRECTORY = Path("results/fdm_06_grid_refinement_numba_32")


def rms(values: np.ndarray) -> float:
    """Calcula a raiz do erro quadratico medio."""
    return float(np.sqrt(np.mean(values**2)))


def initialize_state(grid) -> FlowState:
    """Inicializa o fluido em repouso."""
    return FlowState(
        u=np.zeros(grid.scalar_shape),
        v=np.zeros(grid.scalar_shape),
        u_faces=np.zeros(grid.u_face_shape),
        v_faces=np.zeros(grid.v_face_shape),
        pressure=np.zeros(grid.scalar_shape),
    )


def run_case(size: int) -> dict:
    """Executa um caso e salva seu historico em CSV."""

    start = perf_counter()
    config = SimulationConfig(
        nx=size,
        ny=size,
        reynolds=REYNOLDS,
        dt=DT,
    )

    grid = create_grid(config)
    state = initialize_state(grid)

    output_path = OUTPUT_DIRECTORY / f"grid_{size}x{size}.csv"

    fieldnames = [
        "step",
        "time",
        "nx",
        "ny",
        "dx",
        "dy",
        "divergence_rms",
        "u_total",
        "u_interior",
        "u_boundary",
        "v_total",
        "v_interior",
        "v_boundary",
        "u_relative_percent",
        "v_relative_percent",
        "max_abs_u",
        "max_abs_v",
        "kinetic_energy_mean",
        "poisson_iterations",
        "poisson_residual",
    ]

    last_row = None

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(1, NUMBER_OF_STEPS + 1):
            result = advance_one_step(
                state,
                grid,
                config,
                pressure_tolerance=PRESSURE_TOLERANCE,
                pressure_max_iterations=PRESSURE_MAX_ITERATIONS,
                backend="numba",
            )

            state = result.state

            consistency = calculate_consistency(state, grid)

            divergence = divergence_faces(
                state.u_faces,
                state.v_faces,
                grid,
            )

            row = {
                "step": step,
                "time": step * DT,
                "nx": grid.nx,
                "ny": grid.ny,
                "dx": grid.dx,
                "dy": grid.dy,
                "divergence_rms": rms(divergence),
                "u_total": consistency.u_total,
                "u_interior": consistency.u_interior,
                "u_boundary": consistency.u_boundary,
                "v_total": consistency.v_total,
                "v_interior": consistency.v_interior,
                "v_boundary": consistency.v_boundary,
                "u_relative_percent": (consistency.u_relative_percent),
                "v_relative_percent": (consistency.v_relative_percent),
                "max_abs_u": float(np.max(np.abs(state.u))),
                "max_abs_v": float(np.max(np.abs(state.v))),
                "kinetic_energy_mean": float(0.5 * np.mean(state.u**2 + state.v**2)),
                "poisson_iterations": (result.pressure_result.iterations),
                "poisson_residual": (result.pressure_result.residual),
            }

            writer.writerow(row)
            last_row = row

            if step % 100 == 0:
                print(
                    f"Malha {size}x{size}: {step}/{NUMBER_OF_STEPS} passos",
                    file=sys.stderr,
                )
    elapsed = perf_counter() - start
    print(f"Tempo total: {elapsed:.2f} segundos")
    return last_row


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    summary = []

    for size in GRID_SIZES:
        print(
            f"Iniciando malha {size}x{size}",
            file=sys.stderr,
        )

        final_result = run_case(size)
        summary.append(final_result)

        print(
            f"Concluida malha {size}x{size}",
            file=sys.stderr,
        )

    summary_path = OUTPUT_DIRECTORY / "summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)

    print(
        f"Resultados salvos em: {OUTPUT_DIRECTORY}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
