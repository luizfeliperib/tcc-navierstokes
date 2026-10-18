"""FDM-07.2.4 — Benchmark de reutilização dos coeficientes de Poisson."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.diagnostics import calculate_diagnostics
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.poisson import assemble_poisson_coefficients
from navierstokes.fdm.solver import initialize_state
from navierstokes.fdm.time_step import advance_one_step


def rms(array):
    return float(np.sqrt(np.mean(np.asarray(array) ** 2)))


def run_case(n, steps, reuse):
    config = SimulationConfig(
        reynolds=100.0,
        nx=n,
        ny=n,
        dt=0.001,
        max_iterations=steps,
    )

    grid = create_grid(config)
    state = initialize_state(grid)

    # Montagem inicial fica dentro da medição.
    start = time.perf_counter()

    coefficients = assemble_poisson_coefficients(grid) if reuse else None

    iterations = []
    residuals = []

    for _ in range(steps):
        result = advance_one_step(
            state,
            grid,
            config,
            backend="numba",
            pressure_tolerance=1e-9,
            pressure_max_iterations=20000,
            residual_check_interval=20,
            coefficients=coefficients,
        )

        state = result.state
        iterations.append(result.pressure_result.iterations)
        residuals.append(result.pressure_result.residual)

    elapsed = time.perf_counter() - start
    diagnostics = calculate_diagnostics(state, grid)

    return state, {
        "mesh": f"{n}x{n}",
        "steps": steps,
        "reuse": reuse,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "mean_poisson_iterations": float(np.mean(iterations)),
        "max_poisson_residual": float(max(residuals)),
        "divergence_rms": diagnostics.divergence_rms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fdm_07_poisson_reuse.csv"),
    )
    args = parser.parse_args()

    # Compila Numba antes das medições.
    run_case(4, 2, False)

    rows = []

    for n in args.sizes:
        baseline_state, baseline = run_case(n, args.steps, False)
        optimized_state, optimized = run_case(n, args.steps, True)

        baseline.update(
            {
                "delta_u_rms": 0.0,
                "delta_v_rms": 0.0,
                "delta_pressure_rms": 0.0,
            }
        )

        optimized.update(
            {
                "delta_u_rms": rms(optimized_state.u - baseline_state.u),
                "delta_v_rms": rms(optimized_state.v - baseline_state.v),
                "delta_pressure_rms": rms(
                    optimized_state.pressure - baseline_state.pressure
                ),
            }
        )

        rows.extend([baseline, optimized])

        speedup = baseline["elapsed_seconds"] / optimized["elapsed_seconds"]

        print(f"\nMalha {n}x{n}")
        print(f"Sem reutilização: {baseline['elapsed_seconds']:.4f} s")
        print(f"Com reutilização: {optimized['elapsed_seconds']:.4f} s")
        print(f"Aceleração: {speedup:.3f}x")
        print(f"Diferença RMS de u: {optimized['delta_u_rms']:.3e}")
        print(f"Diferença RMS de v: {optimized['delta_v_rms']:.3e}")
        print(f"Diferença RMS de p: {optimized['delta_pressure_rms']:.3e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV salvo em: {args.output}")


if __name__ == "__main__":
    main()
