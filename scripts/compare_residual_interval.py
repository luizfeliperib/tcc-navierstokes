"""FDM-07.2.4: compara intervalos no solver completo."""

import csv
import time
from pathlib import Path

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.operators import divergence_faces
from navierstokes.fdm.solver import solve


def rms(field):
    return float(np.sqrt(np.mean(field**2)))


def run_case(n, steps, interval):
    config = SimulationConfig(
        reynolds=100.0,
        nx=n,
        ny=n,
        dt=0.001,
        max_iterations=steps,
    )

    start = time.perf_counter()

    result = solve(
        config,
        backend="numba",
        pressure_tolerance=1e-9,
        pressure_max_iterations=20_000,
        residual_check_interval=interval,
        stop_on_steady=False,
        record_interval=1,
    )

    elapsed = time.perf_counter() - start

    divergence = divergence_faces(
        result.state.u_faces,
        result.state.v_faces,
        result.grid,
    )

    return result, {
        "mesh": f"{n}x{n}",
        "steps": steps,
        "interval": interval,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "mean_poisson_iterations": float(
            np.mean([r.poisson_iterations for r in result.history])
        ),
        "max_poisson_residual": float(max(r.poisson_residual for r in result.history)),
        "divergence_rms": rms(divergence),
    }


def main():
    # Aquecimento da compilação Numba, fora das medições.
    run_case(4, 2, 1)

    rows = []

    for n, steps in [(16, 30), (32, 20)]:
        reference, reference_row = run_case(n, steps, 1)
        reference_row.update(
            {
                "delta_u_rms": 0.0,
                "delta_v_rms": 0.0,
                "delta_pressure_rms": 0.0,
            }
        )
        rows.append(reference_row)

        print(f"\nMalha {n}x{n}, {steps} passos")
        print(f"k=1: {reference_row['elapsed_seconds']:.3f} s")

        for interval in (10, 20):
            result, row = run_case(n, steps, interval)

            row["delta_u_rms"] = rms(result.state.u - reference.state.u)
            row["delta_v_rms"] = rms(result.state.v - reference.state.v)
            row["delta_pressure_rms"] = rms(
                result.state.pressure - reference.state.pressure
            )

            rows.append(row)

            speedup = reference_row["elapsed_seconds"] / row["elapsed_seconds"]

            print(
                f"k={interval}: "
                f"{row['elapsed_seconds']:.3f} s | "
                f"speedup={speedup:.2f}x | "
                f"delta_u={row['delta_u_rms']:.3e} | "
                f"delta_v={row['delta_v_rms']:.3e} | "
                f"delta_p={row['delta_pressure_rms']:.3e}"
            )

    output = Path("results/fdm_07_full_solver_interval.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV salvo em: {output}")


if __name__ == "__main__":
    main()
