
"""FDM-07.2.7 — Sondagem de convergência estacionária em Re=100."""

import argparse
import csv
import json
import time
from pathlib import Path

from navierstokes.config import SimulationConfig
from navierstokes.fdm.solver import solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fdm_07_steady_probe"),
    )
    args = parser.parse_args()

    config = SimulationConfig(
        reynolds=100.0,
        nx=args.size,
        ny=args.size,
        dt=args.dt,
        tolerance=1e-5,
        max_iterations=args.steps,
    )

    print(
        f"Iniciando: Re=100, malha={args.size}x{args.size}, "
        f"dt={args.dt}, máximo={args.steps} passos",
        flush=True,
    )

    start = time.perf_counter()

    result = solve(
        config,
        backend="numba",
        pressure_tolerance=1e-9,
        pressure_max_iterations=20000,
        residual_check_interval=20,
        stop_on_steady=True,
        steady_consecutive_steps=20,
        min_steps=100,
        record_interval=20,
    )

    elapsed = time.perf_counter() - start

    args.output.mkdir(parents=True, exist_ok=True)

    history_path = args.output / "history.csv"

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "step",
            "time",
            "steady_residual",
            "poisson_residual",
            "poisson_iterations",
            "divergence_rms",
            "kinetic_energy_mean",
            "max_abs_u",
            "max_abs_v",
        ])

        for record in result.history:
            diagnostics = record.diagnostics
            writer.writerow([
                record.step,
                record.time,
                record.steady_residual,
                record.poisson_residual,
                record.poisson_iterations,
                diagnostics.divergence_rms,
                diagnostics.kinetic_energy_mean,
                diagnostics.max_abs_u,
                diagnostics.max_abs_v,
            ])

    summary = {
        "reynolds": 100.0,
        "nx": args.size,
        "ny": args.size,
        "dt": args.dt,
        "steps": result.steps,
        "final_time": result.final_time,
        "converged": result.converged,
        "stop_reason": result.stop_reason,
        "steady_residual": result.steady_residual,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / result.steps,
        "pressure_tolerance": 1e-9,
        "residual_check_interval": 20,
    }

    with (args.output / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nResultado:")
    print(json.dumps(summary, indent=2))
    print(f"\nHistórico: {history_path}")


if __name__ == "__main__":
    main()
