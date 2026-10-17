"""FDM-07.1: execução reproduzível e exportação de resultados."""

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import subprocess
import sys

from navierstokes.config import SimulationConfig
from navierstokes.fdm.solver import solve


def export_results(result, config, output_dir, elapsed, settings):
    """Salva metadados, histórico e campos finais."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Coordenadas 1D: os campos centrais têm shape (ny, nx).
    grid = result.grid
    state = result.state

    np.savez_compressed(
        output_dir / "fields.npz",
        x=grid.x,
        y=grid.y,
        u=state.u,
        v=state.v,
        pressure=state.pressure,
        u_faces=state.u_faces,
        v_faces=state.v_faces,
    )

    fieldnames = [
        "step",
        "time",
        "steady_residual",
        "poisson_residual",
        "poisson_iterations",
        "divergence_rms",
        "center_face_u_rms",
        "center_face_v_rms",
        "max_abs_u",
        "max_abs_v",
        "kinetic_energy_mean",
    ]

    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in result.history:
            writer.writerow(
                {
                    "step": record.step,
                    "time": record.time,
                    "steady_residual": record.steady_residual,
                    "poisson_residual": record.poisson_residual,
                    "poisson_iterations": record.poisson_iterations,
                    **asdict(record.diagnostics),
                }
            )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "problem": "lid_driven_cavity",
        "method": "FDM",
        "configuration": asdict(config),
        "kinematic_viscosity": config.kinematic_viscosity,
        "execution_settings": settings,
        "result": {
            "steps": result.steps,
            "final_time": result.final_time,
            "converged": result.converged,
            "stop_reason": result.stop_reason,
            "steady_residual": result.steady_residual,
            "elapsed_seconds": elapsed,
        },
        "grid": {
            "nx": grid.nx,
            "ny": grid.ny,
            "dx": grid.dx,
            "dy": grid.dy,
        },
    }

    with (output_dir / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Executa e exporta uma simulação FDM da cavidade."
    )
    parser.add_argument("--reynolds", type=float, default=100.0)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--steady-tolerance", type=float, default=1e-5)
    parser.add_argument("--pressure-tolerance", type=float, default=1e-9)
    parser.add_argument("--pressure-max-iterations", type=int, default=10000)
    parser.add_argument("--record-interval", type=int, default=10)
    parser.add_argument(
        "--backend", choices=("python", "numba"), default="python"
    )
    parser.add_argument("--stop-on-steady", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/fdm_07_smoke")
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Gera figuras exploratórias após a simulação.",
    )

    args = parser.parse_args()

    config = SimulationConfig(
        reynolds=args.reynolds,
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        tolerance=args.steady_tolerance,
        max_iterations=args.steps,
    )

    settings = {
        "backend": args.backend,
        "pressure_tolerance": args.pressure_tolerance,
        "pressure_max_iterations": args.pressure_max_iterations,
        "record_interval": args.record_interval,
        "stop_on_steady": args.stop_on_steady,
        "steady_consecutive_steps": 20,
        "min_steps": 100,
    }

    start = perf_counter()

    result = solve(
        config,
        backend=args.backend,
        pressure_tolerance=args.pressure_tolerance,
        pressure_max_iterations=args.pressure_max_iterations,
        record_interval=args.record_interval,
        stop_on_steady=args.stop_on_steady,
    )

    elapsed = perf_counter() - start

    export_results(
        result, config, args.output, elapsed, settings
    )

    if args.plot:
        plot_script = Path(__file__).with_name(
            "plot_fdm_validation.py"
        )

        subprocess.run(
            [
                sys.executable,
                str(plot_script),
                "--input",
                str(args.output),
            ],
            check=True,
        )

    print(f"Resultados salvos em: {args.output}")
    print(f"Passos executados: {result.steps}")
    print(f"Tempo físico final: {result.final_time:.6f}")
    print(f"Estacionário: {result.converged}")
    print(f"Motivo da parada: {result.stop_reason}")
    print(f"Resíduo estacionário: {result.steady_residual:.6e}")
    print(f"Tempo computacional: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
