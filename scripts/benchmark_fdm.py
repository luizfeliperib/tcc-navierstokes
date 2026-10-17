
"""FDM-07.2.1: benchmark de desempenho do solver FDM."""

import argparse
import csv
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.predictor import stable_timestep_limit
from navierstokes.fdm.solver import initialize_state, solve


def run_benchmark(
    sizes: list[int],
    steps: int,
    dt: float,
    reynolds: float,
    backend: str,
    output: Path,
) -> list[dict]:
    """Mede tempo e custo do solucionador em diferentes malhas."""
    if steps < 1:
        raise ValueError("steps deve ser positivo.")

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt deve ser positivo e finito.")

    # Aquecimento: exclui a compilação inicial do Numba
    # do tempo medido nos experimentos.
    if backend == "numba":
        print("Aquecendo Numba...")
        warmup_config = SimulationConfig(
            reynolds=reynolds,
            nx=8,
            ny=8,
            dt=dt,
            max_iterations=2,
        )
        start = perf_counter()
        solve(
            warmup_config,
            backend="numba",
            stop_on_steady=False,
        )
        print(
            f"Aquecimento: {perf_counter() - start:.2f} s"
        )

    rows = []

    for size in sizes:
        config = SimulationConfig(
            reynolds=reynolds,
            nx=size,
            ny=size,
            dt=dt,
            max_iterations=steps,
        )

        grid = create_grid(config)
        initial = initialize_state(grid)

        initial_dt_limit = stable_timestep_limit(
            initial.u_faces,
            initial.v_faces,
            grid,
            config.kinematic_viscosity,
        )

        if dt > initial_dt_limit:
            raise ValueError(
                f"Malha {size}x{size}: dt={dt} excede "
                f"o limite inicial {initial_dt_limit:.6g}"
            )

        print(f"Executando {size}x{size}...")

        start = perf_counter()

        result = solve(
            config,
            backend=backend,
            pressure_tolerance=1e-8,
            pressure_max_iterations=20000,
            stop_on_steady=False,
            record_interval=1,
        )

        elapsed = perf_counter() - start

        iterations = [
            record.poisson_iterations
            for record in result.history
        ]

        residuals = [
            record.poisson_residual
            for record in result.history
        ]

        final_record = result.history[-1]

        row = {
            "reynolds": reynolds,
            "nx": size,
            "ny": size,
            "cells": size * size,
            "dt": dt,
            "steps": result.steps,
            "final_time": result.final_time,
            "backend": backend,
            "elapsed_seconds": elapsed,
            "seconds_per_step": elapsed / result.steps,
            "poisson_iterations_mean": mean(iterations),
            "poisson_iterations_max": max(iterations),
            "poisson_residual_max": max(residuals),
            "steady_residual_final": result.steady_residual,
            "divergence_rms_final": (
                final_record.diagnostics.divergence_rms
            ),
            "initial_dt_limit": initial_dt_limit,
        }

        rows.append(row)

        print(
            f"  Tempo: {elapsed:.2f} s | "
            f"Tempo/passo: {elapsed / result.steps:.4f} s | "
            f"GS médio: {mean(iterations):.1f}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resultados salvos em: {output}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de desempenho FDM."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64],
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--reynolds", type=float, default=100)
    parser.add_argument(
        "--backend",
        choices=("python", "numba"),
        default="numba",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fdm_07_benchmark.csv"),
    )

    args = parser.parse_args()

    if not args.sizes or any(n < 2 for n in args.sizes):
        parser.error("Todas as malhas devem ter pelo menos 2 células.")

    run_benchmark(
        sizes=args.sizes,
        steps=args.steps,
        dt=args.dt,
        reynolds=args.reynolds,
        backend=args.backend,
        output=args.output,
    )


if __name__ == "__main__":
    main()
