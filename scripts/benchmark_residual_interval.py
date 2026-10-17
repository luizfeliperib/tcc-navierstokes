"""Benchmark de frequência de cálculo do resíduo."""

import csv
import time
from pathlib import Path

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.gauss_seidel import solve_poisson_gauss_seidel
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.poisson import (
    apply_poisson_system,
    assemble_poisson_coefficients,
)


def run(n, interval):
    grid = create_grid(SimulationConfig(nx=n, ny=n))
    coeff = assemble_poisson_coefficients(grid)

    x, y = grid.cell_centers()
    phi_ref = np.cos(np.pi * x) * np.cos(np.pi * y)
    phi_ref -= np.mean(phi_ref)
    rhs = apply_poisson_system(phi_ref, coeff)

    start = time.perf_counter()

    result = solve_poisson_gauss_seidel(
        rhs,
        coeff,
        tolerance=1e-9,
        max_iterations=50000,
        backend="numba",
        residual_check_interval=interval,
    )

    elapsed = time.perf_counter() - start

    true_residual = np.linalg.norm(
        rhs - apply_poisson_system(result.phi, coeff)
    ) / np.sqrt(rhs.size)

    return {
        "mesh": f"{n}x{n}",
        "interval": interval,
        "converged": result.converged,
        "iterations": result.iterations,
        "residual_checks": len(result.residual_history),
        "elapsed_seconds": elapsed,
        "true_residual": true_residual,
        "relative_error": (
            np.linalg.norm(result.phi - phi_ref) / np.linalg.norm(phi_ref)
        ),
    }


def main():
    # Aquecimento da compilação Numba.
    run(4, 1)

    rows = []

    for n in (16, 32, 64):
        for interval in (1, 10, 20):
            result = run(n, interval)
            rows.append(result)

            print(
                f"{result['mesh']} | "
                f"k={interval:2d} | "
                f"tempo={result['elapsed_seconds']:.4f}s | "
                f"iterações={result['iterations']} | "
                f"resíduo={result['true_residual']:.3e}"
            )

    output = Path("results/fdm_07_residual_interval.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResultados salvos em: {output}")

    if not all(row["converged"] for row in rows):
        raise SystemExit("Algum experimento não convergiu.")


if __name__ == "__main__":
    main()
