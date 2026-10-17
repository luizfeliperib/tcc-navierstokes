"""FDM-07.2.2 — Verificação isolada do solucionador de Poisson."""

import argparse
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


def run_case(n: int, tolerance: float, max_iterations: int) -> dict:
    config = SimulationConfig(nx=n, ny=n)
    grid = create_grid(config)
    coefficients = assemble_poisson_coefficients(grid)

    # Solução discreta conhecida, suave e não constante.
    x, y = np.meshgrid(grid.x, grid.y)
    phi_ref = np.cos(np.pi * x) * np.cos(np.pi * y)
    phi_ref -= np.mean(phi_ref)

    # Construção exata do problema discreto.
    rhs = apply_poisson_system(phi_ref, coefficients)

    start = time.perf_counter()
    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=tolerance,
        max_iterations=max_iterations,
        backend="numba",
    )
    elapsed = time.perf_counter() - start

    error = result.phi - phi_ref
    relative_error = np.linalg.norm(error) / np.linalg.norm(phi_ref)
    max_error = np.max(np.abs(error))

    true_residual = rhs - apply_poisson_system(result.phi, coefficients)
    residual_rms = np.linalg.norm(true_residual) / np.sqrt(rhs.size)

    return {
        "mesh": f"{n}x{n}",
        "n": n,
        "converged": result.converged,
        "iterations": result.iterations,
        "elapsed_seconds": elapsed,
        "initial_residual": result.residual_history[0],
        "final_residual": residual_rms,
        "relative_error": relative_error,
        "max_error": max_error,
        "pressure_mean": np.mean(result.phi),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--max-iterations", type=int, default=50000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fdm_07_poisson_verification.csv"),
    )
    args = parser.parse_args()

    # Aquecimento da compilação Numba.
    run_case(4, 1e-6, 1000)

    rows = []
    for n in args.sizes:
        row = run_case(n, args.tolerance, args.max_iterations)
        rows.append(row)

        print(
            f"{row['mesh']}: "
            f"converged={row['converged']}, "
            f"iterations={row['iterations']}, "
            f"residual={row['final_residual']:.3e}, "
            f"relative_error={row['relative_error']:.3e}, "
            f"time={row['elapsed_seconds']:.2f}s"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResultados salvos em: {args.output}")

    if not all(row["converged"] for row in rows):
        raise SystemExit("Uma ou mais malhas não convergiram.")


if __name__ == "__main__":
    main()
