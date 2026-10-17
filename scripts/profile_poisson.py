
"""FDM-07.2.3 — Perfil de desempenho do Gauss-Seidel.

Reproduz o ciclo iterativo do backend Numba sem modificar
a implementação original.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.poisson import (
    apply_poisson_system,
    assemble_poisson_coefficients,
)
from navierstokes.fdm.gauss_seidel_numba import gauss_seidel_sweep
from navierstokes.fdm.gauss_seidel import solve_poisson_gauss_seidel


def profile_case(n, tolerance, max_iterations):
    grid = create_grid(SimulationConfig(nx=n, ny=n))

    start = time.perf_counter()
    coefficients = assemble_poisson_coefficients(grid)
    assembly_time = time.perf_counter() - start

    x, y = np.meshgrid(grid.x, grid.y)
    phi_ref = np.cos(np.pi * x) * np.cos(np.pi * y)
    phi_ref -= np.mean(phi_ref)

    rhs = apply_poisson_system(phi_ref, coefficients)
    phi = np.zeros_like(rhs)

    sweep_time = 0.0
    normalization_time = 0.0
    residual_time = 0.0
    iterations = 0
    residual = float("inf")

    total_start = time.perf_counter()

    for iteration in range(1, max_iterations + 1):
        start = time.perf_counter()

        gauss_seidel_sweep(
            phi,
            rhs,
            coefficients.center,
            coefficients.east,
            coefficients.west,
            coefficients.north,
            coefficients.south,
        )

        sweep_time += time.perf_counter() - start

        start = time.perf_counter()
        phi -= np.mean(phi)
        normalization_time += time.perf_counter() - start

        start = time.perf_counter()
        residual_field = rhs - apply_poisson_system(
            phi, coefficients
        )
        residual = float(
            np.linalg.norm(residual_field) / np.sqrt(rhs.size)
        )
        residual_time += time.perf_counter() - start

        iterations = iteration

        if not np.isfinite(residual):
            raise FloatingPointError("Resíduo não finito")

        if residual <= tolerance:
            break

    total_time = time.perf_counter() - total_start
    measured = sweep_time + normalization_time + residual_time
    overhead_time = max(0.0, total_time - measured)

    # Confirmação independente com o solver original.
    reference = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=tolerance,
        max_iterations=max_iterations,
        backend="numba",
    )

    if (
        reference.converged != (residual <= tolerance)
        or reference.iterations != iterations
        or not np.allclose(
            reference.phi, phi, atol=1e-12, rtol=0.0
        )
    ):
        raise AssertionError(
            "O perfil divergiu do solver original."
        )

    return {
        "mesh": f"{n}x{n}",
        "iterations": iterations,
        "converged": residual <= tolerance,
        "residual": residual,
        "assembly_seconds": assembly_time,
        "total_seconds": total_time,
        "sweep_seconds": sweep_time,
        "normalization_seconds": normalization_time,
        "residual_seconds": residual_time,
        "overhead_seconds": overhead_time,
        "sweep_percent": 100 * sweep_time / total_time,
        "normalization_percent":
            100 * normalization_time / total_time,
        "residual_percent": 100 * residual_time / total_time,
        "overhead_percent": 100 * overhead_time / total_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[16, 32, 64]
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-9
    )
    parser.add_argument(
        "--max-iterations", type=int, default=50000
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fdm_07_poisson_profile.csv"),
    )
    args = parser.parse_args()

    # Aquecimento Numba, excluído das medições.
    profile_case(4, 1e-6, 1000)

    rows = []

    for n in args.sizes:
        result = profile_case(
            n, args.tolerance, args.max_iterations
        )
        rows.append(result)

        print(f"\nMalha: {result['mesh']}")
        print(f"Iterações: {result['iterations']}")
        print(f"Convergiu: {result['converged']}")
        print(f"Tempo total: {result['total_seconds']:.4f} s")
        print(
            f"Varredura GS: {result['sweep_percent']:.1f}%"
        )
        print(
            "Normalização: "
            f"{result['normalization_percent']:.1f}%"
        )
        print(
            f"Resíduo: {result['residual_percent']:.1f}%"
        )
        print(
            f"Overhead: {result['overhead_percent']:.1f}%"
        )
        print(
            "Montagem de Poisson: "
            f"{result['assembly_seconds']:.6f} s"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV salvo em: {args.output}")

    if not all(row["converged"] for row in rows):
        raise SystemExit("Uma ou mais malhas não convergiram.")


if __name__ == "__main__":
    main()
