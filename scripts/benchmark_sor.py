"""FDM-07.5.4: benchmark do fator de relaxacao SOR."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.gauss_seidel import solve_poisson_gauss_seidel
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.poisson import (
    apply_poisson_system,
    assemble_poisson_coefficients,
)


def make_problem(n):
    """Problema discreto com solucao conhecida e media zero."""
    grid = create_grid(SimulationConfig(nx=n, ny=n))
    coefficients = assemble_poisson_coefficients(grid)

    x, y = grid.cell_centers()

    phi_reference = np.cos(np.pi * x) * np.cos(np.pi * y)
    phi_reference -= np.mean(phi_reference)

    rhs = apply_poisson_system(phi_reference, coefficients)

    return coefficients, phi_reference, rhs


def run_case(
    rhs,
    coefficients,
    phi_reference,
    *,
    omega,
    tolerance,
    max_iterations,
    residual_check_interval,
):
    """Executa e verifica uma resolucao de Poisson."""

    start = perf_counter()

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        backend="numba",
        omega=omega,
        tolerance=tolerance,
        max_iterations=max_iterations,
        residual_check_interval=residual_check_interval,
    )

    elapsed = perf_counter() - start

    residual_field = rhs - apply_poisson_system(result.phi, coefficients)

    true_residual = float(np.linalg.norm(residual_field) / np.sqrt(rhs.size))

    error_rms = float(np.sqrt(np.mean((result.phi - phi_reference) ** 2)))

    return {
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "elapsed_seconds": elapsed,
        "reported_residual": float(result.residual),
        "true_residual": true_residual,
        "error_rms": error_rms,
        "residual_checks": len(result.residual_history) - 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gauss-Seidel versus SOR.")

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[16, 32, 48],
    )
    parser.add_argument(
        "--omegas",
        type=float,
        nargs="+",
        default=[1.0, 1.3, 1.5, 1.7, 1.8, 1.9, 1.95],
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--residual-check-interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fdm_075_sor_benchmark"),
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats deve ser >= 1")

    if any(n < 2 for n in args.sizes):
        parser.error("Todas as malhas devem ter n >= 2")

    if any(not np.isfinite(w) or not 0 < w < 2 for w in args.omegas):
        parser.error("Todos os omegas devem estar em (0, 2)")

    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance deve ser positiva e finita")

    if args.max_iterations < 1:
        parser.error("--max-iterations deve ser >= 1")

    if args.residual_check_interval < 1:
        parser.error("--residual-check-interval deve ser >= 1")

    args.output.mkdir(parents=True, exist_ok=True)

    # Compilacao inicial, excluida das medicoes.
    warm_coeff, warm_phi, warm_rhs = make_problem(4)
    run_case(
        warm_rhs,
        warm_coeff,
        warm_phi,
        omega=1.0,
        tolerance=1e-9,
        max_iterations=100,
        residual_check_interval=10,
    )

    measurements = []
    summaries = []

    print("\nFDM-07.5.4 | Benchmark SOR\n", flush=True)

    for n in args.sizes:
        coefficients, phi_ref, rhs = make_problem(n)

        # Alternar a ordem de execucao entre repeticoes
        # reduz vieses sistematicos de aquecimento.
        cases = {omega: [] for omega in args.omegas}

        # Aquecimento especifico por configuracao.
        for omega in args.omegas:
            run_case(
                rhs,
                coefficients,
                phi_ref,
                omega=omega,
                tolerance=args.tolerance,
                max_iterations=args.max_iterations,
                residual_check_interval=(args.residual_check_interval),
            )

        for repeat in range(1, args.repeats + 1):
            order = list(args.omegas) if repeat % 2 else list(reversed(args.omegas))

            for omega in order:
                result = run_case(
                    rhs,
                    coefficients,
                    phi_ref,
                    omega=omega,
                    tolerance=args.tolerance,
                    max_iterations=args.max_iterations,
                    residual_check_interval=(args.residual_check_interval),
                )

                row = {
                    "mesh": f"{n}x{n}",
                    "n": n,
                    "omega": omega,
                    "repeat": repeat,
                    **result,
                }

                measurements.append(row)
                cases[omega].append(row)

        for omega in args.omegas:
            rows = cases[omega]

            summary = {
                "mesh": f"{n}x{n}",
                "n": n,
                "omega": omega,
                "all_converged": all(r["converged"] for r in rows),
                "iterations_median": median(r["iterations"] for r in rows),
                "time_median_seconds": median(r["elapsed_seconds"] for r in rows),
                "time_min_seconds": min(r["elapsed_seconds"] for r in rows),
                "time_max_seconds": max(r["elapsed_seconds"] for r in rows),
                "true_residual_max": max(r["true_residual"] for r in rows),
                "error_rms_max": max(r["error_rms"] for r in rows),
            }

            summaries.append(summary)

            print(
                f"{n:2d}x{n:<2d} | "
                f"omega={omega:4.2f} | "
                f"iter={summary['iterations_median']:6.0f} | "
                f"tempo={summary['time_median_seconds']:.5f}s | "
                f"res={summary['true_residual_max']:.2e} | "
                f"ok={summary['all_converged']}",
                flush=True,
            )

        print(flush=True)

    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        args.output / "measurements.csv",
        measurements,
    )
    write_csv(
        args.output / "summary.csv",
        summaries,
    )

    metadata = {
        "experiment": "FDM-07.5.4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "problem": "discrete_poisson_known_solution",
        "boundary_condition": "homogeneous_neumann",
        "pressure_reference": "zero_mean",
        "backend": "numba",
        "sizes": args.sizes,
        "omegas": args.omegas,
        "repeats": args.repeats,
        "tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
        "residual_check_interval": (args.residual_check_interval),
        "timing": (
            "perf_counter; compilation warmup excluded; median of repeated solves"
        ),
    }

    with (args.output / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(
        f"Arquivos salvos em: {args.output}",
        flush=True,
    )

    if not all(r["all_converged"] for r in summaries):
        raise SystemExit("Benchmark concluido com casos nao convergidos.")


if __name__ == "__main__":
    main()
