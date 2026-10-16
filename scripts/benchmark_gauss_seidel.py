"""Compara o tempo de execucao do Gauss-Seidel Python e Numba."""

from time import perf_counter

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.gauss_seidel import solve_poisson_gauss_seidel
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.poisson import assemble_poisson_coefficients


def main():
    for size in (8, 16, 32):
        grid = create_grid(SimulationConfig(nx=size, ny=size))
        coefficients = assemble_poisson_coefficients(grid)
        x, y = grid.cell_centers()

        rhs = np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        rhs -= np.mean(rhs)

        results = {}

        for backend in ("python", "numba"):
            # Aquecimento: exclui compilacao JIT da medicao.
            if backend == "numba":
                solve_poisson_gauss_seidel(
                    rhs,
                    coefficients,
                    backend=backend,
                    tolerance=1e-9,
                    max_iterations=10_000,
                )

            start = perf_counter()

            result = solve_poisson_gauss_seidel(
                rhs,
                coefficients,
                backend=backend,
                tolerance=1e-9,
                max_iterations=10_000,
            )

            elapsed = perf_counter() - start

            if not result.converged:
                raise RuntimeError(f"Poisson nao convergiu: {size}, {backend}")

            results[backend] = elapsed

            print(
                f"{size}x{size} | {backend:6s} | "
                f"{elapsed:.4f}s | "
                f"{result.iterations} iteracoes | "
                f"residuo={result.residual:.3e}"
            )

        speedup = results["python"] / results["numba"]
        print(f"Aceleracao: {speedup:.2f}x\n")


if __name__ == "__main__":
    main()
