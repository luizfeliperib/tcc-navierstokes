"""Verificacao do solver de Poisson por solucao manufaturada.

Problema:
    -laplaciano(phi) = f

Condicoes de contorno:
    derivada normal de phi = 0

Solucao analitica:
    phi = cos(pi*x/Lx) * cos(pi*y/Ly)

A discretizacao utiliza centros de celulas e A = -L_h.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from navierstokes.config import SimulationConfig
from navierstokes.fdm.gauss_seidel import solve_poisson_gauss_seidel
from navierstokes.fdm.grid import CartesianGrid, create_grid
from navierstokes.fdm.poisson import (
    apply_poisson_system,
    assemble_poisson_coefficients,
)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    nx: int
    ny: int
    dx: float
    dy: float
    l2_error: float
    linf_error: float
    residual: float
    iterations: int
    converged: bool
    rhs_mean: float
    algebraic_error: float


def manufactured_solution(
    grid: CartesianGrid,
) -> NDArray[np.float64]:
    """Avalia a solucao analitica nos centros das celulas."""

    x, y = grid.cell_centers()

    lx = grid.nx * grid.dx
    ly = grid.ny * grid.dy

    return np.cos(np.pi * x / lx) * np.cos(np.pi * y / ly)


def manufactured_source(
    grid: CartesianGrid,
) -> NDArray[np.float64]:
    """Calcula f = -laplaciano(phi) analiticamente."""

    lx = grid.nx * grid.dx
    ly = grid.ny * grid.dy

    eigenvalue = np.pi**2 * (1.0 / lx**2 + 1.0 / ly**2)

    return eigenvalue * manufactured_solution(grid)


def calculate_errors(
    numerical: NDArray[np.float64],
    exact: NDArray[np.float64],
) -> tuple[float, float]:
    """Retorna erros RMS (L2 discreto) e L-infinito."""

    numerical = np.asarray(numerical, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)

    if numerical.shape != exact.shape:
        raise ValueError("Campos devem ter a mesma dimensao")

    if numerical.ndim != 2 or numerical.size == 0:
        raise ValueError("Campos devem ser matrizes nao vazias")

    if not (np.all(np.isfinite(numerical)) and np.all(np.isfinite(exact))):
        raise ValueError("Campos devem conter valores finitos")

    difference = numerical - exact

    l2_error = float(np.linalg.norm(difference) / np.sqrt(difference.size))

    linf_error = float(np.max(np.abs(difference)))

    return l2_error, linf_error


def observed_order(
    coarse_error: float,
    fine_error: float,
    refinement_ratio: float = 2.0,
) -> float:
    """Calcula a ordem observada entre duas malhas."""

    if (
        not np.isfinite(coarse_error)
        or not np.isfinite(fine_error)
        or coarse_error <= 0
        or fine_error <= 0
    ):
        raise ValueError("Erros devem ser finitos e positivos")

    if not np.isfinite(refinement_ratio) or refinement_ratio <= 1:
        raise ValueError("refinement_ratio deve ser finito e maior que 1")

    return float(np.log(coarse_error / fine_error) / np.log(refinement_ratio))


def verify_poisson_solver(
    nx: int,
    ny: int,
    *,
    lx: float = 1.0,
    ly: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 30_000,
) -> VerificationResult:
    """Resolve Poisson manufaturado e mede erros numericos."""

    grid = create_grid(SimulationConfig(nx=nx, ny=ny, lx=lx, ly=ly))

    coefficients = assemble_poisson_coefficients(grid)

    exact = manufactured_solution(grid)
    rhs = manufactured_source(grid)

    # A solucao analitica tem media zero nessa malha.
    # Nao projetamos o termo-fonte silenciosamente.
    rhs_mean = float(np.mean(rhs))

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    if not result.converged:
        raise RuntimeError(
            "Gauss-Seidel nao convergiu na verificacao: "
            f"nx={nx}, ny={ny}, "
            f"residual={result.residual:.3e}, "
            f"iterations={result.iterations}"
        )

    l2_error, linf_error = calculate_errors(result.phi, exact)

    # Erro algébrico: compara A(phi_numerica) com rhs.
    algebraic_residual = rhs - apply_poisson_system(result.phi, coefficients)

    algebraic_error = float(
        np.linalg.norm(algebraic_residual) / np.sqrt(algebraic_residual.size)
    )

    return VerificationResult(
        nx=nx,
        ny=ny,
        dx=grid.dx,
        dy=grid.dy,
        l2_error=l2_error,
        linf_error=linf_error,
        residual=result.residual,
        iterations=result.iterations,
        converged=result.converged,
        rhs_mean=rhs_mean,
        algebraic_error=algebraic_error,
    )
