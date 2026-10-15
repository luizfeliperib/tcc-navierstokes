
"""Solver Gauss-Seidel para Poisson com Neumann homogenea."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from navierstokes.fdm.poisson import (
    PoissonCoefficients,
    apply_poisson_system,
)


@dataclass(frozen=True, slots=True)
class GaussSeidelResult:
    phi: NDArray[np.float64]
    converged: bool
    iterations: int
    residual: float
    residual_history: tuple[float, ...]


def solve_poisson_gauss_seidel(
    rhs: NDArray[np.float64],
    coefficients: PoissonCoefficients,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
    initial_phi: NDArray[np.float64] | None = None,
    compatibility_tolerance: float = 1e-10,
) -> GaussSeidelResult:
    """Resolve A phi = rhs usando Gauss-Seidel lexicografico.

    A = -L_h. Exige rhs compativel com Neumann homogenea.
    Usa media(phi) = 0 como referencia.
    Residuo: ||rhs - A(phi)||_2 / sqrt(numero de celulas).
    """

    shape = coefficients.center.shape

    rhs = np.asarray(rhs, dtype=np.float64)

    if rhs.shape != shape:
        raise ValueError("Shape de rhs incompatível com o sistema")

    if not np.all(np.isfinite(rhs)):
        raise ValueError("rhs deve conter apenas valores finitos")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance deve ser finita e positiva")

    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations deve ser inteiro positivo")

    if (
        not np.isfinite(compatibility_tolerance)
        or compatibility_tolerance < 0
    ):
        raise ValueError(
            "compatibility_tolerance deve ser finita e nao negativa"
        )

    # Como a malha e uniforme, soma(rhs) = 0 equivale a media(rhs) = 0.
    # A escala evita comparar um erro de arredondamento absoluto
    # com campos de amplitudes muito diferentes.
    mean_rhs = float(np.mean(rhs))
    scale_rhs = max(1.0, float(np.max(np.abs(rhs))))

    if abs(mean_rhs) > compatibility_tolerance * scale_rhs:
        raise ValueError(
            "rhs incompatível com Neumann homogênea: "
            "a média discreta deve ser aproximadamente zero"
        )

    if initial_phi is None:
        phi = np.zeros(shape, dtype=np.float64)
    else:
        initial_phi = np.asarray(initial_phi, dtype=np.float64)
        if initial_phi.shape != shape:
            raise ValueError("Shape de initial_phi incompatível")
        if not np.all(np.isfinite(initial_phi)):
            raise ValueError("initial_phi deve ser finito")
        phi = initial_phi.copy()

    # Fixa o representante da familia de solucoes.
    phi -= np.mean(phi)

    ny, nx = shape

    def true_residual() -> float:
        residual_field = rhs - apply_poisson_system(
            phi, coefficients
        )
        return float(np.linalg.norm(residual_field) / np.sqrt(rhs.size))

    residual = true_residual()
    history = [residual]

    if residual <= tolerance:
        return GaussSeidelResult(
            phi=phi,
            converged=True,
            iterations=0,
            residual=residual,
            residual_history=tuple(history),
        )

    for iteration in range(1, max_iterations + 1):
        for j in range(ny):
            for i in range(nx):
                diagonal = coefficients.center[j, i]

                if diagonal <= 0 or not np.isfinite(diagonal):
                    raise ValueError("Diagonal de Poisson inválida")

                neighbor_sum = 0.0

                if i + 1 < nx:
                    neighbor_sum += (
                        coefficients.east[j, i] * phi[j, i + 1]
                    )
                if i > 0:
                    neighbor_sum += (
                        coefficients.west[j, i] * phi[j, i - 1]
                    )
                if j + 1 < ny:
                    neighbor_sum += (
                        coefficients.north[j, i] * phi[j + 1, i]
                    )
                if j > 0:
                    neighbor_sum += (
                        coefficients.south[j, i] * phi[j - 1, i]
                    )

                phi[j, i] = (
                    rhs[j, i] + neighbor_sum
                ) / diagonal

        # Remover a media nao altera A(phi), pois A(1) = 0.
        phi -= np.mean(phi)

        residual = true_residual()
        history.append(residual)

        if not np.isfinite(residual):
            raise FloatingPointError(
                "Residuo nao finito durante Gauss-Seidel"
            )

        if residual <= tolerance:
            return GaussSeidelResult(
                phi=phi,
                converged=True,
                iterations=iteration,
                residual=residual,
                residual_history=tuple(history),
            )

    return GaussSeidelResult(
        phi=phi,
        converged=False,
        iterations=max_iterations,
        residual=residual,
        residual_history=tuple(history),
    )
