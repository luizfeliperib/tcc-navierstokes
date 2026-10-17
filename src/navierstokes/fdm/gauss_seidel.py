"""Solver Gauss-Seidel para Poisson com Neumann homogenea."""

from dataclasses import dataclass
from typing import Literal

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
    backend: Literal["python", "numba"] = "python",
    residual_check_interval: int = 1,
) -> GaussSeidelResult:
    """Resolve A phi = rhs usando Gauss-Seidel lexicografico.

    A = -L_h. Exige rhs compativel com Neumann homogenea.
    Usa media(phi) = 0 como referencia.

    Residuo:
        ||rhs - A(phi)||_2 / sqrt(numero de celulas).

    residual_check_interval:
        Controla a frequencia de avaliacao do residuo verdadeiro.

        1: verifica em todas as iteracoes (comportamento original).
        k: verifica a cada k iteracoes e na ultima permitida.

    residual_history:
        Contem o residuo inicial e os residuos efetivamente
        avaliados. Com intervalo maior que 1, o historico
        e amostrado e nao possui um valor por iteracao.

    A normalizacao da pressao permanece em todas as iteracoes.
    """

    shape = coefficients.center.shape

    rhs = np.asarray(rhs, dtype=np.float64)

    # ========================================================
    # 1. VALIDACAO DAS ENTRADAS
    # ========================================================

    if rhs.shape != shape:
        raise ValueError("Shape de rhs incompatível com o sistema")

    if not np.all(np.isfinite(rhs)):
        raise ValueError("rhs deve conter apenas valores finitos")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance deve ser finita e positiva")

    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
    ):
        raise ValueError("max_iterations deve ser inteiro positivo")

    if (
        isinstance(residual_check_interval, (bool, np.bool_))
        or not isinstance(
            residual_check_interval,
            (int, np.integer),
        )
        or residual_check_interval < 1
    ):
        raise ValueError("residual_check_interval deve ser inteiro positivo")

    if backend not in ("python", "numba"):
        raise ValueError("backend deve ser 'python' ou 'numba'")

    if not np.isfinite(compatibility_tolerance) or compatibility_tolerance < 0:
        raise ValueError("compatibility_tolerance deve ser finita e nao negativa")

    # ========================================================
    # 2. COMPATIBILIDADE DO TERMO-FONTE
    # ========================================================

    # Para Neumann homogenea, o termo-fonte precisa ter
    # media discreta aproximadamente igual a zero.
    #
    # A escala evita comparar erros absolutos de
    # arredondamento com campos de amplitudes diferentes.

    mean_rhs = float(np.mean(rhs))
    scale_rhs = max(
        1.0,
        float(np.max(np.abs(rhs))),
    )

    if abs(mean_rhs) > compatibility_tolerance * scale_rhs:
        raise ValueError(
            "rhs incompatível com Neumann homogênea: "
            "a média discreta deve ser aproximadamente zero"
        )

    # ========================================================
    # 3. INICIALIZACAO DA PRESSAO
    # ========================================================

    if initial_phi is None:
        phi = np.zeros(
            shape,
            dtype=np.float64,
        )
    else:
        initial_phi = np.asarray(
            initial_phi,
            dtype=np.float64,
        )

        if initial_phi.shape != shape:
            raise ValueError("Shape de initial_phi incompatível")

        if not np.all(np.isfinite(initial_phi)):
            raise ValueError("initial_phi deve ser finito")

        # Evita modificar o array recebido.
        phi = initial_phi.copy()

    # ========================================================
    # 4. VALIDACAO DOS COEFICIENTES
    # ========================================================

    for name, array in (
        ("center", coefficients.center),
        ("east", coefficients.east),
        ("west", coefficients.west),
        ("north", coefficients.north),
        ("south", coefficients.south),
    ):
        if array.shape != shape:
            raise ValueError(f"Shape de {name} incompatível")

        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} deve conter valores finitos")

    if np.any(coefficients.center <= 0):
        raise ValueError("Diagonal de Poisson inválida")

    # ========================================================
    # 5. REFERENCIA DE PRESSAO
    # ========================================================

    # Como A(1) = 0, subtrair uma constante de phi
    # nao altera o sistema discreto de Poisson.
    #
    # Fixamos o representante de media zero.

    phi -= np.mean(phi)

    ny, nx = shape

    # ========================================================
    # 6. CALCULO DO RESIDUO VERDADEIRO
    # ========================================================

    def true_residual() -> float:
        residual_field = rhs - apply_poisson_system(
            phi,
            coefficients,
        )

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            residual_norm = np.linalg.norm(residual_field)

        return float(residual_norm / np.sqrt(rhs.size))

    # ========================================================
    # 7. RESIDUO INICIAL E HISTORICO
    # ========================================================

    residual = true_residual()

    if not np.isfinite(residual):
        raise FloatingPointError("Residuo inicial nao finito")

    # A variavel history e inicializada antes do laco.
    history = [residual]

    # Permite convergencia imediata, sem varreduras.
    if residual <= tolerance:
        return GaussSeidelResult(
            phi=phi,
            converged=True,
            iterations=0,
            residual=residual,
            residual_history=tuple(history),
        )

    # ========================================================
    # 8. PREPARACAO DO BACKEND NUMBA
    # ========================================================

    if backend == "numba":
        try:
            from navierstokes.fdm.gauss_seidel_numba import (
                gauss_seidel_sweep,
            )
        except ImportError as exc:
            raise ImportError(
                "O backend Numba exige instalar o extra "
                "'performance': "
                "pip install -e '.[performance]'"
            ) from exc

    # ========================================================
    # 9. ITERACOES DO GAUSS-SEIDEL
    # ========================================================

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        # ----------------------------------------------------
        # 9.1. VARREDURA GAUSS-SEIDEL
        # ----------------------------------------------------

        if backend == "numba":
            gauss_seidel_sweep(
                phi,
                rhs,
                coefficients.center,
                coefficients.east,
                coefficients.west,
                coefficients.north,
                coefficients.south,
            )

        else:
            for j in range(ny):
                for i in range(nx):
                    neighbor_sum = 0.0

                    if i + 1 < nx:
                        neighbor_sum += coefficients.east[j, i] * phi[j, i + 1]

                    if i > 0:
                        neighbor_sum += coefficients.west[j, i] * phi[j, i - 1]

                    if j + 1 < ny:
                        neighbor_sum += coefficients.north[j, i] * phi[j + 1, i]

                    if j > 0:
                        neighbor_sum += coefficients.south[j, i] * phi[j - 1, i]

                    phi[j, i] = (rhs[j, i] + neighbor_sum) / coefficients.center[j, i]

        # ----------------------------------------------------
        # 9.2. NORMALIZACAO DA PRESSAO
        # ----------------------------------------------------

        # Mantida em TODAS as iteracoes, independentemente
        # do intervalo escolhido para verificar o residuo.

        phi -= np.mean(phi)

        # ----------------------------------------------------
        # 9.3. VERIFICACAO PERIODICA DO RESIDUO
        # ----------------------------------------------------

        should_check_residual = (
            iteration % residual_check_interval == 0 or iteration == max_iterations
        )

        if should_check_residual:
            residual = true_residual()

            history.append(residual)

            if not np.isfinite(residual):
                raise FloatingPointError("Residuo nao finito durante Gauss-Seidel")

            # A convergencia so e aceita depois de
            # calcular o residuo verdadeiro.

            if residual <= tolerance:
                return GaussSeidelResult(
                    phi=phi,
                    converged=True,
                    iterations=iteration,
                    residual=residual,
                    residual_history=tuple(history),
                )

    # ========================================================
    # 10. LIMITE DE ITERACOES ATINGIDO
    # ========================================================

    # A ultima iteracao sempre calcula o residuo,
    # mesmo que nao seja multipla do intervalo.
    #
    # Assim, o resultado final nunca utiliza
    # um residuo desatualizado.

    return GaussSeidelResult(
        phi=phi,
        converged=False,
        iterations=max_iterations,
        residual=residual,
        residual_history=tuple(history),
    )
