"""FDM-06: construção e projeção dos fluxos nas faces."""

import numpy as np
from numpy.typing import NDArray

from navierstokes.fdm.boundary import interpolate_impermeable_faces
from navierstokes.fdm.gauss_seidel import (
    GaussSeidelResult,
    solve_poisson_gauss_seidel,
)
from navierstokes.fdm.grid import CartesianGrid
from navierstokes.fdm.operators import divergence_faces, gradient_faces
from navierstokes.fdm.poisson import assemble_poisson_coefficients
from navierstokes.fdm.predictor import _check_field


def pressure_gradient_centers(
    pressure: NDArray,
    grid: CartesianGrid,
) -> tuple[NDArray, NDArray]:
    """Calcula o gradiente central a partir dos gradientes nas faces.

    No interior, equivale a diferenças centrais de segunda ordem.
    Nas células adjacentes às paredes, utiliza Neumann homogênea
    nas faces externas, conforme o operador G_h existente.
    """
    _check_field(pressure, grid.scalar_shape, "pressure")

    grad_px_faces, grad_py_faces = gradient_faces(pressure, grid)

    grad_px = 0.5 * (
        grad_px_faces[:, :-1] + grad_px_faces[:, 1:]
    )
    grad_py = 0.5 * (
        grad_py_faces[:-1, :] + grad_py_faces[1:, :]
    )

    return grad_px, grad_py


def correct_center_velocities(
    u_star: NDArray,
    v_star: NDArray,
    pressure: NDArray,
    grid: CartesianGrid,
    *,
    density: float,
    dt: float,
) -> tuple[NDArray, NDArray]:
    """Corrige diretamente as velocidades nos centros das células.

    Não reconstrói velocidades a partir dos fluxos projetados.
    As condições de Dirichlet nas paredes são aplicadas pelo
    mecanismo de células fantasmas já existente.
    """
    _check_field(u_star, grid.scalar_shape, "u_star")
    _check_field(v_star, grid.scalar_shape, "v_star")
    _check_field(pressure, grid.scalar_shape, "pressure")

    if not np.isfinite(density) or density <= 0:
        raise ValueError("density deve ser finita e positiva")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt deve ser finito e positivo")

    grad_px, grad_py = pressure_gradient_centers(
        pressure, grid
    )

    factor = dt / density

    u_new = u_star - factor * grad_px
    v_new = v_star - factor * grad_py

    return u_new, v_new

def correct_face_fluxes(
    u_faces_star: NDArray,
    v_faces_star: NDArray,
    pressure: NDArray,
    grid: CartesianGrid,
    *,
    density: float,
    dt: float,
) -> tuple[NDArray, NDArray]:
    """Corrige os fluxos provisórios pelo gradiente discreto da pressão.

    Usa o mesmo operador G_h da montagem de Poisson, garantindo
    compatibilidade com L_h = D_h G_h.

    Aplicável à cavidade fechada com Neumann homogênea para pressão.
    Não modifica os arrays de entrada.
    """
    _check_field(u_faces_star, grid.u_face_shape, "u_faces_star")
    _check_field(v_faces_star, grid.v_face_shape, "v_faces_star")
    _check_field(pressure, grid.scalar_shape, "pressure")

    if not np.isfinite(density) or density <= 0:
        raise ValueError("density deve ser finita e positiva")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt deve ser finito e positivo")

    if not (
        np.allclose(
            u_faces_star[:, [0, -1]], 0.0, atol=1e-14, rtol=0
        )
        and np.allclose(
            v_faces_star[[0, -1], :], 0.0, atol=1e-14, rtol=0
        )
    ):
        raise ValueError(
            "Cavidade impermeável exige fluxos normais externos nulos"
        )

    grad_px, grad_py = gradient_faces(pressure, grid)
    correction_factor = dt / density

    u_faces_new = u_faces_star - correction_factor * grad_px
    v_faces_new = v_faces_star - correction_factor * grad_py

    return u_faces_new, v_faces_new

def solve_projection_pressure(
    u_faces_star: NDArray,
    v_faces_star: NDArray,
    grid: CartesianGrid,
    *,
    density: float,
    dt: float,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
    initial_pressure: NDArray | None = None,
) -> GaussSeidelResult:
    """Resolve A p = -(rho/dt) D_h(u_f*), com A = -D_h G_h.

    Exige cavidade impermeável e compatibilidade do termo-fonte.
    Rejeita soluções que não atingiram a tolerância especificada.
    """
    _check_field(u_faces_star, grid.u_face_shape, "u_faces_star")
    _check_field(v_faces_star, grid.v_face_shape, "v_faces_star")

    if not np.isfinite(density) or density <= 0:
        raise ValueError("density deve ser finita e positiva")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt deve ser finito e positivo")

    if not (
        np.allclose(
            u_faces_star[:, [0, -1]], 0.0, atol=1e-14, rtol=0
        )
        and np.allclose(
            v_faces_star[[0, -1], :], 0.0, atol=1e-14, rtol=0
        )
    ):
        raise ValueError(
            "Cavidade impermeável exige fluxos normais externos nulos"
        )

    divergence = divergence_faces(
        u_faces_star, v_faces_star, grid
    )
    rhs = -(density / dt) * divergence

    if not np.all(np.isfinite(rhs)):
        raise ValueError("Termo-fonte de pressão não finito")

    coefficients = assemble_poisson_coefficients(grid)

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        tolerance=tolerance,
        max_iterations=max_iterations,
        initial_phi=initial_pressure,
    )

    if not result.converged:
        raise RuntimeError(
            "Poisson não convergiu: "
            f"iterações={result.iterations}, "
            f"resíduo={result.residual:.6e}"
        )

    return result

def predict_face_fluxes(
    u: NDArray,
    v: NDArray,
    u_star: NDArray,
    v_star: NDArray,
    u_faces: NDArray,
    v_faces: NDArray,
    grid: CartesianGrid,
) -> tuple[NDArray, NDArray]:
    """Atualiza incrementalmente os fluxos normais da cavidade fechada.

    Os fluxos de entrada representam o estado projetado no instante n.
    Os campos centrais provisórios são produzidos pelo preditor FDM-05.

    Não modifica os arrays recebidos nem realiza projeção de pressão.
    """
    for name, field, shape in (
        ("u", u, grid.scalar_shape),
        ("v", v, grid.scalar_shape),
        ("u_star", u_star, grid.scalar_shape),
        ("v_star", v_star, grid.scalar_shape),
        ("u_faces", u_faces, grid.u_face_shape),
        ("v_faces", v_faces, grid.v_face_shape),
    ):
        _check_field(field, shape, name)

    if not (
        np.allclose(u_faces[:, [0, -1]], 0.0, atol=1e-14, rtol=0)
        and np.allclose(v_faces[[0, -1], :], 0.0, atol=1e-14, rtol=0)
    ):
        raise ValueError(
            "Cavidade impermeável exige fluxos normais externos nulos"
        )

    delta_u = u_star - u
    delta_v = v_star - v

    delta_u_faces, delta_v_faces = interpolate_impermeable_faces(
        delta_u, delta_v, grid
    )

    u_faces_star = u_faces + delta_u_faces
    v_faces_star = v_faces + delta_v_faces

    # Impõe exatamente a impermeabilidade nas fronteiras.
    u_faces_star[:, 0] = 0.0
    u_faces_star[:, -1] = 0.0
    v_faces_star[0, :] = 0.0
    v_faces_star[-1, :] = 0.0

    return u_faces_star, v_faces_star
