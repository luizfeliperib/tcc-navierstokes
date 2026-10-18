
"""Diagnostico do acoplamento entre velocidades centrais e fluxos nas faces.

Executa o algoritmo FDM existente sem modificar o solver original.

Saidas:
    coupling_history.csv
    coupling_fields.npz
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.boundary import interpolate_impermeable_faces
from navierstokes.fdm.diagnostics import calculate_diagnostics
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.operators import divergence_faces, gradient_faces
from navierstokes.fdm.poisson import assemble_poisson_coefficients
from navierstokes.fdm.predictor import velocity_predictor
from navierstokes.fdm.projection import (
    correct_center_velocities,
    correct_face_fluxes,
    predict_face_fluxes,
    pressure_gradient_centers,
    solve_projection_pressure,
)
from navierstokes.fdm.solver import initialize_state
from navierstokes.fdm.time_step import FlowState


def rms(array):
    """Calcula a raiz do erro quadratico medio."""
    return float(np.sqrt(np.mean(np.square(array))))


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostico do acoplamento centro-face do solver FDM"
    )

    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--reynolds", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--record-every", type=int, default=100)
    parser.add_argument("--omega", type=float, default=1.9)
    parser.add_argument(
        "--backend",
        choices=("python", "numba"),
        default="numba",
    )
    parser.add_argument(
        "--pressure-tolerance",
        type=float,
        default=1e-9,
    )
    parser.add_argument(
        "--pressure-max-iterations",
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
        default=Path("results/fdm_coupling_diagnostic"),
    )

    args = parser.parse_args()

    if args.steps < 1 or args.record_every < 1:
        parser.error("steps e record-every devem ser positivos")

    # ---------------------------------------------------------
    # 1. CONFIGURACAO E INICIALIZACAO
    # ---------------------------------------------------------

    config = SimulationConfig(
        reynolds=args.reynolds,
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        max_iterations=args.steps,
    )

    grid = create_grid(config)
    coefficients = assemble_poisson_coefficients(grid)
    state = initialize_state(grid)

    history = []
    snapshots = {}

    start_time = perf_counter()

    print("\n=== DIAGNOSTICO FDM: ACOPLAMENTO CENTRO-FACE ===")
    print(f"Reynolds: {args.reynolds}")
    print(f"Malha: {args.nx} x {args.ny}")
    print(f"dt: {args.dt}")
    print(f"Passos: {args.steps}")
    print(f"SOR omega: {args.omega}")
    print(f"Backend: {args.backend}")
    print()

    # ---------------------------------------------------------
    # 2. INTEGRACAO TEMPORAL
    # ---------------------------------------------------------

    for step in range(1, args.steps + 1):

        previous = state

        # 2.1 Predicao das velocidades centrais
        u_star, v_star = velocity_predictor(
            previous.u,
            previous.v,
            previous.u_faces,
            previous.v_faces,
            grid,
            config,
        )

        # 2.2 Predicao incremental dos fluxos nas faces
        u_faces_star, v_faces_star = predict_face_fluxes(
            previous.u,
            previous.v,
            u_star,
            v_star,
            previous.u_faces,
            previous.v_faces,
            grid,
        )

        # 2.3 Solucao da equacao de Poisson
        pressure_result = solve_projection_pressure(
            u_faces_star,
            v_faces_star,
            grid,
            density=config.density,
            dt=config.dt,
            tolerance=args.pressure_tolerance,
            max_iterations=args.pressure_max_iterations,
            initial_pressure=previous.pressure,
            backend=args.backend,
            residual_check_interval=args.residual_check_interval,
            coefficients=coefficients,
            omega=args.omega,
        )

        pressure = pressure_result.phi

        # 2.4 Projecao dos fluxos nas faces
        u_faces, v_faces = correct_face_fluxes(
            u_faces_star,
            v_faces_star,
            pressure,
            grid,
            density=config.density,
            dt=config.dt,
        )

        # 2.5 Correcao das velocidades centrais
        u, v = correct_center_velocities(
            u_star,
            v_star,
            pressure,
            grid,
            density=config.density,
            dt=config.dt,
        )

        state = FlowState(
            u=u,
            v=v,
            u_faces=u_faces,
            v_faces=v_faces,
            pressure=pressure,
        )

        # -----------------------------------------------------
        # 3. DIAGNOSTICOS PERIODICOS
        # -----------------------------------------------------

        should_record = (
            step == 1
            or step % args.record_every == 0
            or step == args.steps
        )

        if not should_record:
            continue

        diagnostics = calculate_diagnostics(state, grid)

        # Interpolacao dos centros para as faces
        old_u_interp, old_v_interp = interpolate_impermeable_faces(
            previous.u,
            previous.v,
            grid,
        )

        new_u_interp, new_v_interp = interpolate_impermeable_faces(
            u,
            v,
            grid,
        )

        # Discrepancia entre fluxos e velocidades interpoladas
        old_error_u = previous.u_faces - old_u_interp
        old_error_v = previous.v_faces - old_v_interp

        new_error_u = u_faces - new_u_interp
        new_error_v = v_faces - new_v_interp

        # -----------------------------------------------------
        # 3.1 VERIFICACAO DA IDENTIDADE ALGEBRICA
        #
        # e(n+1) = e(n) + dt/rho * [I(Gc p) - Gf p]
        # -----------------------------------------------------

        grad_center_u, grad_center_v = pressure_gradient_centers(
            pressure,
            grid,
        )

        grad_interp_u, grad_interp_v = interpolate_impermeable_faces(
            grad_center_u,
            grad_center_v,
            grid,
        )

        grad_face_u, grad_face_v = gradient_faces(
            pressure,
            grid,
        )

        factor = config.dt / config.density

        identity_error_u = rms(
            new_error_u
            - old_error_u
            - factor * (grad_interp_u - grad_face_u)
        )

        identity_error_v = rms(
            new_error_v
            - old_error_v
            - factor * (grad_interp_v - grad_face_v)
        )

        # -----------------------------------------------------
        # 3.2 DIVERGENCIA DOS CAMPOS
        # -----------------------------------------------------

        # Divergencia dos fluxos projetados
        divergence_projected = divergence_faces(
            u_faces,
            v_faces,
            grid,
        )

        # Divergencia das velocidades centrais interpoladas
        divergence_interpolated = divergence_faces(
            new_u_interp,
            new_v_interp,
            grid,
        )

        # Discrepancia apenas nas faces internas
        error_u_interior = rms(new_error_u[:, 1:-1])
        error_v_interior = rms(new_error_v[1:-1, :])

        # -----------------------------------------------------
        # 3.3 RESIDUO ESTACIONARIO
        # -----------------------------------------------------

        steady_residual = max(
            rms(u - previous.u),
            rms(v - previous.v),
        ) / config.dt

        elapsed = perf_counter() - start_time

        # -----------------------------------------------------
        # 3.4 REGISTRO
        # -----------------------------------------------------

        record = {
            "step": step,
            "physical_time": step * config.dt,
            "elapsed_seconds": elapsed,
            "steady_residual": steady_residual,
            "poisson_iterations": pressure_result.iterations,
            "poisson_residual": pressure_result.residual,
            "face_divergence_rms": rms(divergence_projected),
            "interpolated_center_divergence_rms": rms(
                divergence_interpolated
            ),
            "center_face_u_rms": diagnostics.center_face_u_rms,
            "center_face_v_rms": diagnostics.center_face_v_rms,
            "face_minus_center_u_interior_rms": error_u_interior,
            "face_minus_center_v_interior_rms": error_v_interior,
            "coupling_identity_error_u": identity_error_u,
            "coupling_identity_error_v": identity_error_v,
            "pressure_gradient_mismatch_u_rms": rms(
                (grad_interp_u - grad_face_u)[:, 1:-1]
            ),
            "pressure_gradient_mismatch_v_rms": rms(
                (grad_interp_v - grad_face_v)[1:-1, :]
            ),
            "max_abs_u": diagnostics.max_abs_u,
            "max_abs_v": diagnostics.max_abs_v,
            "kinetic_energy_mean": diagnostics.kinetic_energy_mean,
        }

        history.append(record)

        print(
            f"Passo {step:6d}/{args.steps} | "
            f"Rsteady={steady_residual:.3e} | "
            f"div_faces={record['face_divergence_rms']:.3e} | "
            f"div_centros={record['interpolated_center_divergence_rms']:.3e} | "
            f"erro_u={error_u_interior:.3e} | "
            f"erro_v={error_v_interior:.3e} | "
            f"Poisson={pressure_result.iterations:4d} | "
            f"Tempo={elapsed:.1f}s",
            flush=True,
        )

        # Guarda os campos inicial e final do experimento
        if step in (1, args.steps):
            snapshots[f"u_{step}"] = u.copy()
            snapshots[f"v_{step}"] = v.copy()
            snapshots[f"uf_{step}"] = u_faces.copy()
            snapshots[f"vf_{step}"] = v_faces.copy()

    # ---------------------------------------------------------
    # 4. EXPORTACAO DOS RESULTADOS
    # ---------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = args.output / "coupling_history.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )

        writer.writeheader()
        writer.writerows(history)

    npz_path = args.output / "coupling_fields.npz"

    np.savez_compressed(
        npz_path,
        x=grid.x,
        y=grid.y,
        **snapshots,
    )

    print("\n=== DIAGNOSTICO CONCLUIDO ===")
    print(f"Tempo total: {perf_counter() - start_time:.2f}s")
    print(f"Historico: {csv_path}")
    print(f"Campos: {npz_path}")
    print()
    print(
        "Nota: identidade algebrica proxima de zero confirma "
        "a implementacao da identidade, nao a validade fisica."
    )


if __name__ == "__main__":
    main()
