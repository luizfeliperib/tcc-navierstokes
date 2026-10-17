"""FDM-07.4.1/7.4.2: execução e exportação consolidadas."""

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.centerline import (
    extract_centerline_profiles,
)
from navierstokes.fdm.solver import solve


def export_results(
    result,
    config,
    output_dir,
    elapsed,
    settings,
):
    """Exporta campos, histórico e metadados da simulação."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = result.grid
    state = result.state

    # --------------------------------------------------
    # 1. Campos físicos
    # --------------------------------------------------

    np.savez_compressed(
        output_dir / "fields.npz",
        x=grid.x,
        y=grid.y,
        u=state.u,
        v=state.v,
        pressure=state.pressure,
        u_faces=state.u_faces,
        v_faces=state.v_faces,
    )

    # --------------------------------------------------
    # Perfis centrais de velocidade
    # --------------------------------------------------

    profiles = extract_centerline_profiles(
        grid,
        state.u,
        state.v,
    )

    # Perfil vertical: u(Lx/2, y)
    with (output_dir / "centerline_u.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["y", "u"])

        writer.writerows(
            zip(
                profiles["y"],
                profiles["u_vertical"],
            )
        )

    # Perfil horizontal: v(x, Ly/2)
    with (output_dir / "centerline_v.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["x", "v"])

        writer.writerows(
            zip(
                profiles["x"],
                profiles["v_horizontal"],
            )
        )

    # --------------------------------------------------
    # 2. Histórico da simulação
    # --------------------------------------------------

    fieldnames = [
        "step",
        "time",
        "steady_residual",
        "poisson_residual",
        "poisson_iterations",
        "divergence_rms",
        "center_face_u_rms",
        "center_face_v_rms",
        "max_abs_u",
        "max_abs_v",
        "kinetic_energy_mean",
    ]

    history_path = output_dir / "history.csv"

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in result.history:
            row = {
                "step": record.step,
                "time": record.time,
                "steady_residual": record.steady_residual,
                "poisson_residual": record.poisson_residual,
                "poisson_iterations": record.poisson_iterations,
                **asdict(record.diagnostics),
            }

            writer.writerow(row)

    # --------------------------------------------------
    # 3. Metadados da execução
    # --------------------------------------------------

    flow_status = "steady" if result.converged else "transient"

    metadata = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "problem": "lid_driven_cavity",
        "method": "FDM",
        "flow_status": flow_status,
        "configuration": asdict(config),
        "kinematic_viscosity": (config.kinematic_viscosity),
        "execution_settings": settings,
        "result": {
            "steps": result.steps,
            "final_time": result.final_time,
            "converged": result.converged,
            "stop_reason": result.stop_reason,
            "steady_residual": (result.steady_residual),
            "elapsed_seconds": elapsed,
            "seconds_per_step": (elapsed / result.steps if result.steps > 0 else None),
        },
        "grid": {
            "nx": grid.nx,
            "ny": grid.ny,
            "dx": grid.dx,
            "dy": grid.dy,
        },
        "centerline_profiles": {
            "vertical": {
                "file": "centerline_u.csv",
                "component": "u",
                "fixed_coordinate": "x",
                "fixed_value": (config.lx / 2.0),
                "sample_coordinate": "y",
                "includes_walls": False,
            },
            "horizontal": {
                "file": "centerline_v.csv",
                "component": "v",
                "fixed_coordinate": "y",
                "fixed_value": (config.ly / 2.0),
                "sample_coordinate": "x",
                "includes_walls": False,
            },
            "interpolation": "linear",
            "field_location": "cell_centers",
        },
        "artifacts": {
            "fields": "fields.npz",
            "history": "history.csv",
            "metadata": "metadata.json",
            "centerline_u": "centerline_u.csv",
            "centerline_v": "centerline_v.csv",
        },
    }

    with (output_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_dir


def build_parser():
    """Define os parâmetros configuráveis da execução."""

    parser = argparse.ArgumentParser(
        description=("Executa e exporta uma simulação FDM da cavidade com tampa móvel.")
    )

    # --------------------------------------------------
    # 1. Parâmetros físicos e espaciais
    # --------------------------------------------------

    parser.add_argument(
        "--reynolds",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
    )

    # --------------------------------------------------
    # 2. Critério estacionário
    # --------------------------------------------------

    parser.add_argument(
        "--steady-tolerance",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--steady-consecutive-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--min-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--stop-on-steady",
        action="store_true",
        help=("Interrompe ao atingir o critério estacionário."),
    )

    # --------------------------------------------------
    # 3. Solucionador de pressão
    # --------------------------------------------------

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
        default=20,
        help=("Intervalo de verificação do resíduo verdadeiro de Poisson."),
    )

    parser.add_argument(
        "--backend",
        choices=("python", "numba"),
        default="numba",
    )

    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help=("Fator de relaxação do Poisson: 1.0=Gauss-Seidel; 0<omega<2=SOR."),
    )

    # --------------------------------------------------
    # 4. Registro e exportação
    # --------------------------------------------------

    parser.add_argument(
        "--record-interval",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fdm_07_validation"),
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help=("Gera figuras após a execução."),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --------------------------------------------------
    # 1. Configuração da simulação
    # --------------------------------------------------

    config = SimulationConfig(
        reynolds=args.reynolds,
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        tolerance=args.steady_tolerance,
        max_iterations=args.steps,
    )

    # --------------------------------------------------
    # 2. Configurações de execução
    # --------------------------------------------------

    settings = {
        "backend": args.backend,
        "pressure_tolerance": args.pressure_tolerance,
        "pressure_max_iterations": (args.pressure_max_iterations),
        "residual_check_interval": (args.residual_check_interval),
        "record_interval": args.record_interval,
        "stop_on_steady": args.stop_on_steady,
        "steady_consecutive_steps": (args.steady_consecutive_steps),
        "min_steps": args.min_steps,
        "omega": args.omega,
    }

    print(
        "\nIniciando simulação FDM",
        flush=True,
    )

    print(
        f"Re={config.reynolds:g} | "
        f"Malha={config.nx}x{config.ny} | "
        f"dt={config.dt:g} | "
        f"Máximo={config.max_iterations} passos",
        flush=True,
    )

    print(
        f"Backend={args.backend} | "
        f"Poisson tol={args.pressure_tolerance:.1e} | "
        f"Intervalo resíduo={args.residual_check_interval}",
        flush=True,
    )

    # --------------------------------------------------
    # 3. Execução
    # --------------------------------------------------

    start = perf_counter()

    result = solve(
        config,
        backend=args.backend,
        pressure_tolerance=args.pressure_tolerance,
        pressure_max_iterations=(args.pressure_max_iterations),
        residual_check_interval=(args.residual_check_interval),
        stop_on_steady=args.stop_on_steady,
        steady_consecutive_steps=(args.steady_consecutive_steps),
        min_steps=args.min_steps,
        record_interval=args.record_interval,
        omega=args.omega,
    )

    elapsed = perf_counter() - start

    # --------------------------------------------------
    # 4. Exportação
    # --------------------------------------------------

    export_results(
        result=result,
        config=config,
        output_dir=args.output,
        elapsed=elapsed,
        settings=settings,
    )

    # --------------------------------------------------
    # 5. Figuras opcionais
    # --------------------------------------------------

    if args.plot:
        plot_script = Path(__file__).with_name("plot_fdm_validation.py")

        subprocess.run(
            [
                sys.executable,
                str(plot_script),
                "--input",
                str(args.output),
            ],
            check=True,
        )

    # --------------------------------------------------
    # 6. Resumo da execução
    # --------------------------------------------------

    print("\nSimulação concluída.")

    print(f"Passos executados: {result.steps}")

    print(f"Tempo físico final: {result.final_time:.6f}")

    print(f"Estacionário: {result.converged}")

    print(f"Motivo da parada: {result.stop_reason}")

    print(f"Resíduo estacionário: {result.steady_residual:.6e}")

    print(f"Tempo computacional: {elapsed:.2f} s")

    print(f"Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
