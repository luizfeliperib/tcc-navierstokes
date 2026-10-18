
"""FDM-07.7.2-D: experimento controlado do acoplamento centro-face.

Variante A: construcao incremental original.
Variante B: reconstrucao dos fluxos provisórios a partir do preditor.

ATENCAO: variante B e apenas diagnostica, nao uma correcao validada.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.boundary import interpolate_impermeable_faces
from navierstokes.fdm.solver import solve
from run_fdm_validation import export_results


def reconstructed_face_predictor(
    u, v, u_star, v_star, u_faces, v_faces, grid
):
    """Reconstrói os fluxos provisórios a partir de u* e v*."""
    return interpolate_impermeable_faces(u_star, v_star, grid)


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def execute(config, args, name, experimental):
    output = args.output / name

    options = dict(
        backend=args.backend,
        omega=args.omega,
        pressure_tolerance=args.pressure_tolerance,
        pressure_max_iterations=args.pressure_max_iterations,
        residual_check_interval=args.residual_check_interval,
        record_interval=args.record_interval,
        stop_on_steady=False,
    )

    print(f"\nExecutando {name}...", flush=True)
    start = perf_counter()

    if experimental:
        # Substituicao temporaria somente durante esta simulacao.
        with patch(
            "navierstokes.fdm.time_step.predict_face_fluxes",
            new=reconstructed_face_predictor,
        ):
            result = solve(config, **options)
    else:
        result = solve(config, **options)

    elapsed = perf_counter() - start

    export_results(
        result=result,
        config=config,
        output_dir=output,
        elapsed=elapsed,
        settings={
            **options,
            "diagnostic_variant": name,
            "experimental": experimental,
        },
    )

    diag = result.history[-1].diagnostics

    print(
        f"{name}: {elapsed:.2f}s | "
        f"Rsteady={result.steady_residual:.6e} | "
        f"div={diag.divergence_rms:.6e} | "
        f"centro-face u={diag.center_face_u_rms:.6e} | "
        f"v={diag.center_face_v_rms:.6e}",
        flush=True,
    )

    return result, elapsed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reynolds", type=float, default=100)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5000)

    parser.add_argument("--backend", default="numba",
                        choices=("python", "numba"))
    parser.add_argument("--omega", type=float, default=1.9)
    parser.add_argument("--pressure-tolerance", type=float,
                        default=1e-9)
    parser.add_argument("--pressure-max-iterations", type=int,
                        default=20000)
    parser.add_argument("--residual-check-interval", type=int,
                        default=10)
    parser.add_argument("--record-interval", type=int,
                        default=100)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fdm_077_projection_diagnostic"),
    )

    args = parser.parse_args()

    config = SimulationConfig(
        reynolds=args.reynolds,
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        max_iterations=args.steps,
        tolerance=1e-5,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    original, original_time = execute(
        config, args, "original", False
    )

    experimental, experimental_time = execute(
        config, args, "reconstructed", True
    )

    a = original.state
    b = experimental.state

    metrics = {
        "u_rms_difference": rms(b.u - a.u),
        "v_rms_difference": rms(b.v - a.v),
        "u_max_difference": float(
            np.max(np.abs(b.u - a.u))
        ),
        "v_max_difference": float(
            np.max(np.abs(b.v - a.v))
        ),
        "face_u_rms_difference": rms(
            b.u_faces - a.u_faces
        ),
        "face_v_rms_difference": rms(
            b.v_faces - a.v_faces
        ),
        "original_steady_residual": original.steady_residual,
        "experimental_steady_residual": (
            experimental.steady_residual
        ),
        "original_seconds": original_time,
        "experimental_seconds": experimental_time,
    }

    with (args.output / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(metrics.keys())
        )
        writer.writeheader()
        writer.writerow(metrics)

    print("\n=== DIFERENCAS ENTRE AS VARIANTES ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.8e}")

    print(f"\nResultados em: {args.output.resolve()}")


if __name__ == "__main__":
    main()
