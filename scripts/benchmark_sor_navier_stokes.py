"""FDM-07.5.5: benchmark do SOR no solver Navier-Stokes."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from run_fdm_validation import export_results

from navierstokes.config import SimulationConfig
from navierstokes.fdm.solver import solve

OUTPUT = Path("results/fdm_0755_sor_navier_stokes")

CASES = {
    16: [1.0, 1.8],
    32: [1.0, 1.8, 1.9],
    48: [1.0, 1.8, 1.9],
}

REYNOLDS = 100.0
DT = 0.001
STEPS = 100

PRESSURE_TOLERANCE = 1e-9
PRESSURE_MAX_ITERATIONS = 20000
RESIDUAL_CHECK_INTERVAL = 10


def run_case(n, omega):
    config = SimulationConfig(
        reynolds=REYNOLDS,
        nx=n,
        ny=n,
        dt=DT,
        max_iterations=STEPS,
    )

    settings = {
        "backend": "numba",
        "omega": omega,
        "pressure_tolerance": PRESSURE_TOLERANCE,
        "pressure_max_iterations": PRESSURE_MAX_ITERATIONS,
        "residual_check_interval": RESIDUAL_CHECK_INTERVAL,
        "record_interval": 1,
        "stop_on_steady": False,
        "steady_consecutive_steps": 20,
        "min_steps": 100,
    }

    # Tempo somente do solver, sem exportacao.
    start = perf_counter()

    result = solve(
        config,
        backend="numba",
        omega=omega,
        pressure_tolerance=PRESSURE_TOLERANCE,
        pressure_max_iterations=PRESSURE_MAX_ITERATIONS,
        residual_check_interval=RESIDUAL_CHECK_INTERVAL,
        record_interval=1,
        stop_on_steady=False,
    )

    elapsed = perf_counter() - start

    case_dir = OUTPUT / f"mesh_{n}" / f"omega_{omega:.2f}"

    export_results(
        result=result,
        config=config,
        output_dir=case_dir,
        elapsed=elapsed,
        settings=settings,
    )

    iterations = [record.poisson_iterations for record in result.history]

    poisson_residuals = [record.poisson_residual for record in result.history]

    divergence = [record.diagnostics.divergence_rms for record in result.history]

    return {
        "mesh": f"{n}x{n}",
        "n": n,
        "omega": omega,
        "steps": result.steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / result.steps,
        "poisson_iterations_total": sum(iterations),
        "poisson_iterations_mean": float(np.mean(iterations)),
        "poisson_iterations_max": max(iterations),
        "poisson_residual_max": max(poisson_residuals),
        "divergence_rms_max": max(divergence),
        "steady_residual_final": result.steady_residual,
        "stop_reason": result.stop_reason,
    }


def compare_fields(n, omega):
    """Compara um caso SOR com o Gauss-Seidel da mesma malha."""

    reference = OUTPUT / f"mesh_{n}" / "omega_1.00" / "fields.npz"

    candidate = OUTPUT / f"mesh_{n}" / f"omega_{omega:.2f}" / "fields.npz"

    comparison = {
        "mesh": f"{n}x{n}",
        "omega": omega,
    }

    with np.load(reference) as gs, np.load(candidate) as sor:
        for field in ("u", "v", "pressure", "u_faces", "v_faces"):
            difference = sor[field] - gs[field]

            comparison[f"{field}_max_abs_diff"] = float(np.max(np.abs(difference)))

            comparison[f"{field}_rms_diff"] = float(np.sqrt(np.mean(difference**2)))

    return comparison


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Aquecimento do Numba, fora das medicoes principais.
    warmup = SimulationConfig(
        reynolds=100.0,
        nx=4,
        ny=4,
        dt=DT,
        max_iterations=2,
    )

    solve(
        warmup,
        backend="numba",
        omega=1.0,
        pressure_tolerance=PRESSURE_TOLERANCE,
        pressure_max_iterations=PRESSURE_MAX_ITERATIONS,
        residual_check_interval=RESIDUAL_CHECK_INTERVAL,
        stop_on_steady=False,
    )

    rows = []

    for n, omegas in CASES.items():
        for omega in omegas:
            print(
                f"Executando {n}x{n}, omega={omega:.2f}",
                flush=True,
            )

            row = run_case(n, omega)
            rows.append(row)

            print(
                f"  Tempo: {row['elapsed_seconds']:.3f}s | "
                f"Poisson: {row['poisson_iterations_total']} "
                f"iteracoes | "
                f"Divergencia max: "
                f"{row['divergence_rms_max']:.3e}",
                flush=True,
            )

    comparisons = []

    for n, omegas in CASES.items():
        for omega in omegas:
            if omega != 1.0:
                comparisons.append(compare_fields(n, omega))

    def write_csv(path, data):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(data[0].keys()),
            )
            writer.writeheader()
            writer.writerows(data)

    write_csv(
        OUTPUT / "benchmark_summary.csv",
        rows,
    )

    write_csv(
        OUTPUT / "field_comparison.csv",
        comparisons,
    )

    metadata = {
        "experiment": "FDM-07.5.5",
        "problem": "lid_driven_cavity",
        "reynolds": REYNOLDS,
        "dt": DT,
        "steps": STEPS,
        "cases": CASES,
        "backend": "numba",
        "pressure_tolerance": PRESSURE_TOLERANCE,
        "pressure_max_iterations": PRESSURE_MAX_ITERATIONS,
        "residual_check_interval": RESIDUAL_CHECK_INTERVAL,
        "record_interval": 1,
        "stop_on_steady": False,
        "timing": (
            "perf_counter around solve(); Numba warmup excluded; export excluded"
        ),
    }

    with (OUTPUT / "benchmark_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("\nBenchmark concluido.")
    print(f"Resultados em: {OUTPUT}")

    # Verificacoes minimas de integridade.
    for row in rows:
        if row["steps"] != STEPS:
            raise SystemExit("Erro: numero de passos inesperado.")

        if row["poisson_residual_max"] > PRESSURE_TOLERANCE:
            raise SystemExit("Erro: tolerancia de Poisson nao atendida.")

        if not np.isfinite(row["divergence_rms_max"]):
            raise SystemExit("Erro: divergencia nao finita.")


if __name__ == "__main__":
    main()
