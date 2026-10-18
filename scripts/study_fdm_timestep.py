"""FDM-07.7 - Estudo de sensibilidade ao passo temporal.

Compara simulacoes com diferentes dt e mesmo tempo fisico final.
Nao altera o solver existente.

Saidas:
    summary.csv
    field_comparison.csv
    study_metadata.json
    <caso>/fields.npz
    <caso>/history.csv
    <caso>/metadata.json
    <caso>/centerline_u.csv
    <caso>/centerline_v.csv
"""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from run_fdm_validation import export_results

from navierstokes.config import SimulationConfig
from navierstokes.fdm.solver import solve


def rms(values):
    """Norma RMS de um campo."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def write_csv(path, rows, columns):
    """Exporta uma tabela CSV."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Estudo de sensibilidade temporal do FDM."
    )

    parser.add_argument("--reynolds", type=float, default=100.0)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)

    parser.add_argument(
        "--dts",
        type=float,
        nargs="+",
        default=[0.001, 0.002, 0.005, 0.010],
    )

    parser.add_argument(
        "--final-time",
        type=float,
        default=10.0,
    )

    parser.add_argument("--backend", choices=("python", "numba"), default="numba")
    parser.add_argument("--omega", type=float, default=1.9)

    parser.add_argument("--pressure-tolerance", type=float, default=1e-9)
    parser.add_argument("--pressure-max-iterations", type=int, default=20000)
    parser.add_argument("--residual-check-interval", type=int, default=10)

    parser.add_argument("--steady-tolerance", type=float, default=1e-5)
    parser.add_argument("--record-interval", type=int, default=100)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fdm_timestep_study"),
    )

    return parser


def run_case(label, dt, steps, args):
    """Executa um caso, registrando tempo e resultados."""

    config = SimulationConfig(
        reynolds=args.reynolds,
        nx=args.nx,
        ny=args.ny,
        dt=dt,
        tolerance=args.steady_tolerance,
        max_iterations=steps,
    )

    settings = {
        "backend": args.backend,
        "omega": args.omega,
        "pressure_tolerance": args.pressure_tolerance,
        "pressure_max_iterations": args.pressure_max_iterations,
        "residual_check_interval": args.residual_check_interval,
        "record_interval": args.record_interval,
        "stop_on_steady": False,
        "steady_consecutive_steps": 20,
        "min_steps": 100,
    }

    output_dir = args.output / label

    print(
        f"\n[{label}] dt={dt:g} | passos={steps} | T={dt * steps:g}",
        flush=True,
    )

    start = perf_counter()

    try:
        result = solve(
            config,
            backend=args.backend,
            pressure_tolerance=args.pressure_tolerance,
            pressure_max_iterations=args.pressure_max_iterations,
            residual_check_interval=args.residual_check_interval,
            stop_on_steady=False,
            record_interval=args.record_interval,
            omega=args.omega,
        )

        elapsed = perf_counter() - start

        export_results(
            result=result,
            config=config,
            output_dir=output_dir,
            elapsed=elapsed,
            settings=settings,
        )

        diagnostics = result.history[-1].diagnostics

        summary = {
            "case": label,
            "status": "ok",
            "dt": dt,
            "steps": result.steps,
            "final_time": result.final_time,
            "elapsed_seconds": elapsed,
            "seconds_per_step": elapsed / result.steps,
            "steady_residual": result.steady_residual,
            "poisson_residual": result.history[-1].poisson_residual,
            "poisson_iterations": result.history[-1].poisson_iterations,
            "divergence_rms": diagnostics.divergence_rms,
            "center_face_u_rms": diagnostics.center_face_u_rms,
            "center_face_v_rms": diagnostics.center_face_v_rms,
            "kinetic_energy_mean": diagnostics.kinetic_energy_mean,
            "error": "",
        }

        print(
            f"[{label}] OK | "
            f"tempo={elapsed:.2f}s | "
            f"Rsteady={result.steady_residual:.3e} | "
            f"div={diagnostics.divergence_rms:.3e}",
            flush=True,
        )

        return summary, result

    except (ValueError, RuntimeError, FloatingPointError) as exc:
        elapsed = perf_counter() - start

        summary = {
            "case": label,
            "status": "failed",
            "dt": dt,
            "steps": steps,
            "final_time": dt * steps,
            "elapsed_seconds": elapsed,
            "seconds_per_step": "",
            "steady_residual": "",
            "poisson_residual": "",
            "poisson_iterations": "",
            "divergence_rms": "",
            "center_face_u_rms": "",
            "center_face_v_rms": "",
            "kinetic_energy_mean": "",
            "error": f"{type(exc).__name__}: {exc}",
        }

        print(
            f"[{label}] FALHOU: {summary['error']}",
            flush=True,
        )

        return summary, None


def compare_fields(reference, candidate, label, dt):
    """Compara campos na mesma malha e tempo fisico."""

    ref = reference.state
    other = candidate.state

    return {
        "case": label,
        "dt": dt,
        "u_rms_difference": rms(other.u - ref.u),
        "v_rms_difference": rms(other.v - ref.v),
        "u_max_difference": float(np.max(np.abs(other.u - ref.u))),
        "v_max_difference": float(np.max(np.abs(other.v - ref.v))),
        "u_faces_rms_difference": rms(other.u_faces - ref.u_faces),
        "v_faces_rms_difference": rms(other.v_faces - ref.v_faces),
    }


def main():
    args = build_parser().parse_args()

    if not np.isfinite(args.final_time) or args.final_time <= 0:
        raise ValueError("final-time deve ser positivo e finito")

    if args.record_interval < 1:
        raise ValueError("record-interval deve ser positivo")

    if not args.dts or any(not np.isfinite(dt) or dt <= 0 for dt in args.dts):
        raise ValueError("Todos os dt devem ser positivos e finitos")

    dts = sorted(set(args.dts))

    # Garante o mesmo tempo fisico final em todos os casos.
    cases = []

    for dt in dts:
        exact_steps = args.final_time / dt
        steps = round(exact_steps)

        if steps < 1 or not np.isclose(
            steps * dt,
            args.final_time,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"T={args.final_time:g} nao e multiplo inteiro de dt={dt:g}"
            )

        cases.append((dt, steps))

    args.output.mkdir(parents=True, exist_ok=True)

    summaries = []
    results = {}

    print("\n=== ESTUDO DE SENSIBILIDADE TEMPORAL ===")
    print(f"Reynolds: {args.reynolds:g}")
    print(f"Malha: {args.nx}x{args.ny}")
    print(f"Tempo fisico final: {args.final_time:g}")
    print(f"Passos temporais: {dts}")
    print("Parada estacionaria desabilitada.")
    print()

    for index, (dt, steps) in enumerate(cases):
        label = f"case_{index + 1:02d}_dt_{dt:g}"

        summary, result = run_case(label, dt, steps, args)

        summaries.append(summary)

        if result is not None:
            results[label] = result

        # Atualiza o resumo mesmo se algum teste falhar.
        write_csv(
            args.output / "summary.csv",
            summaries,
            list(summaries[0].keys()),
        )

    # O menor dt e a referencia numérica do experimento.
    reference_label = f"case_01_dt_{cases[0][0]:g}"

    comparisons = []

    if reference_label in results:
        reference = results[reference_label]

        for summary in summaries:
            label = summary["case"]

            if label not in results:
                continue

            comparisons.append(
                compare_fields(
                    reference,
                    results[label],
                    label,
                    summary["dt"],
                )
            )

        if comparisons:
            write_csv(
                args.output / "field_comparison.csv",
                comparisons,
                list(comparisons[0].keys()),
            )
    else:
        print("\nATENCAO: menor dt falhou. Nao foi possivel comparar os campos.")

    metadata = {
        "study": "FDM-07.7 temporal sensitivity",
        "reference_case": reference_label,
        "reference_is_exact": False,
        "configuration": {
            "reynolds": args.reynolds,
            "nx": args.nx,
            "ny": args.ny,
            "final_time": args.final_time,
            "dts": dts,
            "backend": args.backend,
            "omega": args.omega,
            "pressure_tolerance": args.pressure_tolerance,
            "pressure_max_iterations": (args.pressure_max_iterations),
            "residual_check_interval": (args.residual_check_interval),
        },
        "summary": summaries,
        "field_comparison": comparisons,
    }

    with (args.output / "study_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== RESUMO FINAL ===")

    for summary in summaries:
        print(
            f"{summary['case']}: "
            f"{summary['status']} | "
            f"{summary['elapsed_seconds']:.2f}s"
        )

    print(f"\nResultados em: {args.output.resolve()}")


if __name__ == "__main__":
    main()
