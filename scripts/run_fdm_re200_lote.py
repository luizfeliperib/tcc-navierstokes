"""Executa os sete casos pendentes de Re=200 com malhas menores que 96x96.

Local no repositório: scripts/run_fdm_re200_lote.py
Execução na raiz: python scripts/run_fdm_re200_lote.py
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_fdm_validation.py"
OUTPUT_ROOT = ROOT / "results" / "fdm_24_cases" / "re200"

REYNOLDS = 200
FINAL_TIME = 40.0

CASES = (
    (16, 0.005),
    (32, 0.001),
    (32, 0.002),
    (32, 0.005),
    (32, 0.010),
    (64, 0.001),
    (64, 0.002),
)

EXPECTED_SETTINGS = {
    "backend": "numba",
    "pressure_tolerance": 1e-9,
    "pressure_max_iterations": 20000,
    "residual_check_interval": 10,
    "omega": 1.9,
    "face_flux_mode": "reconstructed",
    "stop_on_steady": False,
}


def case_name(n, dt):
    return f"re{REYNOLDS}_n{n}_dt{dt:.4f}".replace(".", "p")


def expected_steps(dt):
    steps = round(FINAL_TIME / dt)
    if abs(steps * dt - FINAL_TIME) > 1e-9:
        raise ValueError(f"T={FINAL_TIME} não é múltiplo de dt={dt}")
    return steps


def existing_is_valid(folder, n, dt, steps):
    required = (
        "metadata.json",
        "fields.npz",
        "centerline_u.csv",
        "centerline_v.csv",
        "history.csv",
    )
    if not all((folder / item).is_file() for item in required):
        return False

    try:
        with (folder / "metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)

        config = metadata["configuration"]
        result = metadata["result"]
        settings = metadata["execution_settings"]

        return (
            metadata.get("problem") == "lid_driven_cavity"
            and metadata.get("method") == "FDM"
            and abs(float(config["reynolds"]) - REYNOLDS) < 1e-12
            and int(config["nx"]) == n
            and int(config["ny"]) == n
            and abs(float(config["dt"]) - dt) < 1e-12
            and int(result["steps"]) == steps
            and abs(float(result["final_time"]) - FINAL_TIME) < 1e-8
            and all(
                (abs(float(settings[key]) - value) < 1e-12)
                if isinstance(value, float)
                else settings[key] == value
                for key, value in EXPECTED_SETTINGS.items()
            )
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reexecuta mesmo quando já existem resultados compatíveis.",
    )
    args = parser.parse_args()

    if not RUNNER.is_file():
        raise FileNotFoundError(f"Executor não encontrado: {RUNNER}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = []
    started = time.perf_counter()

    for index, (n, dt) in enumerate(CASES, start=1):
        steps = expected_steps(dt)
        name = case_name(n, dt)
        folder = OUTPUT_ROOT / name
        log_path = OUTPUT_ROOT / f"{name}.log"

        print(
            f"\n[{index}/{len(CASES)}] {name} | {steps} passos | T={FINAL_TIME:g}",
            flush=True,
        )

        if not args.force and existing_is_valid(folder, n, dt, steps):
            print("  Resultado compatível encontrado: ignorando.", flush=True)
            report.append(
                {
                    "caso": name,
                    "status": "JA_EXISTE",
                    "segundos": "",
                    "log": str(log_path),
                }
            )
            continue

        if folder.exists() and any(folder.iterdir()):
            print(
                "  AVISO: pasta já contém arquivos; serão atualizados pelo executor.",
                flush=True,
            )

        command = [
            sys.executable,
            str(RUNNER),
            "--reynolds",
            str(REYNOLDS),
            "--nx",
            str(n),
            "--ny",
            str(n),
            "--dt",
            str(dt),
            "--steps",
            str(steps),
            "--backend",
            "numba",
            "--omega",
            "1.9",
            "--pressure-tolerance",
            "1e-9",
            "--pressure-max-iterations",
            "20000",
            "--residual-check-interval",
            "10",
            "--steady-tolerance",
            "1e-5",
            "--steady-consecutive-steps",
            "20",
            "--min-steps",
            "100",
            "--record-interval",
            "100",
            "--face-flux-mode",
            "reconstructed",
            "--output",
            str(folder),
        ]

        case_started = time.perf_counter()

        try:
            with log_path.open("w", encoding="utf-8") as logfile:
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            duration = time.perf_counter() - case_started
            valid = existing_is_valid(folder, n, dt, steps)
            status = "OK" if process.returncode == 0 and valid else "FALHOU"

            print(
                f"  {status} | {duration:.1f} s | log: {log_path}",
                flush=True,
            )

            report.append(
                {
                    "caso": name,
                    "status": status,
                    "segundos": round(duration, 2),
                    "log": str(log_path),
                }
            )

            if status != "OK":
                print(
                    "  Interrompendo o lote para evitar resultados incompletos.",
                    flush=True,
                )
                break

        except (OSError, subprocess.SubprocessError) as exc:
            print(f"  FALHOU: {exc}", flush=True)
            report.append(
                {
                    "caso": name,
                    "status": "FALHOU",
                    "segundos": "",
                    "log": str(log_path),
                }
            )
            break

    summary_path = OUTPUT_ROOT / "resumo_lote.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("caso", "status", "segundos", "log"),
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(report)

    print(f"\nResumo: {summary_path}")
    print(f"Tempo total: {(time.perf_counter() - started) / 60:.1f} min")

    return 1 if any(item["status"] == "FALHOU" for item in report) else 0


if __name__ == "__main__":
    sys.exit(main())
