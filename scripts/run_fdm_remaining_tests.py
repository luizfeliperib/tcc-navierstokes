
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# CAMPANHA FINAL — NAVIER-STOKES 2D / FDM
# 7 experimentos pendentes (80x80 executado separadamente)
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_fdm_validation.py"
OUTPUT_ROOT = ROOT / "results" / "fdm_campanha_final"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Todos os testes de Re=100 terminam em T=16.
# Todos os testes de Re=400 terminam em T=40.
#
# Sem parada antecipada por estacionariedade:
# assim, cada serie temporal usa o mesmo tempo fisico.

TESTES = [
    {"id": "01", "re": 100, "n": 16, "dt": 0.005, "steps": 3200},
    {"id": "02", "re": 100, "n": 16, "dt": 0.010, "steps": 1600},
    {"id": "10", "re": 400, "n": 16, "dt": 0.005, "steps": 8000},
    {"id": "11", "re": 400, "n": 32, "dt": 0.001, "steps": 40000},
    {"id": "12", "re": 400, "n": 32, "dt": 0.002, "steps": 20000},
    {"id": "13", "re": 400, "n": 32, "dt": 0.005, "steps": 8000},
    {"id": "14", "re": 400, "n": 32, "dt": 0.010, "steps": 4000},
]

if not RUNNER.is_file():
    raise FileNotFoundError(f"Runner nao encontrado: {RUNNER}")

relatorio = []

for indice, teste in enumerate(TESTES, start=1):
    re = teste["re"]
    n = teste["n"]
    dt = teste["dt"]
    steps = teste["steps"]

    nome = (
        f"cfg_{teste['id']}_re{re}_"
        f"{n}x{n}_dt{dt:.4f}".replace(".", "")
    )

    pasta = OUTPUT_ROOT / nome
    log_path = OUTPUT_ROOT / f"{nome}.log"
    metadata_path = pasta / "metadata.json"

    print(f"\n{'=' * 65}", flush=True)
    print(f"TESTE {indice}/{len(TESTES)}: {nome}", flush=True)
    print(f"Re={re} | Malha={n}x{n} | dt={dt} | T={dt * steps:g}",
          flush=True)
    print(f"{'=' * 65}", flush=True)

    # Evita repetir execucoes ja exportadas.
    if metadata_path.is_file():
        try:
            with metadata_path.open(encoding="utf-8") as f:
                metadata = json.load(f)

            config = metadata["configuration"]
            result = metadata["result"]
            settings = metadata["execution_settings"]

            valido = (
                float(config["reynolds"]) == re
                and int(config["nx"]) == n
                and int(config["ny"]) == n
                and abs(float(config["dt"]) - dt) < 1e-12
                and int(result["steps"]) == steps
                and settings["face_flux_mode"] == "reconstructed"
            )

            if valido:
                print("Resultado existente compativel. Pulando.")
                relatorio.append({
                    "teste": nome,
                    "status": "JA_EXISTE",
                    "segundos": "",
                    "detalhe": "Resultado previamente exportado",
                })
                continue
        except (KeyError, ValueError, TypeError, OSError):
            pass

    comando = [
        sys.executable,
        str(RUNNER),
        "--reynolds", str(re),
        "--nx", str(n),
        "--ny", str(n),
        "--dt", str(dt),
        "--steps", str(steps),
        "--backend", "numba",
        "--omega", "1.9",
        "--pressure-tolerance", "1e-9",
        "--pressure-max-iterations", "20000",
        "--residual-check-interval", "10",
        "--steady-tolerance", "1e-5",
        "--steady-consecutive-steps", "20",
        "--min-steps", "100",
        "--record-interval", "100",
        "--face-flux-mode", "reconstructed",
        "--output", str(pasta),
    ]

    inicio = time.perf_counter()
    status = "FALHOU"
    detalhe = ""

    try:
        with log_path.open("w", encoding="utf-8") as log:
            processo = subprocess.run(
                comando,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if processo.returncode == 0 and metadata_path.is_file():
            status = "OK"
        else:
            detalhe = (
                f"Codigo de saida: {processo.returncode}. "
                f"Consulte {log_path.name}"
            )

    except Exception as erro:
        detalhe = str(erro)

    duracao = time.perf_counter() - inicio

    relatorio.append({
        "teste": nome,
        "status": status,
        "segundos": round(duracao, 2),
        "detalhe": detalhe,
    })

    print(f"Status: {status}", flush=True)
    print(f"Tempo computacional: {duracao:.2f} s", flush=True)

    if status != "OK":
        print(f"Detalhes: {detalhe}", flush=True)

# ============================================================
# RELATORIO FINAL
# ============================================================

csv_path = OUTPUT_ROOT / "resumo_execucoes.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["teste", "status", "segundos", "detalhe"],
        delimiter=";",
    )
    writer.writeheader()
    writer.writerows(relatorio)

print("\n" + "=" * 65)
print("RESUMO FINAL")
print("=" * 65)

for item in relatorio:
    print(f"{item['teste']}: {item['status']}")

print(f"\nRelatorio salvo em: {csv_path}")
print("Campanha encerrada.")
