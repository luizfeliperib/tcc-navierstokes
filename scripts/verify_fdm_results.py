"""FDM-07.4: verificação de resultados exportados."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REQUIRED_FILES = (
    "fields.npz",
    "history.csv",
    "metadata.json",
    "centerline_u.csv",
    "centerline_v.csv",
)

FIELD_NAMES = (
    "x",
    "y",
    "u",
    "v",
    "pressure",
    "u_faces",
    "v_faces",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_results(directory):
    """Verifica arquivos, dimensões e consistência numérica."""

    directory = Path(directory)

    for filename in REQUIRED_FILES:
        check(
            (directory / filename).is_file(),
            f"Arquivo ausente: {filename}",
        )

    with (directory / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)

    check(
        metadata.get("schema_version") == "1.0",
        "Versão de metadados incompatível.",
    )

    config = metadata["configuration"]
    result = metadata["result"]
    grid_info = metadata["grid"]

    nx = int(config["nx"])
    ny = int(config["ny"])
    dt = float(config["dt"])

    check(nx >= 2 and ny >= 2, "Malha inválida.")
    check(np.isfinite(dt) and dt > 0, "dt inválido.")

    check(
        grid_info["nx"] == nx and grid_info["ny"] == ny,
        "Metadados da malha inconsistentes.",
    )

    expected_status = "steady" if result["converged"] else "transient"

    check(
        metadata["flow_status"] == expected_status,
        "Estado do escoamento inconsistente.",
    )

    expected_reason = "steady_tolerance" if result["converged"] else "max_steps"

    check(
        result["stop_reason"] == expected_reason,
        "Motivo de parada inconsistente.",
    )

    steps = int(result["steps"])

    check(
        1 <= steps <= int(config["max_iterations"]),
        "Número de passos inválido.",
    )

    check(
        np.isclose(
            float(result["final_time"]),
            steps * dt,
            rtol=1e-12,
            atol=1e-14,
        ),
        "Tempo físico final inconsistente.",
    )

    with np.load(
        directory / "fields.npz",
        allow_pickle=False,
    ) as data:
        check(
            all(name in data.files for name in FIELD_NAMES),
            "Campos obrigatórios ausentes.",
        )

        fields = {name: data[name].copy() for name in FIELD_NAMES}

    x = fields["x"]
    y = fields["y"]
    u = fields["u"]
    v = fields["v"]

    shapes = {
        "x": (nx,),
        "y": (ny,),
        "u": (ny, nx),
        "v": (ny, nx),
        "pressure": (ny, nx),
        "u_faces": (ny, nx + 1),
        "v_faces": (ny + 1, nx),
    }

    for name, shape in shapes.items():
        check(
            fields[name].shape == shape,
            f"Dimensão inválida: {name}",
        )
        check(
            np.all(np.isfinite(fields[name])),
            f"Valores não finitos: {name}",
        )

    dx = float(grid_info["dx"])
    dy = float(grid_info["dy"])

    check(
        dx > 0 and dy > 0,
        "Espaçamento de malha inválido.",
    )

    np.testing.assert_allclose(
        x,
        (np.arange(nx) + 0.5) * dx,
        rtol=1e-12,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        y,
        (np.arange(ny) + 0.5) * dy,
        rtol=1e-12,
        atol=1e-14,
    )

    check(
        np.isclose(nx * dx, float(config["lx"]))
        and np.isclose(ny * dy, float(config["ly"])),
        "Dimensões físicas inconsistentes.",
    )

    history = read_csv(directory / "history.csv")

    check(bool(history), "Histórico vazio.")

    history_steps = np.array([int(row["step"]) for row in history])

    history_times = np.array([float(row["time"]) for row in history])

    check(
        history_steps[0] == 1 and history_steps[-1] == steps,
        "Passos do histórico inconsistentes.",
    )

    check(
        np.all(np.diff(history_steps) > 0),
        "Passos do histórico fora de ordem.",
    )

    np.testing.assert_allclose(
        history_times,
        history_steps * dt,
        rtol=1e-12,
        atol=1e-14,
    )

    numeric_columns = (
        "steady_residual",
        "poisson_residual",
        "poisson_iterations",
        "divergence_rms",
        "center_face_u_rms",
        "center_face_v_rms",
        "max_abs_u",
        "max_abs_v",
        "kinetic_energy_mean",
    )

    for column in numeric_columns:
        values = np.array([float(row[column]) for row in history])
        check(
            np.all(np.isfinite(values)),
            f"Histórico não finito: {column}",
        )

    np.testing.assert_allclose(
        float(history[-1]["steady_residual"]),
        float(result["steady_residual"]),
        rtol=1e-12,
        atol=1e-14,
    )

    u_rows = read_csv(directory / "centerline_u.csv")
    v_rows = read_csv(directory / "centerline_v.csv")

    check(
        len(u_rows) == ny,
        "Perfil vertical com tamanho incorreto.",
    )
    check(
        len(v_rows) == nx,
        "Perfil horizontal com tamanho incorreto.",
    )

    check(
        list(u_rows[0]) == ["y", "u"],
        "Cabeçalho incorreto em centerline_u.csv.",
    )
    check(
        list(v_rows[0]) == ["x", "v"],
        "Cabeçalho incorreto em centerline_v.csv.",
    )

    u_coords = np.array([float(row["y"]) for row in u_rows])
    u_values = np.array([float(row["u"]) for row in u_rows])

    v_coords = np.array([float(row["x"]) for row in v_rows])
    v_values = np.array([float(row["v"]) for row in v_rows])

    for name, values in (
        ("perfil u", u_values),
        ("perfil v", v_values),
    ):
        check(
            np.all(np.isfinite(values)),
            f"Valores não finitos no {name}.",
        )

    np.testing.assert_allclose(u_coords, y, rtol=0, atol=1e-14)
    np.testing.assert_allclose(v_coords, x, rtol=0, atol=1e-14)

    x_center = float(config["lx"]) / 2
    y_center = float(config["ly"]) / 2

    expected_u = np.array([np.interp(x_center, x, u[j, :]) for j in range(ny)])

    expected_v = np.array([np.interp(y_center, y, v[:, i]) for i in range(nx)])

    u_error = float(np.max(np.abs(u_values - expected_u)))
    v_error = float(np.max(np.abs(v_values - expected_v)))

    np.testing.assert_allclose(u_values, expected_u, rtol=1e-12, atol=1e-14)

    np.testing.assert_allclose(v_values, expected_v, rtol=1e-12, atol=1e-14)

    return {
        "status": "passed",
        "steps": steps,
        "flow_status": expected_status,
        "nx": nx,
        "ny": ny,
        "max_centerline_u_error": u_error,
        "max_centerline_v_error": v_error,
    }


def main():
    parser = argparse.ArgumentParser(description="Verifica resultados FDM exportados.")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    report = verify_results(args.input)

    print("\nVerificação FDM aprovada.")

    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
