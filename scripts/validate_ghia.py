
"""FDM-07.7.2: comparacao dos perfis centrais com Ghia (1982)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from navierstokes.validation.ghia_reference import ghia_re100


def read_profile(path, coordinate, velocity):
    data = np.genfromtxt(
        path, delimiter=",", names=True, encoding="utf-8"
    )

    x = np.atleast_1d(data[coordinate])
    v = np.atleast_1d(data[velocity])

    order = np.argsort(x)
    return x[order], v[order]


def read_case(folder):
    folder = Path(folder)

    with (folder / "metadata.json").open(
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    config = metadata["configuration"]

    if not np.isclose(config["reynolds"], 100):
        raise ValueError(
            f"{folder}: a referencia selecionada exige Re=100"
        )

    yu, u = read_profile(
        folder / "centerline_u.csv", "y", "u"
    )
    xv, v = read_profile(
        folder / "centerline_v.csv", "x", "v"
    )

    lx = config["lx"]
    ly = config["ly"]
    lid = config["lid_velocity"]

    if lx <= 0 or ly <= 0 or lid == 0:
        raise ValueError("Dimensoes ou velocidade invalidas")

    # Normalizacao adimensional
    yu = yu / ly
    xv = xv / lx
    u = u / lid
    v = v / lid

    # Acrescenta os valores prescritos nas paredes.
    yu = np.r_[0.0, yu, 1.0]
    u = np.r_[0.0, u, 1.0]

    xv = np.r_[0.0, xv, 1.0]
    v = np.r_[0.0, v, 0.0]

    return {
        "folder": folder,
        "metadata": metadata,
        "yu": yu,
        "u": u,
        "xv": xv,
        "v": v,
    }


def evaluate_profile(coordinates, values, ref_x, ref_values):
    predicted = np.interp(ref_x, coordinates, values)
    errors = predicted - ref_values

    metrics = {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "max_abs_error": float(np.max(np.abs(errors))),
    }

    return predicted, errors, metrics


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputs", nargs="+", type=Path, required=True
    )
    parser.add_argument(
        "--labels", nargs="+", required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fdm_077_ghia"),
    )

    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        parser.error("Informe um label para cada pasta de entrada")

    args.output.mkdir(parents=True, exist_ok=True)

    reference = ghia_re100()
    cases = [
        (label, read_case(folder))
        for label, folder in zip(args.labels, args.inputs)
    ]

    specifications = [
        {
            "name": "u",
            "filename": "ghia_u_centerline.png",
            "coordinate": "yu",
            "velocity": "u",
            "ref_x": reference.y_over_l,
            "ref_v": reference.u_over_u_lid,
            "xlabel": "u / U_tampa",
            "ylabel": "y / L",
        },
        {
            "name": "v",
            "filename": "ghia_v_centerline.png",
            "coordinate": "xv",
            "velocity": "v",
            "ref_x": reference.x_over_l,
            "ref_v": reference.v_over_u_lid,
            "xlabel": "x / L",
            "ylabel": "v / U_tampa",
        },
    ]

    summary = []
    point_rows = []

    for spec in specifications:
        fig, ax = plt.subplots(figsize=(7, 6))

        if spec["name"] == "u":
            ax.scatter(
                spec["ref_v"], spec["ref_x"],
                marker="o", color="black",
                label="Ghia et al. (1982)", zorder=5
            )
        else:
            ax.scatter(
                spec["ref_x"], spec["ref_v"],
                marker="o", color="black",
                label="Ghia et al. (1982)", zorder=5
            )

        for label, case in cases:
            coordinates = case[spec["coordinate"]]
            velocities = case[spec["velocity"]]

            predicted, errors, metrics = evaluate_profile(
                coordinates,
                velocities,
                spec["ref_x"],
                spec["ref_v"],
            )

            metadata = case["metadata"]
            result = metadata["result"]

            summary.append({
                "case": label,
                "component": spec["name"],
                "final_time": result["final_time"],
                "converged": result["converged"],
                "steady_residual": result["steady_residual"],
                **metrics,
            })

            for x, ref_value, value, error in zip(
                spec["ref_x"], spec["ref_v"],
                predicted, errors
            ):
                point_rows.append({
                    "case": label,
                    "component": spec["name"],
                    "coordinate": float(x),
                    "ghia": float(ref_value),
                    "fdm": float(value),
                    "error": float(error),
                    "abs_error": float(abs(error)),
                })

            if spec["name"] == "u":
                ax.plot(
                    velocities, coordinates,
                    label=f"FDM {label}"
                )
            else:
                ax.plot(
                    coordinates, velocities,
                    label=f"FDM {label}"
                )

        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(spec["ylabel"])
        ax.set_title(
            f"Re = 100 | Perfil central {spec['name']}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            args.output / spec["filename"], dpi=180
        )
        plt.close(fig)

    write_csv(
        args.output / "ghia_metrics.csv",
        summary,
        [
            "case", "component", "final_time",
            "converged", "steady_residual",
            "rmse", "mae", "max_abs_error",
        ],
    )

    write_csv(
        args.output / "ghia_pointwise.csv",
        point_rows,
        [
            "case", "component", "coordinate",
            "ghia", "fdm", "error", "abs_error",
        ],
    )

    print("\n=== COMPARACAO COM GHIA (Re=100) ===")

    for row in summary:
        print(
            f"{row['case']} | {row['component']} | "
            f"RMSE={row['rmse']:.6f} | "
            f"erro max={row['max_abs_error']:.6f} | "
            f"estacionario={row['converged']}"
        )

    print(f"\nResultados: {args.output.resolve()}")


if __name__ == "__main__":
    main()
