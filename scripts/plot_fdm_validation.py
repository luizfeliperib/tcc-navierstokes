"""FDM-07.1: visualizações exploratórias da cavidade com tampa móvel."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, output_dir: Path, name: str) -> None:
    """Exporta figura em PNG e PDF e libera a memória."""
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_history(path: Path) -> dict[str, np.ndarray]:
    """Lê as colunas numéricas do histórico."""
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("O arquivo history.csv está vazio.")

    columns = (
        "time",
        "steady_residual",
        "poisson_residual",
        "poisson_iterations",
        "divergence_rms",
        "kinetic_energy_mean",
    )

    history = {
        column: np.asarray(
            [float(row[column]) for row in rows],
            dtype=float,
        )
        for column in columns
    }

    if any(not np.all(np.isfinite(values)) for values in history.values()):
        raise ValueError("O histórico contém valores não finitos.")

    return history


def plot_results(input_dir: Path) -> None:
    """Gera quatro figuras a partir dos resultados exportados."""
    input_dir = Path(input_dir)

    with (input_dir / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)

    with np.load(
        input_dir / "fields.npz",
        allow_pickle=False,
    ) as data:
        x = data["x"].copy()
        y = data["y"].copy()
        u = data["u"].copy()
        v = data["v"].copy()

    history = load_history(input_dir / "history.csv")
    config = metadata["configuration"]
    result_info = metadata["result"]

    nx = int(config["nx"])
    ny = int(config["ny"])
    reynolds = float(config["reynolds"])
    final_time = float(result_info["final_time"])
    tolerance = float(config["tolerance"])

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("As coordenadas devem ser unidimensionais.")

    if x.size != nx or y.size != ny or u.shape != (ny, nx) or v.shape != (ny, nx):
        raise ValueError("Dimensões incompatíveis com a malha.")

    if nx < 2 or ny < 2 or not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0):
        raise ValueError("A malha deve ter centros crescentes.")

    if not all(np.all(np.isfinite(array)) for array in (x, y, u, v)):
        raise ValueError("Campos ou coordenadas não finitos.")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerância inválida.")

    output_dir = input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    title_suffix = f"Re={reynolds:g} | Malha {nx}×{ny} | t={final_time:.3f}"

    flow_status = (
        "Critério estacionário atingido"
        if result_info["converged"]
        else "Critério estacionário não atingido"
    )

    xx, yy = np.meshgrid(x, y, indexing="xy")
    speed = np.hypot(u, v)

    # 1. Magnitude da velocidade.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    mesh = ax.pcolormesh(
        xx,
        yy,
        speed,
        shading="auto",
        cmap="viridis",
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label=r"Magnitude da velocidade $|\mathbf{u}|$",
    )

    ax.set(
        title=f"Magnitude da velocidade\n{title_suffix}",
        xlabel="Coordenada x",
        ylabel="Coordenada y",
        aspect="equal",
    )

    save_figure(fig, output_dir, "velocity_magnitude")

    # 2. Linhas de corrente.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    stream_density = 0.7 if min(nx, ny) <= 16 else 1.2

    ax.streamplot(
        x,
        y,
        u,
        v,
        density=stream_density,
        color=speed,
        cmap="viridis",
        linewidth=1.0,
        arrowsize=1.0,
    )

    ax.set(
        title=f"Linhas de corrente\n{title_suffix}",
        xlabel="Coordenada x",
        ylabel="Coordenada y",
        aspect="equal",
    )

    save_figure(fig, output_dir, "streamlines")

    # 3. Campo vetorial.
    stride = max(1, min(nx, ny) // 20)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.quiver(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
        angles="xy",
        scale_units="xy",
        scale=None,
    )

    ax.set(
        title=f"Campo vetorial de velocidades\n{title_suffix}",
        xlabel="Coordenada x",
        ylabel="Coordenada y",
        aspect="equal",
    )

    save_figure(fig, output_dir, "velocity_vectors")

    # 4. Indicador de estacionariedade.
    fig, ax = plt.subplots(figsize=(7, 4))

    residual = history["steady_residual"]
    times = history["time"]
    positive = residual > 0

    ax.semilogy(
        times[positive],
        residual[positive],
        linewidth=1.8,
        label="Taxa RMS de variação da velocidade",
    )

    ax.axhline(
        tolerance,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label=f"Tolerância: {tolerance:.1e}",
    )

    ax.set(
        title=(f"Evolução do indicador de estacionariedade\n{title_suffix}"),
        xlabel="Tempo físico",
        ylabel="Indicador RMS / unidade de tempo",
    )

    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(loc="best", fontsize=9)

    ax.text(
        0.02,
        0.03,
        flow_status,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.9,
        },
    )

    save_figure(fig, output_dir, "steady_convergence")

    print(f"Figuras geradas em: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera figuras exploratórias FDM.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Diretório dos resultados exportados.",
    )

    args = parser.parse_args()
    plot_results(args.input)


if __name__ == "__main__":
    main()
