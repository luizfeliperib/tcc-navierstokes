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
    """Exporta uma figura em PNG (300 dpi) e PDF vetorial."""
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_history(path: Path) -> dict[str, np.ndarray]:
    """Carrega as colunas relevantes do histórico temporal."""
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
        column: np.asarray([float(row[column]) for row in rows], dtype=float)
        for column in columns
    }
    if any(not np.all(np.isfinite(values)) for values in history.values()):
        raise ValueError("O histórico contém valores não finitos.")
    return history


def plot_results(input_dir: Path) -> None:
    """Gera quatro figuras a partir dos dados já exportados pelo solver."""
    input_dir = Path(input_dir)

    with (input_dir / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)

    with np.load(input_dir / "fields.npz", allow_pickle=False) as data:
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
        raise ValueError("As coordenadas x e y devem ser unidimensionais.")
    if x.size != nx or y.size != ny or u.shape != (ny, nx) or v.shape != (ny, nx):
        raise ValueError("As dimensões dos campos são incompatíveis com a malha.")
    if nx < 2 or ny < 2 or not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0):
        raise ValueError("A malha deve possuir coordenadas crescentes e ao menos dois centros.")
    if not all(np.all(np.isfinite(array)) for array in (x, y, u, v)):
        raise ValueError("Os campos ou as coordenadas contêm valores não finitos.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("A tolerância de estacionariedade deve ser positiva e finita.")

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