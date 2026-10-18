
"""Dados de referência de Ghia, Ghia e Shin (1982).

Re = 100, cavidade quadrada com tampa móvel.

Fonte:
Journal of Computational Physics, 48 (1982), 387-411.
DOI: 10.1016/0021-9991(82)90058-4

Tabela I: u/U ao longo de x/L = 0.5.
Tabela II: v/U ao longo de y/L = 0.5.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GhiaReference:
    reynolds: int
    y_over_l: NDArray[np.float64]
    u_over_u_lid: NDArray[np.float64]
    x_over_l: NDArray[np.float64]
    v_over_u_lid: NDArray[np.float64]


def ghia_re100() -> GhiaReference:
    """Retorna os perfis de referência para Re = 100."""

    # Tabela I: u na linha vertical x/L = 0.5
    yu = np.array([
        [1.0000,  1.00000],
        [0.9766,  0.84123],
        [0.9688,  0.78871],
        [0.9609,  0.73722],
        [0.9531,  0.68717],
        [0.8516,  0.23151],
        [0.7344,  0.00332],
        [0.6172, -0.13641],
        [0.5000, -0.20581],
        [0.4531, -0.21090],
        [0.2813, -0.15662],
        [0.1719, -0.10150],
        [0.1016, -0.06434],
        [0.0703, -0.04775],
        [0.0625, -0.04192],
        [0.0547, -0.03717],
        [0.0000,  0.00000],
    ], dtype=np.float64)

    # Tabela II: v na linha horizontal y/L = 0.5
    xv = np.array([
        [1.0000,  0.00000],
        [0.9688, -0.05906],
        [0.9609, -0.07391],
        [0.9531, -0.08864],
        [0.9453, -0.10313],
        [0.9063, -0.16914],
        [0.8594, -0.22445],
        [0.8047, -0.24533],
        [0.5000,  0.05454],
        [0.2344,  0.17527],
        [0.2266,  0.17507],
        [0.1563,  0.16077],
        [0.0938,  0.12317],
        [0.0781,  0.10890],
        [0.0703,  0.10091],
        [0.0625,  0.09233],
        [0.0000,  0.00000],
    ], dtype=np.float64)

    # Ordenacao crescente das coordenadas
    yu = yu[np.argsort(yu[:, 0])]
    xv = xv[np.argsort(xv[:, 0])]

    return GhiaReference(
        reynolds=100,
        y_over_l=yu[:, 0].copy(),
        u_over_u_lid=yu[:, 1].copy(),
        x_over_l=xv[:, 0].copy(),
        v_over_u_lid=xv[:, 1].copy(),
    )


if __name__ == "__main__":
    import csv
    from pathlib import Path

    reference = ghia_re100()

    output = Path("results/ghia_reference")
    output.mkdir(parents=True, exist_ok=True)

    profiles = (
        (
            "ghia_re100_u_centerline.csv",
            reference.y_over_l,
            reference.u_over_u_lid,
            "y_over_l",
            "u_over_u_lid",
        ),
        (
            "ghia_re100_v_centerline.csv",
            reference.x_over_l,
            reference.v_over_u_lid,
            "x_over_l",
            "v_over_u_lid",
        ),
    )

    for filename, coordinates, values, coord_name, value_name in profiles:
        with (output / filename).open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow([coord_name, value_name])
            writer.writerows(zip(coordinates, values))

    print(f"Dados exportados para: {output.resolve()}")
