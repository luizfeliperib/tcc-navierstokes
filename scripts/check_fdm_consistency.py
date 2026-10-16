"""Executa a cavidade e registra a consistência centro-face."""

import csv
import sys

import numpy as np

from navierstokes.config import SimulationConfig
from navierstokes.fdm.consistency import calculate_consistency
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.time_step import FlowState, advance_one_step


def main() -> None:
    config = SimulationConfig(
        nx=8,
        ny=8,
        dt=0.001,
        reynolds=100.0,
    )
    grid = create_grid(config)

    state = FlowState(
        u=np.zeros(grid.scalar_shape),
        v=np.zeros(grid.scalar_shape),
        u_faces=np.zeros(grid.u_face_shape),
        v_faces=np.zeros(grid.v_face_shape),
        pressure=np.zeros(grid.scalar_shape),
    )

    writer = csv.writer(sys.stdout)

    writer.writerow(
        [
            "step",
            "time",
            "u_total",
            "u_interior",
            "u_boundary",
            "v_total",
            "v_interior",
            "v_boundary",
            "u_relative_percent",
            "v_relative_percent",
        ]
    )

    for step in range(1, 1001):
        result = advance_one_step(
            state,
            grid,
            config,
            pressure_tolerance=1e-9,
            pressure_max_iterations=10_000,
        )

        state = result.state
        diagnostics = calculate_consistency(state, grid)

        writer.writerow(
            [
                step,
                step * config.dt,
                diagnostics.u_total,
                diagnostics.u_interior,
                diagnostics.u_boundary,
                diagnostics.v_total,
                diagnostics.v_interior,
                diagnostics.v_boundary,
                diagnostics.u_relative_percent,
                diagnostics.v_relative_percent,
            ]
        )


if __name__ == "__main__":
    main()
