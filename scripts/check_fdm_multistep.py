
"""Executa um diagnóstico exploratório de 100 passos."""

from navierstokes.config import SimulationConfig
from navierstokes.fdm.diagnostics import calculate_diagnostics
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.time_step import FlowState, advance_one_step


def main():
    config = SimulationConfig(
        nx=8,
        ny=8,
        dt=0.001,
        reynolds=100.0,
    )
    grid = create_grid(config)

    state = FlowState(
        u=grid.zeros_scalar(),
        v=grid.zeros_scalar(),
        u_faces=grid.zeros_u_faces(),
        v_faces=grid.zeros_v_faces(),
        pressure=grid.zeros_scalar(),
    )

    print(
        "step,time,div_rms,center_face_u_rms,"
        "center_face_v_rms,max_u,max_v,energy,"
        "poisson_iterations,poisson_residual"
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
        diag = calculate_diagnostics(state, grid)
        pressure = result.pressure_result

        print(
            f"{step},{step * config.dt:.6f},"
            f"{diag.divergence_rms:.6e},"
            f"{diag.center_face_u_rms:.6e},"
            f"{diag.center_face_v_rms:.6e},"
            f"{diag.max_abs_u:.6e},"
            f"{diag.max_abs_v:.6e},"
            f"{diag.kinetic_energy_mean:.6e},"
            f"{pressure.iterations},"
            f"{pressure.residual:.6e}"
        )


if __name__ == "__main__":
    main()
