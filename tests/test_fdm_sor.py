"""FDM-07.5.3: testes do SOR e compatibilidade com GS."""

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.gauss_seidel import (
    solve_poisson_gauss_seidel,
)
from navierstokes.fdm.grid import create_grid
from navierstokes.fdm.poisson import (
    apply_poisson_system,
    assemble_poisson_coefficients,
)
from navierstokes.fdm.solver import solve


def make_problem(nx=8, ny=6):
    grid = create_grid(SimulationConfig(nx=nx, ny=ny))

    coefficients = assemble_poisson_coefficients(grid)

    rng = np.random.default_rng(42)

    phi_ref = rng.normal(size=grid.scalar_shape)
    phi_ref -= np.mean(phi_ref)

    rhs = apply_poisson_system(phi_ref, coefficients)

    return coefficients, phi_ref, rhs


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_omega_one_matches_default(backend):
    coefficients, _, rhs = make_problem()

    default = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        backend=backend,
        tolerance=1e-9,
        max_iterations=10000,
    )

    explicit = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        backend=backend,
        omega=1.0,
        tolerance=1e-9,
        max_iterations=10000,
    )

    assert default.converged
    assert explicit.converged
    assert default.iterations == explicit.iterations

    np.testing.assert_array_equal(
        default.phi,
        explicit.phi,
    )


@pytest.mark.parametrize("omega", [1.0, 1.3, 1.5, 1.7])
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_sor_recovers_known_solution(backend, omega):
    coefficients, phi_ref, rhs = make_problem()

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        backend=backend,
        omega=omega,
        tolerance=1e-9,
        max_iterations=10000,
    )

    assert result.converged
    assert result.residual <= 1e-9

    np.testing.assert_allclose(
        result.phi,
        phi_ref,
        rtol=0,
        atol=1e-7,
    )

    assert abs(np.mean(result.phi)) < 1e-12

    true_residual = np.linalg.norm(
        rhs
        - apply_poisson_system(
            result.phi,
            coefficients,
        )
    ) / np.sqrt(rhs.size)

    assert result.residual == pytest.approx(
        true_residual,
        rel=1e-10,
        abs=1e-12,
    )


@pytest.mark.parametrize("omega", [1.0, 1.3, 1.7])
def test_python_numba_equivalence(omega):
    coefficients, _, rhs = make_problem(9, 7)

    python_result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        backend="python",
        omega=omega,
        tolerance=1e-9,
        max_iterations=10000,
    )

    numba_result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        backend="numba",
        omega=omega,
        tolerance=1e-9,
        max_iterations=10000,
    )

    assert python_result.converged
    assert numba_result.converged

    np.testing.assert_allclose(
        python_result.phi,
        numba_result.phi,
        rtol=0,
        atol=1e-10,
    )

    assert abs(python_result.iterations - numba_result.iterations) <= 1


@pytest.mark.parametrize(
    "omega",
    [
        -1.0,
        0.0,
        2.0,
        2.1,
        np.nan,
        np.inf,
        True,
        "1.5",
    ],
)
def test_rejects_invalid_omega(omega):
    coefficients, _, rhs = make_problem()

    with pytest.raises(ValueError, match="omega"):
        solve_poisson_gauss_seidel(
            rhs,
            coefficients,
            omega=omega,
        )


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_sor_residual_check_interval(backend):
    coefficients, _, rhs = make_problem()

    result = solve_poisson_gauss_seidel(
        rhs,
        coefficients,
        backend=backend,
        omega=1.5,
        tolerance=1e-9,
        max_iterations=10000,
        residual_check_interval=10,
    )

    assert result.converged
    assert result.residual <= 1e-9

    # O histórico contém o resíduo inicial
    # e os resíduos avaliados periodicamente.
    assert len(result.residual_history) == (1 + result.iterations // 10)


@pytest.mark.parametrize("omega", [1.0, 1.5])
def test_solver_full_integration(omega):
    config = SimulationConfig(
        reynolds=100,
        nx=8,
        ny=8,
        dt=0.001,
        max_iterations=3,
    )

    result = solve(
        config,
        backend="numba",
        omega=omega,
        pressure_tolerance=1e-9,
        pressure_max_iterations=20000,
        residual_check_interval=10,
        stop_on_steady=False,
        record_interval=1,
    )

    assert result.steps == 3
    assert result.stop_reason == "max_steps"

    for field in (
        result.state.u,
        result.state.v,
        result.state.pressure,
    ):
        assert np.all(np.isfinite(field))

    for record in result.history:
        assert record.poisson_residual <= 1e-9
        assert record.diagnostics.divergence_rms < 1e-9
