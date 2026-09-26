"""Configuração compartilhada pelos solvers e experimentos."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    reynolds: float = 100.0
    nx: int = 51
    ny: int = 51
    dt: float = 1.0e-3
    tolerance: float = 1.0e-5
    max_iterations: int = 12_000
    pressure_iterations: int = 100
    lid_velocity: float = 1.0
    density: float = 1.0
    lx: float = 1.0
    ly: float = 1.0

    @property
    def kinematic_viscosity(self) -> float:
        """Calcula nu = U*L/Re para a cavidade com tampa móvel."""
        if self.reynolds <= 0:
            raise ValueError("reynolds deve ser positivo")
        return self.lid_velocity * self.lx / self.reynolds
