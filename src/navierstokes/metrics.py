"""Métricas comuns para comparação FDM x FVM."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationMetrics:
    method: str
    iterations: int
    residual: float
    runtime_seconds: float
    u_min: float
    u_max: float
    v_min: float
    v_max: float
