import pytest

from navierstokes.config import SimulationConfig


def test_default_viscosity_for_re100():
    config = SimulationConfig(reynolds=100.0)
    assert config.kinematic_viscosity == pytest.approx(0.01)


def test_reynolds_must_be_positive():
    config = SimulationConfig(reynolds=0.0)
    with pytest.raises(ValueError):
        _ = config.kinematic_viscosity
