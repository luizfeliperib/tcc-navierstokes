"""Testes de integridade dos resultados FDM-07.4."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from navierstokes.config import SimulationConfig
from navierstokes.fdm.solver import solve

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name

    spec = importlib.util.spec_from_file_location(path.stem, path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def exported_results(tmp_path):
    """Gera uma simulação pequena para os testes."""

    config = SimulationConfig(
        reynolds=100,
        nx=8,
        ny=6,
        dt=0.001,
        max_iterations=3,
    )

    result = solve(
        config,
        backend="python",
        stop_on_steady=False,
        record_interval=1,
    )

    runner = load_script("run_fdm_validation.py")

    runner.export_results(
        result=result,
        config=config,
        output_dir=tmp_path,
        elapsed=0.5,
        settings={
            "backend": "python",
            "pressure_tolerance": 1e-9,
            "pressure_max_iterations": 10000,
            "residual_check_interval": 1,
            "record_interval": 1,
            "stop_on_steady": False,
            "steady_consecutive_steps": 20,
            "min_steps": 100,
        },
    )

    return tmp_path


def test_integrity_passes(exported_results):
    verifier = load_script("verify_fdm_results.py")

    report = verifier.verify_results(exported_results)

    assert report["status"] == "passed"
    assert report["steps"] == 3
    assert report["flow_status"] == "transient"
    assert report["nx"] == 8
    assert report["ny"] == 6


def test_metadata_classification(exported_results):
    path = exported_results / "metadata.json"

    with path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["schema_version"] == "1.0"
    assert metadata["flow_status"] == "transient"
    assert metadata["result"]["converged"] is False
    assert metadata["result"]["stop_reason"] == "max_steps"

    assert metadata["centerline_profiles"]["vertical"]["fixed_coordinate"] == "x"


def test_missing_file_is_rejected(exported_results):
    (exported_results / "centerline_v.csv").unlink()

    verifier = load_script("verify_fdm_results.py")

    with pytest.raises(ValueError, match="Arquivo ausente"):
        verifier.verify_results(exported_results)


def test_corrupted_profile_is_rejected(exported_results):
    path = exported_results / "centerline_u.csv"

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    rows[0]["u"] = str(float(rows[0]["u"]) + 0.1)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["y", "u"])
        writer.writeheader()
        writer.writerows(rows)

    verifier = load_script("verify_fdm_results.py")

    with pytest.raises((AssertionError, ValueError)):
        verifier.verify_results(exported_results)


def test_corrupted_field_is_rejected(exported_results):
    path = exported_results / "fields.npz"

    with np.load(path, allow_pickle=False) as data:
        fields = {key: data[key].copy() for key in data.files}

    fields["u"][0, 0] = np.nan

    np.savez_compressed(path, **fields)

    verifier = load_script("verify_fdm_results.py")

    with pytest.raises(ValueError, match="Valores não finitos"):
        verifier.verify_results(exported_results)


def test_corrupted_status_is_rejected(exported_results):
    path = exported_results / "metadata.json"

    with path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    metadata["flow_status"] = "steady"

    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file)

    verifier = load_script("verify_fdm_results.py")

    with pytest.raises(ValueError, match="Estado do escoamento"):
        verifier.verify_results(exported_results)
