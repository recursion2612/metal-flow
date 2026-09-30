import os
import subprocess
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.cad import _read_terminal_capacitance
from src.genetic_algorithm import produce_next_generation
from src.surrogate import MeshGraphNet, PhysicsNeMoSurrogate
from src.palace import max_mpi_procs, update_qiskit_geometry
from generate_samples import (
    BOUNDS,
    boundary_samples,
    latin_hypercube_samples,
    split_sample_indices,
)


def test_ga_children_stay_inside_bounds():
    bounds = np.array([[0.0, 1.0], [10.0, 20.0]])
    population = np.array([[0.0, 10.0], [1.0, 20.0], [0.5, 15.0]])
    children = produce_next_generation(
        population,
        np.array([3.0, 2.0, 1.0]),
        bounds,
        n_elites=1,
        mutation_rate=0.5,
    )
    assert np.all(children >= bounds[:, 0])
    assert np.all(children <= bounds[:, 1])


def test_mpi_limit_uses_ninety_percent_of_cores_rounded_up():
    assert max_mpi_procs(11) == 10
    assert max_mpi_procs(20) == 18


def test_latin_hypercube_samples_cover_each_stratum():
    samples = latin_hypercube_samples(np.random.default_rng(42), BOUNDS, 10)
    assert samples.shape == (10, 4)
    assert np.all(samples >= BOUNDS[:, 0])
    assert np.all(samples <= BOUNDS[:, 1])
    for parameter_index in range(samples.shape[1]):
        normalized = (samples[:, parameter_index] - BOUNDS[parameter_index, 0]) / (
            BOUNDS[parameter_index, 1] - BOUNDS[parameter_index, 0]
        )
        assert sorted(np.floor(normalized * 10).astype(int)) == list(range(10))


def test_ej_target_transform_round_trips():
    targets = np.array([[12000.0, 19.3], [24000.0, 19.5]])
    transformed = PhysicsNeMoSurrogate._transform_targets(targets)
    restored = PhysicsNeMoSurrogate._inverse_transform_targets(transformed)
    assert np.allclose(restored, targets)


def test_sample_split_counts_round_up_holdouts():
    assignments = split_sample_indices(np.random.default_rng(42), 11, 60, 20, 20)
    assert sum(assignments == "train") == 5
    assert sum(assignments == "validation") == 3
    assert sum(assignments == "test") == 3


def test_boundary_samples_cover_all_parameter_corners():
    corners = boundary_samples(BOUNDS)
    assert corners.shape == (16, 4)
    assert np.all(np.min(corners, axis=0) == BOUNDS[:, 0])
    assert np.all(np.max(corners, axis=0) == BOUNDS[:, 1])


def test_qiskit_geometry_updates_nested_options():
    class Component:
        options = {"nested": {"width": "1um"}}

    class Design:
        components = {"Q1": Component()}
        rebuilt = False

        def rebuild(self):
            self.rebuilt = True

    design = Design()
    update_qiskit_geometry(design, np.array([12.5]), ["Q1.nested.width"])
    assert design.components["Q1"].options["nested"]["width"] == "12.5000um"
    assert design.rebuilt


def test_terminal_capacitance_reader(tmp_path: Path):
    path = tmp_path / "terminal-C.csv"
    pd.DataFrame(
        [[1.0e-12, -2.0e-13], [-2.0e-13, 8.0e-13]],
        index=[1, 2],
    ).to_csv(path)
    matrix = _read_terminal_capacitance(path)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 1.0e-12


def test_meshgraphnet_predicts_scalar_regression_outputs():
    model = MeshGraphNet(in_features=4, out_features=2, hidden_dim=16, n_layers=2)
    x = torch.randn(3, 4)
    y = model(x)
    assert y.shape == (3, 2)


def test_meshgraphnet_dropout_changes_training_predictions():
    model = MeshGraphNet(in_features=4, out_features=2, hidden_dim=16, n_layers=2)
    model.train()
    x = torch.randn(8, 4)
    first = model(x)
    second = model(x)
    assert not torch.allclose(first, second)

def test_setup_scripts_exist_and_pass_syntax():
    root = Path(__file__).resolve().parent.parent
    for script_name in ("setup_environment.sh", "setup_env.sh", "run_container.sh"):
        script = root / script_name
        assert script.exists(), f"{script_name} missing"
        assert os.access(script, os.X_OK), f"{script_name} is not executable"
        ret = subprocess.run(["sh", "-n", str(script)], capture_output=True, text=True)
        assert ret.returncode == 0, f"{script_name} syntax error: {ret.stderr}"


def test_setup_script_help():
    root = Path(__file__).resolve().parent.parent
    script = root / "setup_environment.sh"
    ret = subprocess.run([str(script), "--help"], capture_output=True, text=True)
    assert ret.returncode == 0
    assert "--docker" in ret.stdout
    assert "--cuda" in ret.stdout


def test_dockerfile_and_dockerignore_validity():
    root = Path(__file__).resolve().parent.parent
    dockerfile = root / "Dockerfile"
    dockerignore = root / ".dockerignore"
    assert dockerfile.exists() and dockerfile.stat().st_size > 0
    assert dockerignore.exists() and dockerignore.stat().st_size > 0

    content = dockerfile.read_text()
    assert "/Users/" not in content
    assert "WORKDIR /workspace" in content
    assert "QUANTUM_DESIGN_ENV" in content


def test_zero_local_paths_in_project():
    root = Path(__file__).resolve().parent.parent
    for ext in ("*.py", "*.sh", "*.md", "*.toml", "Dockerfile"):
        for p in root.glob(ext):
            t = p.read_text(errors="replace")
            assert "/Users/" not in t, f"Found /Users/ in {p.name}"
            assert "/home/" not in t or "PATH" in t or "WORKDIR" in t or "username" in t, f"Found local /home/ in {p.name}"
