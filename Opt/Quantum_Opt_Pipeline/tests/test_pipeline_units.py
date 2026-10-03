import os
import subprocess
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.cad import _read_terminal_capacitance, extract_transmon_c_sigma
from src.genetic_algorithm import produce_next_generation
from src.surrogate import (
    MeshGraphNet,
    PhysicsNeMoSurrogate,
    deduplicate_samples,
    find_sample_logs,
)
from src.palace import max_mpi_procs, update_qiskit_geometry
from generate_samples import (
    BOUNDS,
    boundary_samples,
    latin_hypercube_samples,
    split_sample_indices,
    write_sample_splits,
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


def test_extract_transmon_c_sigma():
    # 3x3 SQDMetal terminal capacitance matrix: [Chip, Cond1, Cond2]
    # c12 = 30 fF, c1_g = 60 fF, c2_g = 60 fF
    # c_g_eff = (60 * 60) / 120 = 30 fF
    # c_sigma = 30 + 30 = 60 fF = 60e-15
    matrix_3x3 = np.array([
        [1.0e-12, -60e-15, -60e-15],
        [-60e-15, 90e-15, -30e-15],
        [-60e-15, -30e-15, 90e-15],
    ])
    c_sigma = extract_transmon_c_sigma(matrix_3x3)
    np.testing.assert_allclose(c_sigma, 60e-15)

    # 2x2 matrix fallback: mutual capacitance
    matrix_2x2 = np.array([
        [50e-15, -45e-15],
        [-45e-15, 50e-15],
    ])
    assert extract_transmon_c_sigma(matrix_2x2) == 45e-15

    # 1x1 matrix fallback
    matrix_1x1 = np.array([[70e-15]])
    assert extract_transmon_c_sigma(matrix_1x1) == 70e-15


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


def test_deduplicate_samples():
    x = np.array([
        [300.0, 20.0, 10.0, 8.0e-9],
        [300.0, 20.0, 10.0, 8.0e-9],  # Exact duplicate
        [450.0, 50.0, 25.0, 10.0e-9],
        [300.00001, 20.00001, 10.0, 8.0e-9],  # Near duplicate within tolerance
        [550.0, 70.0, 35.0, 12.0e-9],
    ])
    y = np.array([
        [20000.0, 320.0],
        [20000.0, 320.0],
        [18000.0, 280.0],
        [20000.0, 320.0],
        [16000.0, 250.0],
    ])
    unique_x, unique_y = deduplicate_samples(x, y, rel_tol=1e-4)
    assert len(unique_x) == 3
    assert len(unique_y) == 3
    assert np.allclose(unique_x[0], [300.0, 20.0, 10.0, 8.0e-9])
    assert np.allclose(unique_x[1], [450.0, 50.0, 25.0, 10.0e-9])
    assert np.allclose(unique_x[2], [550.0, 70.0, 35.0, 12.0e-9])


def test_find_sample_logs_and_load_additional(tmp_path: Path):
    run1_dir = tmp_path / "run1" / "training_data"
    run1_dir.mkdir(parents=True)
    run2_dir = tmp_path / "run2" / "training_data"
    run2_dir.mkdir(parents=True)

    cols = ["param_0", "param_1", "param_2", "param_3", "Ej_MHz", "Ec_MHz"]
    df1 = pd.DataFrame([
        [300.0, 20.0, 10.0, 8.0e-9, 20000.0, 320.0],
        [400.0, 40.0, 20.0, 9.0e-9, 19000.0, 300.0],
    ], columns=cols)
    df1.to_csv(run1_dir / "active_learning_log.csv", index=False)

    df2 = pd.DataFrame([
        [300.0, 20.0, 10.0, 8.0e-9, 20000.0, 320.0],  # Duplicate of run1
        [500.0, 60.0, 30.0, 11.0e-9, 17000.0, 270.0],  # New sample
    ], columns=cols)
    df2.to_csv(run2_dir / "active_learning_log.csv", index=False)

    found = find_sample_logs([tmp_path])
    assert len(found) == 2

    surrogate = PhysicsNeMoSurrogate(
        n_features=4,
        data_log_path=str(run1_dir / "active_learning_log.csv"),
    )
    assert len(surrogate.X_train) == 2

    # Load older samples from run2
    added = surrogate.load_additional_samples([run2_dir], deduplicate=True)
    assert added == 1  # Only 1 unique sample added, 1 duplicate ignored
    assert len(surrogate.X_train) == 3

    # Test export
    consolidated = tmp_path / "consolidated.csv"
    surrogate.save_training_data(consolidated)
    assert consolidated.exists()
    saved_df = pd.read_csv(consolidated)
    assert len(saved_df) == 3


def test_write_sample_splits_on_combined_dataset(tmp_path: Path):
    data_log = tmp_path / "active_learning_log.csv"
    cols = ["param_0", "param_1", "param_2", "param_3", "Ej_MHz", "Ec_MHz"]
    # 20 samples total (e.g. 10 old + 10 new)
    data = np.random.uniform(1.0, 10.0, size=(20, 6))
    pd.DataFrame(data, columns=cols).to_csv(data_log, index=False)

    rng = np.random.default_rng(42)
    assignments = split_sample_indices(rng, 20, 70, 10, 20)
    split_manifest = tmp_path / "sample_splits.csv"
    write_sample_splits(data_log, split_manifest, tmp_path, assignments)

    assert split_manifest.exists()
    assert (tmp_path / "training_data" / "train_samples.csv").exists()
    assert (tmp_path / "training_data" / "validation_samples.csv").exists()
    assert (tmp_path / "training_data" / "test_samples.csv").exists()
    train_df = pd.read_csv(tmp_path / "training_data" / "train_samples.csv")
    val_df = pd.read_csv(tmp_path / "training_data" / "validation_samples.csv")
    test_df = pd.read_csv(tmp_path / "training_data" / "test_samples.csv")
    assert len(train_df) + len(val_df) + len(test_df) == 20

