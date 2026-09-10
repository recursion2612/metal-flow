import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cad_adapter import _read_terminal_capacitance
from src.ga_optimizer import produce_next_generation
from src.palace_cad_interface import update_qiskit_geometry


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
