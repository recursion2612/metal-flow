"""Unit tests for the inverse design engine, connection pads, and performance metrics."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.inverse_design import (
    parse_frequency,
    parse_energy,
    parse_time,
    parse_range_or_bound,
    parse_target_params,
    parse_selection_criteria,
    invert_hamiltonian,
    compute_purcell_t1,
    compute_inverse_cost,
)
from src.cad import (
    create_transmon_design,
    extract_transmon_c_sigma,
    extract_transmon_hamiltonian_lom,
)
from pipeline import optimize_design


def test_parse_frequency():
    assert parse_frequency("4GHz") == 4.0
    assert parse_frequency("4.5 GHz") == 4.5
    assert parse_frequency("5000MHz") == 5.0
    assert parse_frequency("6e9 Hz") == 6.0
    assert parse_frequency(5.2) == 5.2


def test_parse_energy_and_coupling():
    assert parse_energy("200 MHz") == 200.0
    assert parse_energy("-220MHz") == 220.0
    assert parse_energy("0.25 GHz") == 250.0
    assert parse_energy(300.0) == 300.0


def test_parse_time():
    assert parse_time(">100us") == 100.0
    assert parse_time("150 us") == 150.0
    assert parse_time("0.2 ms") == 200.0
    assert parse_time(1e-4) == 100.0


def test_parse_range_or_bound():
    min_r, max_r = parse_range_or_bound(">50, <100")
    assert min_r == 50.0
    assert max_r == 100.0

    min_r, max_r = parse_range_or_bound(">40")
    assert min_r == 40.0
    assert np.isinf(max_r)

    min_r, max_r = parse_range_or_bound("50-80")
    assert min_r == 50.0
    assert max_r == 80.0


def test_parse_target_params_and_selection():
    p = parse_target_params({"fQ": "4GHz", "alpha": "200 Mhz", "g": "10 MHz"})
    assert p["fQ_GHz"] == 4.0
    assert p["alpha_MHz"] == 200.0
    assert p["g_MHz"] == 10.0

    sel = parse_selection_criteria({"T1": ">100us", "ratio": ">50, <100", "chi": ">1MHz, <5MHz"})
    assert sel["min_T1_us"] == 100.0
    assert sel["min_ratio"] == 50.0
    assert sel["max_ratio"] == 100.0
    assert sel["min_chi_MHz"] == 1.0
    assert sel["max_chi_MHz"] == 5.0


def test_invert_hamiltonian_math():
    # Target: 4 GHz, alpha = 200 MHz
    res = invert_hamiltonian(4.0, 200.0)
    assert np.isclose(res["Ec_MHz"], 200.0)
    assert np.isclose(res["Ej_MHz"], 11025.0)
    assert np.isclose(res["ratio"], 55.125)
    assert 50.0 <= res["ratio"] <= 100.0
    assert np.isclose(res["C_sigma_fF"], 96.85, atol=0.1)
    assert np.isclose(res["Lj_nH"], 14.83, atol=0.1)

    # Across 4-6 GHz space
    for f_ghz in [4.0, 4.5, 5.0, 5.5, 6.0]:
        h = invert_hamiltonian(f_ghz, 220.0)
        assert h["ratio"] > 40.0
        assert h["C_sigma_fF"] > 0
        assert h["Lj_nH"] > 0


def test_compute_purcell_t1():
    t1 = compute_purcell_t1(f_q_GHz=5.0, g_MHz=30.0, f_r_GHz=7.0, q_readout=1e4)
    assert t1 > 100.0  # T1 Purcell should be hundreds of microseconds


def test_compute_inverse_cost_penalties():
    targets = {"fQ_GHz": 4.0, "alpha_MHz": 200.0, "g_MHz": 10.0}
    selection = {"min_T1_us": 100.0, "min_ratio": 50.0, "max_ratio": 100.0}
    
    # Perfect match
    good_metrics = {"fQ_GHz": 4.0, "alpha_MHz": 200.0, "g_MHz": 10.0, "ratio": 55.0, "T1_us": 150.0}
    cost_good = compute_inverse_cost(good_metrics, targets, selection)
    assert cost_good < 0.01

    # T1 violation
    bad_t1 = {"fQ_GHz": 4.0, "alpha_MHz": 200.0, "g_MHz": 10.0, "ratio": 55.0, "T1_us": 20.0}
    cost_bad_t1 = compute_inverse_cost(bad_t1, targets, selection)
    assert cost_bad_t1 > 100.0

    # Ratio violation
    bad_ratio = {"fQ_GHz": 4.0, "alpha_MHz": 200.0, "g_MHz": 10.0, "ratio": 30.0, "T1_us": 150.0}
    cost_bad_ratio = compute_inverse_cost(bad_ratio, targets, selection)
    assert cost_bad_ratio > 100.0


def test_cad_generalized_c_sigma():
    # 3x3 isolated floating transmon
    m3 = np.array([
        [1e-12, -25e-15, -25e-15],
        [-25e-15, 60e-15, -35e-15],
        [-25e-15, -35e-15, 60e-15]
    ])
    c3 = extract_transmon_c_sigma(m3)
    assert np.isclose(c3, 35e-15 + (25e-15 * 25e-15) / (50e-15))
    assert np.isclose(c3, 47.5e-15)

    # 4x4 with readout connection pad at index 3
    m4 = np.array([
        [1e-12, -25e-15, -25e-15, -15e-15],
        [-25e-15, 64e-15, -35e-15, -4e-15],
        [-25e-15, -35e-15, 61e-15, -1e-15],
        [-15e-15, -4e-15, -1e-15, 20e-15]
    ])
    c4 = extract_transmon_c_sigma(m4)
    # C1S = 25 + 4 = 29 fF, C2S = 25 + 1 = 26 fF
    # eff = (29*26)/(29+26) = 754/55 = 13.70909 fF => total = 35 + 13.70909 = 48.70909 fF
    assert np.isclose(c4, 48.7090909e-15, atol=1e-18)


def test_cad_transmon_with_connection_pads():
    pytest.importorskip("qiskit_metal")
    design = create_transmon_design(include_readout_pad=True)
    q1 = design.components["Q1"]
    assert "readout" in q1.options.connection_pads
    assert q1.options.connection_pads["readout"]["pad_width"] == "125um"


def test_optimize_design_api_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("physicsnemo")
    from src.surrogate import PhysicsNeMoSurrogate

    # Target: 4 GHz, alpha: 200 MHz => Ej = 11025 MHz, Ec = 200 MHz
    def mock_predict(self, population):
        n = len(population)
        preds = np.zeros((n, 2))
        preds[:, 0] = 11030.0  # close to 11025
        preds[:, 1] = 200.5    # close to 200.0
        return preds, np.zeros((n, 2))

    monkeypatch.setattr(PhysicsNeMoSurrogate, "predict", mock_predict)

    class _MockComponent:
        options = {"pad_width": "455um", "pad_height": "50um", "pad_gap": "25um"}

    class _MockDesign:
        components = {"Q1": _MockComponent()}
        def rebuild(self):
            pass

    design = optimize_design(
        params_dict={"fQ": "4GHz", "alpha": "200 MHz", "g": "10 MHz"},
        selection={"T1": ">100us", "ratio": ">50, <100"},
        design=_MockDesign(),
        output_root=str(tmp_path),
        pop_size=5,
        generations=2,
        use_palace=False,
    )

    assert hasattr(design, "_optimization_result")
    res = design._optimization_result
    assert res["converged_within_tolerance"] is True
    assert "achieved_cQED_metrics" in res
    assert np.isclose(res["achieved_cQED_metrics"]["fQ_GHz"], 4.0, atol=0.05)
    assert np.isclose(res["achieved_cQED_metrics"]["alpha_MHz"], 200.0, atol=2.0)
    assert res["achieved_cQED_metrics"]["Ej_Ec_ratio"] >= 50.0
    assert (tmp_path / "optimization_result.json").exists()


def test_show_metrics_display(capsys):
    from show_metrics import print_metrics
    data = {
        "target_metrics_mhz": {"Ej": 11025.0, "Ec": 200.0},
        "best_metrics_mhz": {"Ej": 11030.0, "Ec": 200.5},
        "relative_error": {"Ej": 0.00045, "Ec": 0.0025},
        "converged_within_tolerance": True,
        "generations_completed": 4,
        "max_generations": 20,
        "population_size": 100,
        "best_parameters": {
            "Q1.pad_width": 412.35,
            "Q1.pad_height": 38.2,
            "Q1.pad_gap": 18.5,
            "lj": 14.82e-9,
        },
        "achieved_cQED_metrics": {
            "fQ_GHz": 4.001,
            "alpha_MHz": 200.5,
            "Ej_Ec_ratio": 55.01,
            "T1_Purcell_us": 1250.0,
        },
        "inverse_specification": {
            "targets": {"fQ_GHz": 4.0, "alpha_MHz": 200.0, "g_MHz": 10.0},
            "selection": {"min_T1_us": 100.0, "min_ratio": 50.0, "max_ratio": 100.0},
        },
    }
    print_metrics(data)
    captured = capsys.readouterr()
    assert "QUANTUM CHIP OPTIMIZATION PERFORMANCE REPORT" in captured.out
    assert "Qubit Frequency" in captured.out
    assert "Purcell Coherence" in captured.out
    assert "Transmon Pad Width" in captured.out
