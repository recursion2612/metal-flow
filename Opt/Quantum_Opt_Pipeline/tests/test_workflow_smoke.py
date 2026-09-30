import numpy as np
import pytest

pytest.importorskip("physicsnemo")

from pipeline import run_pipeline


class _Component:
    options = {"pad_width": "455um", "pad_height": "50um", "pad_gap": "25um"}


class _Design:
    components = {"Q1": _Component()}

    def rebuild(self):
        pass


def test_pipeline_runs_end_to_end_without_palace(tmp_path):
    calls = []

    def fake_evaluator(design, params, run_name):
        calls.append(run_name)
        return float(18000.0 + params[0]), float(320.0 + params[2])

    result = run_pipeline(
        design=_Design(),
        output_root=str(tmp_path),
        pop_size=5,
        generations=1,
        palace_evaluator=fake_evaluator,
        train_surrogate=True,
        use_palace=True,
    )

    assert len(calls) == 1
    assert result["training_phase_separate"] is True
    assert result["optimization_is_frozen"] is True
    assert result["result_source"] == "Palace validation of frozen surrogate-selected design"
    assert (tmp_path / "optimization_result.json").exists()


def test_pipeline_early_convergence_within_five_percent(tmp_path, monkeypatch):
    from src.surrogate import PhysicsNeMoSurrogate

    # Mock surrogate predict to return values strictly within 5% of target (20000, 320)
    # 20100 is 0.5% error, 321 is 0.3% error -> within 5% tolerance
    def mock_predict(self, population):
        n = len(population)
        preds = np.zeros((n, 2))
        preds[:, 0] = 20100.0  # 0.5% error from 20000
        preds[:, 1] = 321.0    # 0.3% error from 320
        return preds, np.zeros((n, 2))

    monkeypatch.setattr(PhysicsNeMoSurrogate, "predict", mock_predict)

    result = run_pipeline(
        design=_Design(),
        output_root=str(tmp_path),
        pop_size=10,
        generations=20,  # Max generations budget
        target_tolerance=0.05,
        use_palace=False,
    )

    # Should converge on generation 1 (completed_generations == 1) rather than running all 20
    assert result["converged_within_tolerance"] is True
    assert result["generations_completed"] == 1
    assert result["max_generations"] == 20
    assert result["population_size"] == 10
    assert not (tmp_path / "generation_data").exists()  # Cleaned up


def test_pipeline_palace_simulation_every_generation(tmp_path):
    calls = []

    def fake_evaluator(design, params, run_name):
        calls.append((run_name, list(params)))
        return float(18000.0 + params[0]), float(320.0 + params[2])

    result = run_pipeline(
        design=_Design(),
        output_root=str(tmp_path),
        pop_size=6,
        generations=3,
        palace_evaluator=fake_evaluator,
        use_palace=True,
        palace_elites=1,
    )

    # Fake evaluator must be called across all 3 generations
    assert len(calls) == 3
    assert calls[0][0] == "gen_00_elite_00"
    assert calls[1][0] == "gen_01_elite_00"
    assert calls[2][0] == "gen_02_elite_00"
    assert result["generations_completed"] == 3
    assert result["result_source"] == "Palace validation of frozen surrogate-selected design"
    assert "best_metrics_mhz" in result


def test_pipeline_default_parameters():
    import inspect
    sig = inspect.signature(run_pipeline)
    assert sig.parameters["pop_size"].default == 100
    assert sig.parameters["generations"].default == 20
    assert sig.parameters["target_tolerance"].default == 0.05
    assert sig.parameters["use_palace"].default is True
    assert sig.parameters["palace_elites"].default == 1