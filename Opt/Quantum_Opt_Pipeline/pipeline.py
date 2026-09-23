import json
from pathlib import Path
import numpy as np

from src.palace import (
    DEFAULT_PALACE_PATH,
    default_mpi_procs,
    update_qiskit_geometry,
)
from src.surrogate import PhysicsNeMoSurrogate
from src.genetic_algorithm import compute_cost, produce_next_generation
from src.cad import SqdmetalCapacitanceRunner


def run_pipeline(
    design,
    output_root: str = ".",
    pop_size: int = 12,
    generations: int = 10,
    palace_mpi_procs: int = default_mpi_procs(),
    palace_bin: str = DEFAULT_PALACE_PATH,
    palace_evaluator=None,
    model_checkpoint: str | None = None,
    model_data_log: str | None = None,
    train_surrogate: bool = False,
    use_palace: bool = False,
):
    # -------------------------------------------------------------------------
    # 1. Parameter Names & Boundary Limits
    # -------------------------------------------------------------------------
    param_names = [
        "Q1.pad_width",       # Targets design.components['Q1'].options['pad_width']
        "Q1.pad_height",      # Targets design.components['Q1'].options['pad_height']
        "Q1.pad_gap",         # Targets design.components['Q1'].options['pad_gap']
        "lj"                  # Circuit parameter (Josephson Inductance)
    ]

    bounds = np.array([
        [300.0, 600.0],    # Q1.pad_width (um)
        [20.0, 80.0],      # Q1.pad_height (um)
        [10.0, 40.0],      # Q1.pad_gap (um)
        [6.0e-9, 14.0e-9]  # lj (H)
    ])

    uncertainty_threshold = 1.5
    mutation_rate = 1.0 / len(bounds)

    if design is None:
        raise ValueError("run_pipeline requires a Qiskit Metal design instance")
    if pop_size < 3 or generations < 1:
        raise ValueError("pop_size must be at least 3 and generations must be positive")

    output_root = Path(output_root)

    # Setup directories
    palace_root = output_root / "data/palace_runs"
    training_root = output_root / "training_data"
    palace_root.mkdir(parents=True, exist_ok=True)
    training_root.mkdir(parents=True, exist_ok=True)

    evaluator = palace_evaluator or SqdmetalCapacitanceRunner(
        output_root=str(palace_root),
        palace_bin=palace_bin,
        n_procs=palace_mpi_procs,
    )

    # -------------------------------------------------------------------------
    # 2. Population & Surrogate Initialization
    # -------------------------------------------------------------------------
    population = np.random.uniform(
        bounds[:, 0], bounds[:, 1], size=(pop_size, len(bounds))
    )

    checkpoint_path = model_checkpoint or str(training_root / "checkpoints/nemo_surrogate.mdlus")
    data_log_path = model_data_log
    if data_log_path is None and model_checkpoint is not None:
        data_log_path = str(Path(model_checkpoint).parent.parent / "active_learning_log.csv")

    surrogate = PhysicsNeMoSurrogate(
        n_features=len(bounds),
        data_log_path=data_log_path or str(training_root / "active_learning_log.csv"),
        checkpoint_path=checkpoint_path,
    )
    if not surrogate.is_trained:
        print(
            "WARNING: optimization is running in inference-only mode with a frozen "
            "surrogate. Train the model separately with train_surrogate.py "
            "before using this phase."
        )

    if train_surrogate:
        print(
            "INFO: train_surrogate is ignored during optimization because training "
            "must remain separate from the optimization phase."
        )

    print("=" * 75)
    print("STARTING FROZEN-SURROGATE QUANTUM CHIP OPTIMIZATION")
    print(f"Palace Executable : {palace_bin}")
    print(f"Components Bound  : {param_names}")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 3. Inference-only Genetic Optimization Loop
    # -------------------------------------------------------------------------
    for gen in range(generations):
        costs = np.zeros(pop_size)
        predictions, _ = surrogate.predict(population)

        for i in range(pop_size):
            ej_mhz, ec_mhz = predictions[i]
            costs[i] = compute_cost(ej_mhz, ec_mhz)

        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        print(
            f"Gen {gen:02d}/{generations:02d} | "
            f"Best Cost: {best_cost:.5f} | "
            f"Palace Runs (Gen/Total): 00/{len(surrogate.X_train):03d}"
        )

        if gen < generations - 1:
            population = produce_next_generation(
                population=population,
                costs=costs,
                bounds=bounds,
                n_elites=2,
                crossover_rate=0.9,
                mutation_rate=mutation_rate,
            )

    # -------------------------------------------------------------------------
    # 4. Results Summary
    # -------------------------------------------------------------------------
    best_idx = int(np.argmin(costs))
    best_candidate = population[best_idx]
    predicted_ej_mhz, predicted_ec_mhz = predictions[best_idx]

    final_run_name = f"final_selection_gen_{generations - 1:02d}"
    if use_palace:
        update_qiskit_geometry(design, best_candidate, param_names, default_unit="um")
        if callable(evaluator):
            best_ej_mhz, best_ec_mhz = evaluator(design, best_candidate, final_run_name)
        else:
            best_ej_mhz, best_ec_mhz = evaluator.evaluate(design, best_candidate, final_run_name)
        final_costs = np.array([compute_cost(best_ej_mhz, best_ec_mhz)])
        result_source = "Palace validation of frozen surrogate-selected design"
    else:
        best_ej_mhz, best_ec_mhz = predicted_ej_mhz, predicted_ec_mhz
        final_costs = np.array([compute_cost(best_ej_mhz, best_ec_mhz)])
        result_source = "Frozen surrogate evaluation only"

    result = {
        "best_parameters": dict(zip(param_names, [float(value) for value in best_candidate])),
        "best_metrics_mhz": {"Ej": float(best_ej_mhz), "Ec": float(best_ec_mhz)},
        "predicted_metrics_mhz": {
            "Ej": float(predicted_ej_mhz),
            "Ec": float(predicted_ec_mhz),
        },
        "best_cost": float(final_costs[0]),
        "result_source": result_source,
        "generations": generations,
        "population_size": pop_size,
        "mutation_rate": mutation_rate,
        "palace_samples": int(len(surrogate.X_train)),
        "palace_executable": palace_bin,
        "training_log": str(training_root / "active_learning_log.csv"),
        "surrogate_checkpoint": str(training_root / "checkpoints/nemo_surrogate.mdlus"),
        "training_phase_separate": True,
        "optimization_is_frozen": True,
    }
    result_path = output_root / "optimization_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    
    print("\n" + "=" * 75)
    print("OPTIMIZATION FINISHED")
    print("=" * 75)
    print("Optimal Parameters:")
    for name, val in zip(param_names, best_candidate):
        if name.lower() in ["lj", "l_j"]:
            print(f"  - {name:<20}: {val * 1e9:.3f} nH")
        else:
            print(f"  - {name:<20}: {val:.3f} um")
    print(f"Result manifest      : {result_path}")
    return result


if __name__ == "__main__":
    raise SystemExit("Import run_pipeline and pass a Qiskit Metal design plus mesh_exporter")
