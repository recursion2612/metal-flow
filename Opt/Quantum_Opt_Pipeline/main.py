import json
from pathlib import Path
import numpy as np

from src.palace_cad_interface import update_qiskit_geometry, DEFAULT_PALACE_PATH
from src.nemo_surrogate import PhysicsNeMoSurrogate
from src.ga_optimizer import compute_cost, produce_next_generation
from src.cad_adapter import SqdmetalCapacitanceRunner


def run_pipeline(
    design,
    output_root: str = ".",
    pop_size: int = 12,
    generations: int = 10,
    palace_mpi_procs: int = 4,
    palace_bin: str = DEFAULT_PALACE_PATH,
    palace_evaluator=None,
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

    surrogate = PhysicsNeMoSurrogate(
        n_features=len(bounds),
        data_log_path=str(training_root / "active_learning_log.csv"),
        checkpoint_path=str(training_root / "checkpoints/nemo_surrogate.mdlus"),
    )

    print("=" * 75)
    print("STARTING ACTIVE-LEARNING QUANTUM CHIP OPTIMIZATION")
    print(f"Palace Executable : {palace_bin}")
    print(f"Components Bound  : {param_names}")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 3. Active Learning Genetic Optimization Loop
    # -------------------------------------------------------------------------
    for gen in range(generations):
        costs = np.zeros(pop_size)
        predictions, uncertainties = surrogate.predict(population)
        fem_runs_this_gen = 0

        for i in range(pop_size):
            candidate = population[i]
            unc = uncertainties[i]

            if unc > uncertainty_threshold:
                fem_runs_this_gen += 1
                
                # 1. Update CAD geometry
                update_qiskit_geometry(design, candidate, param_names, default_unit="um")

                # SQDMetal generates the mesh/config with physical tags from the design.
                run_name = f"gen_{gen:02d}_ind_{i:02d}"
                if callable(evaluator):
                    ej_mhz, ec_mhz = evaluator(design, candidate, run_name)
                else:
                    ej_mhz, ec_mhz = evaluator.evaluate(design, candidate, run_name)

                # 5. Log ground-truth sample
                surrogate.log_and_append_sample(candidate, [ej_mhz, ec_mhz])
            else:
                # Surrogate inference
                ej_mhz, ec_mhz = predictions[i]

            costs[i] = compute_cost(ej_mhz, ec_mhz)

        # Update surrogate model
        surrogate.fit()

        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        total_samples = len(surrogate.X_train)

        print(
            f"Gen {gen:02d}/{generations:02d} | "
            f"Best Cost: {best_cost:.5f} | "
            f"Palace Runs (Gen/Total): {fem_runs_this_gen:02d}/{total_samples:03d}"
        )

        # Produce next generation
        if gen < generations - 1:
            population = produce_next_generation(
                population=population,
                costs=costs,
                bounds=bounds,
                n_elites=2,
                crossover_rate=0.9,
                mutation_rate=mutation_rate
            )

    # -------------------------------------------------------------------------
    # 4. Results Summary
    # -------------------------------------------------------------------------
    actual_costs = np.array([
        compute_cost(ej_mhz, ec_mhz)
        for ej_mhz, ec_mhz in surrogate.Y_train
    ])
    best_actual_idx = int(np.argmin(actual_costs))
    best_candidate = surrogate.X_train[best_actual_idx]
    best_ej_mhz, best_ec_mhz = surrogate.Y_train[best_actual_idx]
    result = {
        "best_parameters": dict(zip(param_names, [float(value) for value in best_candidate])),
        "best_metrics_mhz": {"Ej": float(best_ej_mhz), "Ec": float(best_ec_mhz)},
        "best_cost": float(actual_costs[best_actual_idx]),
        "generations": generations,
        "population_size": pop_size,
        "mutation_rate": mutation_rate,
        "palace_samples": int(len(surrogate.X_train)),
        "palace_executable": palace_bin,
        "training_log": str(training_root / "active_learning_log.csv"),
        "surrogate_checkpoint": str(training_root / "checkpoints/nemo_surrogate.mdlus"),
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
