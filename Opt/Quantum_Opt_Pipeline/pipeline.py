from __future__ import annotations

import gc
import json
from pathlib import Path
import shutil
from typing import Any
import numpy as np

from src.palace import (
    DEFAULT_PALACE_PATH,
    default_mpi_procs,
    update_qiskit_geometry,
)
from src.surrogate import PhysicsNeMoSurrogate, find_sample_logs
from src.genetic_algorithm import compute_cost, produce_next_generation
from src.cad import SqdmetalCapacitanceRunner


def run_pipeline(
    design,
    output_root: str = ".",
    pop_size: int = 100,
    generations: int = 20,
    target_tolerance: float = 0.05,
    ej_target: float = 20000.0,
    ec_target: float = 320.0,
    palace_mpi_procs: int | None = None,
    palace_bin: str = DEFAULT_PALACE_PATH,
    palace_evaluator=None,
    model_checkpoint: str | None = None,
    model_data_log: str | None = None,
    train_surrogate: bool = False,
    use_palace: bool = True,
    palace_elites: int = 1,
    min_ratio: float = 40.0,
    max_ratio: float = float("inf"),
    inverse_meta: dict[str, Any] | None = None,
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

    if palace_mpi_procs is None:
        palace_mpi_procs = default_mpi_procs()

    output_root = Path(output_root)

    # Setup directories
    palace_root = output_root / "data/palace_runs"
    training_root = output_root / "training_data"
    palace_root.mkdir(parents=True, exist_ok=True)
    training_root.mkdir(parents=True, exist_ok=True)

    # Check Palace availability if live simulation is requested
    if use_palace and palace_evaluator is None:
        palace_available = bool(
            shutil.which(palace_bin)
            or (Path(palace_bin).is_file() and os.access(palace_bin, os.X_OK))
        )
        if not palace_available:
            print(
                f"WARNING: Palace executable not found at '{palace_bin}'. "
                "Running in surrogate-only optimization mode."
            )
            use_palace = False

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
    print("STARTING QUANTUM CHIP OPTIMIZATION")
    print(f"Palace Executable : {palace_bin}")
    print(f"Palace Simulation : {'Enabled (best candidates & final design)' if use_palace else 'Disabled (surrogate only)'}")
    print(f"Components Bound  : {param_names}")
    print("=" * 75)

    generation_data_dir = output_root / "generation_data"
    generation_data_dir.mkdir(parents=True, exist_ok=True)

    # Cache simulation results to avoid re-evaluating identical geometries
    sim_cache: dict[tuple[float, ...], tuple[float, float]] = {}
    if surrogate.X_train.size:
        for x_meas, y_meas in zip(surrogate.X_train, surrogate.Y_train):
            sim_cache[tuple(np.round(x_meas, 6))] = (float(y_meas[0]), float(y_meas[1]))

    def _eval_candidate_with_palace(candidate: np.ndarray, run_name: str) -> tuple[float, float]:
        cand_key = tuple(np.round(candidate, 6))
        if cand_key in sim_cache:
            return sim_cache[cand_key]

        # Prune older simulation folders if necessary to preserve disk space
        if palace_root.exists():
            for child in palace_root.iterdir():
                if child.is_dir() and child.name != run_name:
                    shutil.rmtree(child, ignore_errors=True)

        update_qiskit_geometry(design, candidate, param_names, default_unit="um")
        if callable(evaluator):
            ej_sim, ec_sim = evaluator(design, candidate, run_name)
        else:
            ej_sim, ec_sim = evaluator.evaluate(design, candidate, run_name)

        sim_cache[cand_key] = (float(ej_sim), float(ec_sim))
        surrogate.log_and_append_sample(candidate, [float(ej_sim), float(ec_sim)])
        return float(ej_sim), float(ec_sim)

    # -------------------------------------------------------------------------
    # 3. Genetic Optimization Loop (Variable Generations)
    # -------------------------------------------------------------------------
    converged = False
    completed_generations = 0

    for gen in range(generations):
        completed_generations = gen + 1
        costs = np.zeros(pop_size)
        predictions, _ = surrogate.predict(population)

        for i in range(pop_size):
            cand_key = tuple(np.round(population[i], 6))
            if cand_key in sim_cache:
                sim_ej, sim_ec = sim_cache[cand_key]
                costs[i] = compute_cost(
                    sim_ej, sim_ec, ej_target=ej_target, ec_target=ec_target,
                    min_ratio=min_ratio, max_ratio=max_ratio
                )
            else:
                ej_mhz, ec_mhz = predictions[i]
                costs[i] = compute_cost(
                    ej_mhz, ec_mhz, ej_target=ej_target, ec_target=ec_target,
                    min_ratio=min_ratio, max_ratio=max_ratio
                )

        # After every population evaluation, simulate the best ranking candidates using Palace
        # so their physical fitness is verified and carried into the next iteration as elites
        if use_palace:
            pre_ranking = np.argsort(costs)
            simulated_in_gen = 0
            for cand_idx in pre_ranking:
                cand = population[cand_idx]
                cand_key = tuple(np.round(cand, 6))
                if cand_key not in sim_cache:
                    sim_run_name = f"gen_{gen:02d}_elite_{simulated_in_gen:02d}"
                    sim_ej, sim_ec = _eval_candidate_with_palace(cand, sim_run_name)
                    costs[cand_idx] = compute_cost(
                        sim_ej, sim_ec, ej_target=ej_target, ec_target=ec_target,
                        min_ratio=min_ratio, max_ratio=max_ratio
                    )
                    simulated_in_gen += 1
                    if simulated_in_gen >= palace_elites:
                        break

        best_idx = int(np.argmin(costs))
        best_cost = float(costs[best_idx])
        best_candidate = population[best_idx]
        best_ej_pred, best_ec_pred = predictions[best_idx]

        cand_key = tuple(np.round(best_candidate, 6))
        if cand_key in sim_cache:
            best_ej_eval, best_ec_eval = sim_cache[cand_key]
            eval_label = "Palace-simulated"
        else:
            best_ej_eval, best_ec_eval = best_ej_pred, best_ec_pred
            eval_label = "Surrogate-predicted"

        # Calculate relative errors to target Hamiltonian parameters
        ej_rel_err = abs(best_ej_eval - ej_target) / ej_target
        ec_rel_err = abs(best_ec_eval - ec_target) / ec_target
        max_rel_err = max(ej_rel_err, ec_rel_err)

        print(
            f"Gen {gen:02d}/{generations:02d} | "
            f"Best Cost: {best_cost:.5f} ({eval_label}) | "
            f"Ej: {best_ej_eval:.1f} MHz (err: {ej_rel_err * 100:.1f}%) | "
            f"Ec: {best_ec_eval:.1f} MHz (err: {ec_rel_err * 100:.1f}%)"
        )

        # Simultaneously delete old generation checkpoint files to clean up after itself
        for old_file in generation_data_dir.glob("generation_*.json"):
            old_file.unlink(missing_ok=True)

        # Write current generation state checkpoint
        current_gen_file = generation_data_dir / f"generation_{gen:02d}.json"
        current_gen_file.write_text(
            json.dumps(
                {
                    "generation": gen,
                    "best_cost": best_cost,
                    "evaluation_mode": eval_label,
                    "Ej_MHz": float(best_ej_eval),
                    "Ec_MHz": float(best_ec_eval),
                    "predicted_Ej_MHz": float(best_ej_pred),
                    "predicted_Ec_MHz": float(best_ec_pred),
                    "Ej_error_percent": float(ej_rel_err * 100),
                    "Ec_error_percent": float(ec_rel_err * 100),
                    "max_error_percent": float(max_rel_err * 100),
                    "best_parameters": dict(zip(param_names, [float(v) for v in best_candidate])),
                },
                indent=2,
            )
            + "\n"
        )

        # Variable stopping: Check if candidate is within tolerance (default: 5%)
        # and satisfies physical regime conditions (transmon ratio Ej/Ec in selection bounds)
        ratio = best_ej_eval / max(1e-3, best_ec_eval)
        ratio_ok = ratio >= min_ratio and (not np.isfinite(max_ratio) or ratio <= max_ratio)
        ej_bounds_ok = (ej_target * 0.7 <= best_ej_eval <= ej_target * 1.3) or (15000.0 <= best_ej_eval <= 30000.0)
        ec_bounds_ok = (ec_target * 0.7 <= best_ec_eval <= ec_target * 1.3) or (200.0 <= best_ec_eval <= 500.0)
        if (
            max_rel_err <= target_tolerance
            and ratio_ok
            and ej_bounds_ok
            and ec_bounds_ok
        ):
            print(
                f"\n--> Convergence achieved: best candidate is within {target_tolerance * 100:.1f}% "
                f"of target params at generation {gen} (Ej err: {ej_rel_err * 100:.2f}%, Ec err: {ec_rel_err * 100:.2f}%)."
            )
            print(f"--> Stopping early after {completed_generations} variable generations.")
            converged = True
            break

        if gen < generations - 1:
            population = produce_next_generation(
                population=population,
                costs=costs,
                bounds=bounds,
                n_elites=2,
                crossover_rate=0.9,
                mutation_rate=mutation_rate,
            )

        # Free temporary array allocations between generations
        gc.collect()

    # -------------------------------------------------------------------------
    # 4. Results Summary & Final Palace Evaluation
    # -------------------------------------------------------------------------
    best_idx = int(np.argmin(costs))
    best_candidate = population[best_idx]
    predicted_ej_mhz, predicted_ec_mhz = predictions[best_idx]

    final_run_name = f"final_selection_gen_{completed_generations - 1:02d}"
    if use_palace:
        cand_key = tuple(np.round(best_candidate, 6))
        if cand_key in sim_cache:
            best_ej_mhz, best_ec_mhz = sim_cache[cand_key]
        else:
            best_ej_mhz, best_ec_mhz = _eval_candidate_with_palace(best_candidate, final_run_name)
        final_costs = np.array([compute_cost(best_ej_mhz, best_ec_mhz, ej_target=ej_target, ec_target=ec_target, min_ratio=min_ratio, max_ratio=max_ratio)])
        result_source = "Palace validation of frozen surrogate-selected design"
    else:
        best_ej_mhz, best_ec_mhz = predicted_ej_mhz, predicted_ec_mhz
        final_costs = np.array([compute_cost(best_ej_mhz, best_ec_mhz, ej_target=ej_target, ec_target=ec_target, min_ratio=min_ratio, max_ratio=max_ratio)])
        result_source = "Frozen surrogate evaluation only"

    best_ej_err = abs(best_ej_mhz - ej_target) / ej_target
    best_ec_err = abs(best_ec_mhz - ec_target) / ec_target

    result = {
        "best_parameters": dict(zip(param_names, [float(value) for value in best_candidate])),
        "best_metrics_mhz": {"Ej": float(best_ej_mhz), "Ec": float(best_ec_mhz)},
        "predicted_metrics_mhz": {
            "Ej": float(predicted_ej_mhz),
            "Ec": float(predicted_ec_mhz),
        },
        "target_metrics_mhz": {"Ej": float(ej_target), "Ec": float(ec_target)},
        "relative_error": {
            "Ej": float(best_ej_err),
            "Ec": float(best_ec_err),
            "max_error": float(max(best_ej_err, best_ec_err)),
        },
        "converged_within_tolerance": bool(max(best_ej_err, best_ec_err) <= target_tolerance),
        "target_tolerance": float(target_tolerance),
        "generations_completed": completed_generations,
        "max_generations": generations,
        "population_size": pop_size,
        "best_cost": float(final_costs[0]),
        "result_source": result_source,
        "mutation_rate": mutation_rate,
        "palace_samples": int(len(surrogate.X_train)),
        "palace_executable": palace_bin,
        "training_log": str(training_root / "active_learning_log.csv"),
        "surrogate_checkpoint": str(training_root / "checkpoints/nemo_surrogate.mdlus"),
        "training_phase_separate": True,
        "optimization_is_frozen": True,
    }

    # Attach achieved cQED Hamiltonian metrics & inverse target metadata
    fq_achieved_ghz = (np.sqrt(8.0 * best_ej_mhz * best_ec_mhz) - best_ec_mhz) / 1000.0
    alpha_achieved_mhz = best_ec_mhz
    from src.inverse_design import compute_purcell_t1
    g_ref = 10.0
    if inverse_meta and "targets" in inverse_meta:
        g_ref = inverse_meta["targets"].get("g_MHz", 10.0)
    t1_achieved_us = compute_purcell_t1(fq_achieved_ghz, g_MHz=g_ref)

    result["achieved_cQED_metrics"] = {
        "fQ_GHz": float(fq_achieved_ghz),
        "alpha_MHz": float(alpha_achieved_mhz),
        "Ej_Ec_ratio": float(best_ej_mhz / max(1e-3, best_ec_mhz)),
        "T1_Purcell_us": float(t1_achieved_us),
    }
    if inverse_meta:
        result["inverse_specification"] = inverse_meta

    # Render and export final design layout PNG
    png_path = output_root / "final_design.png"
    try:
        from qiskit_metal import view
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        update_qiskit_geometry(design, best_candidate, param_names)
        fig = view(design)
        fig.savefig(str(png_path), bbox_inches="tight", dpi=300)
        plt.close(fig)
        result["final_design_image"] = str(png_path)
    except Exception as exc:
        result["final_design_image"] = None

    result_path = output_root / "optimization_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    # Clean up transient generation data folder now that final manifest is written
    if generation_data_dir.exists():
        shutil.rmtree(generation_data_dir, ignore_errors=True)

    print("\n" + "=" * 75)
    print("OPTIMIZATION FINISHED")
    print("=" * 75)
    print(f"Generations Run      : {completed_generations} (Max: {generations}, Converged: {result['converged_within_tolerance']})")
    print(f"Population Size      : {pop_size}")
    print(f"Target Errors        : Ej: {best_ej_err * 100:.2f}%, Ec: {best_ec_err * 100:.2f}% (Tolerance: {target_tolerance * 100:.1f}%)")
    print(f"Achieved cQED Values : fQ = {fq_achieved_ghz:.3f} GHz, alpha = {alpha_achieved_mhz:.1f} MHz, T1 = {t1_achieved_us:.1f} us")
    print("Optimal Parameters:")
    for name, val in zip(param_names, best_candidate):
        if name.lower() in ["lj", "l_j"]:
            print(f"  - {name:<20}: {val * 1e9:.3f} nH")
        else:
            print(f"  - {name:<20}: {val:.3f} um")
    print(f"Result manifest      : {result_path}")
    return result


def optimize_design(
    params_dict: dict[str, Any],
    selection: dict[str, Any] | None = None,
    *,
    design=None,
    output_root: str = "run_inverse_design",
    pop_size: int = 100,
    generations: int = 20,
    target_tolerance: float = 0.05,
    use_palace: bool = False,
    include_readout_pad: bool = True,
    **kwargs,
):
    """
    Inverse design optimization for superconducting transmon qubits.

    Args:
        params_dict: Target specification dictionary, e.g.:
                     {"fQ": "4GHz", "alpha": "200 MHz", "g": "10 MHz"}
        selection: Selection criteria and barrier thresholds, e.g.:
                   {"T1": ">100us", "ratio": ">50, <100"}
        design: Optional pre-existing Qiskit Metal design. If None, created automatically.
        output_root: Root directory for optimization artifacts and checkpoints.
        pop_size: Genetic algorithm population size (default: 100).
        generations: Maximum variable generations (default: 20).
        target_tolerance: Convergence tolerance (default: 0.05 = 5%).
        use_palace: Whether to simulate elites and final design with Palace (default: False).
        include_readout_pad: Whether to include a readout coupling pad (default: True).

    Returns:
        The optimized Qiskit Metal design object with ._optimization_result attached.
    """
    from src.cad import create_transmon_design
    from src.inverse_design import (
        parse_target_params,
        parse_selection_criteria,
        invert_hamiltonian,
    )

    # 1. Parse target specifications and selection bounds
    targets = parse_target_params(params_dict)
    sel = parse_selection_criteria(selection)

    # 2. Invert (f_q, alpha) to exact Hamiltonian targets (Ej, Ec)
    hamiltonian = invert_hamiltonian(targets["fQ_GHz"], targets["alpha_MHz"])
    ej_target = hamiltonian["Ej_MHz"]
    ec_target = hamiltonian["Ec_MHz"]

    print("\n" + "=" * 75)
    print("INVERSE DESIGN SPECIFICATION")
    print(f"Target Qubit Frequency (fQ) : {targets['fQ_GHz']:.3f} GHz")
    print(f"Target Anharmonicity (alpha): {targets['alpha_MHz']:.1f} MHz")
    print(f"Target Coupler Rate (g)     : {targets['g_MHz']:.1f} MHz")
    print(f"Inverted Ej Target          : {ej_target:.1f} MHz ({ej_target/1000.0:.2f} GHz)")
    print(f"Inverted Ec Target          : {ec_target:.1f} MHz")
    print(f"Target Differential C_sigma : {hamiltonian['C_sigma_fF']:.2f} fF")
    print(f"Target Josephson L_j        : {hamiltonian['Lj_nH']:.3f} nH")
    print(f"Selection Criteria          : T1 >= {sel['min_T1_us']:.1f} us, "
          f"Ratio in [{sel['min_ratio']:.1f}, {sel['max_ratio']:.1f}]")
    print("=" * 75)

    # 3. Create transmon design if not provided
    if design is None:
        design = create_transmon_design(include_readout_pad=include_readout_pad)

    # 4. Run optimization pipeline
    inverse_meta = {
        "targets": targets,
        "selection": sel,
        "inverted_hamiltonian": hamiltonian,
    }
    result = run_pipeline(
        design=design,
        output_root=output_root,
        pop_size=pop_size,
        generations=generations,
        target_tolerance=target_tolerance,
        ej_target=ej_target,
        ec_target=ec_target,
        min_ratio=sel["min_ratio"],
        max_ratio=sel["max_ratio"],
        inverse_meta=inverse_meta,
        use_palace=use_palace,
        **kwargs,
    )

    design._optimization_result = result
    return design


if __name__ == "__main__":
    raise SystemExit("Import run_pipeline and pass a Qiskit Metal design plus mesh_exporter")
