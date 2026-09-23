"""Run the default Qiskit Metal + SQDMetal + PhysicsNeMo optimization."""

import argparse

from pipeline import run_pipeline
from src.cad import create_transmon_design
from src.palace import DEFAULT_PALACE_PATH, default_mpi_procs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="run_01")
    parser.add_argument(
        "--palace-bin",
        default=DEFAULT_PALACE_PATH,
    )
    parser.add_argument("--population", type=int, default=100, help="GA population size (default: 100)")
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Maximum number of variable generations (default: 20)",
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=0.05,
        help="Target parameter relative tolerance for early stopping (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--ej-target",
        type=float,
        default=20000.0,
        help="Target Josephson energy in MHz (default: 20000.0)",
    )
    parser.add_argument(
        "--ec-target",
        type=float,
        default=320.0,
        help="Target charging energy in MHz (default: 320.0)",
    )
    parser.add_argument("--mpi-procs", type=int, default=default_mpi_procs())
    parser.add_argument(
        "--model-checkpoint",
        help="Trained .mdlus checkpoint used for prediction-only optimization",
    )
    parser.add_argument(
        "--model-data-log",
        help="CSV used to restore the samples and normalization for the checkpoint",
    )
    parser.add_argument(
        "--train-surrogate",
        action="store_true",
        help="Opt in to active-learning training during optimization",
    )
    parser.add_argument(
        "--use-palace",
        dest="use_palace",
        action="store_true",
        default=True,
        help="Simulate best ranking candidates with Palace after each generation and evaluate final design (default: True)",
    )
    parser.add_argument(
        "--no-palace",
        dest="use_palace",
        action="store_false",
        help="Disable live Palace simulation (surrogate evaluation only)",
    )
    parser.add_argument(
        "--palace-elites",
        type=int,
        default=1,
        help="Number of best ranking candidates to simulate with Palace each generation (default: 1)",
    )
    args = parser.parse_args()

    design = create_transmon_design()
    run_pipeline(
        design=design,
        output_root=args.output_root,
        pop_size=args.population,
        generations=args.generations,
        target_tolerance=args.target_tolerance,
        ej_target=args.ej_target,
        ec_target=args.ec_target,
        palace_mpi_procs=args.mpi_procs,
        palace_bin=args.palace_bin,
        model_checkpoint=args.model_checkpoint,
        model_data_log=args.model_data_log,
        train_surrogate=args.train_surrogate,
        use_palace=args.use_palace,
        palace_elites=args.palace_elites,
    )


if __name__ == "__main__":
    main()
