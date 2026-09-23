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
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=10)
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
        action="store_true",
        help="Opt in to Palace evaluation for high-uncertainty candidates",
    )
    args = parser.parse_args()

    design = create_transmon_design()
    run_pipeline(
        design=design,
        output_root=args.output_root,
        pop_size=args.population,
        generations=args.generations,
        palace_mpi_procs=args.mpi_procs,
        palace_bin=args.palace_bin,
        model_checkpoint=args.model_checkpoint,
        model_data_log=args.model_data_log,
        train_surrogate=args.train_surrogate,
        use_palace=args.use_palace,
    )


if __name__ == "__main__":
    main()
