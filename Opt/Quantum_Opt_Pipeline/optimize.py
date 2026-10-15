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
    # Inverse design arguments
    parser.add_argument(
        "--fq",
        type=str,
        default=None,
        help="Target qubit transition frequency for inverse design (e.g. '4GHz', '5.2 GHz')",
    )
    parser.add_argument(
        "--alpha",
        type=str,
        default=None,
        help="Target anharmonicity for inverse design (e.g. '200MHz', '-220 MHz')",
    )
    parser.add_argument(
        "--coupling-g",
        type=str,
        default=None,
        help="Target coupling rate g to readout/bus (e.g. '10MHz')",
    )
    parser.add_argument(
        "--selection-t1",
        type=str,
        default=">100us",
        help="Selection threshold for Purcell coherence T1 (default: '>100us')",
    )
    parser.add_argument(
        "--selection-ratio",
        type=str,
        default=">50, <100",
        help="Selection bounds for transmon Ej/Ec ratio (default: '>50, <100')",
    )
    parser.add_argument(
        "--include-readout-pad",
        action="store_true",
        default=True,
        help="Include readout connection pad in transmon design (default: True)",
    )
    args = parser.parse_args()

    from show_metrics import print_metrics

    if args.fq is not None:
        from pipeline import optimize_design
        params_dict = {"fQ": args.fq}
        if args.alpha:
            params_dict["alpha"] = args.alpha
        if args.coupling_g:
            params_dict["g"] = args.coupling_g
        
        selection = {
            "T1": args.selection_t1,
            "ratio": args.selection_ratio,
        }
        design = optimize_design(
            params_dict=params_dict,
            selection=selection,
            output_root=args.output_root,
            pop_size=args.population,
            generations=args.generations,
            target_tolerance=args.target_tolerance,
            palace_mpi_procs=args.mpi_procs,
            palace_bin=args.palace_bin,
            model_checkpoint=args.model_checkpoint,
            model_data_log=args.model_data_log,
            train_surrogate=args.train_surrogate,
            use_palace=args.use_palace,
            palace_elites=args.palace_elites,
            include_readout_pad=args.include_readout_pad,
        )
        result = design._optimization_result
    else:
        design = create_transmon_design(include_readout_pad=args.include_readout_pad)
        result = run_pipeline(
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

    print("\n")
    print_metrics(result)


if __name__ == "__main__":
    main()
