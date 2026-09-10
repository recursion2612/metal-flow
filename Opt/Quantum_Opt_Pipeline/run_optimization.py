"""Run the default Qiskit Metal + SQDMetal + PhysicsNeMo optimization."""

import argparse

from main import run_pipeline
from src.cad_adapter import create_transmon_design


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="run_01")
    parser.add_argument(
        "--palace-bin",
        default="/Users/akhshatkampassi/Documents/CDAC/quantum_design_env/palace/build/bin/palace",
    )
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--mpi-procs", type=int, default=4)
    args = parser.parse_args()

    design = create_transmon_design()
    run_pipeline(
        design=design,
        output_root=args.output_root,
        pop_size=args.population,
        generations=args.generations,
        palace_mpi_procs=args.mpi_procs,
        palace_bin=args.palace_bin,
    )


if __name__ == "__main__":
    main()
