"""Generate Palace measurements for offline PhysicsNeMo surrogate training."""

import argparse
from pathlib import Path

import numpy as np

from src.cad_adapter import SqdmetalCapacitanceRunner, create_transmon_design
from src.nemo_surrogate import PhysicsNeMoSurrogate
from src.palace_cad_interface import DEFAULT_PALACE_PATH, update_qiskit_geometry


BOUNDS = np.array([
    [300.0, 600.0],
    [20.0, 80.0],
    [10.0, 40.0],
    [6.0e-9, 14.0e-9],
])
PARAM_NAMES = ["Q1.pad_width", "Q1.pad_height", "Q1.pad_gap", "lj"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="training_run")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mpi-procs", type=int, default=4)
    parser.add_argument("--palace-bin", default=DEFAULT_PALACE_PATH)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not 1 <= args.mpi_procs <= 15:
        parser.error("--mpi-procs must be between 1 and 15")

    output_root = Path(args.output_root)
    data_log = output_root / "training_data" / "active_learning_log.csv"
    checkpoint = output_root / "training_data" / "checkpoints" / "nemo_surrogate.mdlus"
    surrogate = PhysicsNeMoSurrogate(
        n_features=4,
        data_log_path=str(data_log),
        checkpoint_path=str(checkpoint),
    )
    design = create_transmon_design()
    evaluator = SqdmetalCapacitanceRunner(
        output_root=str(output_root / "data" / "palace_runs"),
        palace_bin=args.palace_bin,
        n_procs=args.mpi_procs,
    )
    samples = np.random.default_rng(args.seed).uniform(
        BOUNDS[:, 0], BOUNDS[:, 1], size=(args.samples, len(BOUNDS))
    )
    for index, sample in enumerate(samples):
        update_qiskit_geometry(design, sample, PARAM_NAMES)
        ej_mhz, ec_mhz = evaluator.evaluate(design, sample, f"sample_{index:04d}")
        surrogate.log_and_append_sample(sample, [ej_mhz, ec_mhz])
        print(f"Sample {index + 1}/{args.samples}: Ej={ej_mhz:.3f} MHz, Ec={ec_mhz:.3f} MHz")

    print(f"Training data: {data_log}")
    print(f"Samples available: {len(surrogate.X_train)}")


if __name__ == "__main__":
    main()