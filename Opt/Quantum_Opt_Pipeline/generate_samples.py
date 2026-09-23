"""Generate Palace measurements for offline PhysicsNeMo surrogate training."""

import argparse
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from src.cad import SqdmetalCapacitanceRunner, create_transmon_design
from src.surrogate import PhysicsNeMoSurrogate
from src.palace import (
    DEFAULT_PALACE_PATH,
    default_mpi_procs,
    max_mpi_procs,
    update_qiskit_geometry,
)


BOUNDS = np.array([
    [300.0, 600.0],
    [20.0, 80.0],
    [10.0, 40.0],
    [6.0e-9, 14.0e-9],
])
PARAM_NAMES = ["Q1.pad_width", "Q1.pad_height", "Q1.pad_gap", "lj"]


def boundary_samples(bounds: np.ndarray) -> np.ndarray:
    """Return all low/high corners of the parameter box."""
    bounds = np.asarray(bounds, dtype=float)
    corners = np.array(np.meshgrid(*bounds, indexing="ij"))
    return corners.reshape(len(bounds), -1).T


def latin_hypercube_samples(
    rng: np.random.Generator,
    bounds: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Generate a randomized Latin-hypercube design across parameter bounds."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (n_parameters, 2)")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("each bound must have a positive width")

    unit_samples = np.empty((sample_count, bounds.shape[0]), dtype=float)
    strata = np.arange(sample_count, dtype=float)
    for parameter_index in range(bounds.shape[0]):
        unit_samples[:, parameter_index] = (
            strata + rng.random(sample_count)
        ) / sample_count
        rng.shuffle(unit_samples[:, parameter_index])
    return bounds[:, 0] + unit_samples * (bounds[:, 1] - bounds[:, 0])


def split_sample_indices(
    rng: np.random.Generator,
    sample_count: int,
    training_percent: float,
    validation_percent: float,
    test_percent: float,
) -> np.ndarray:
    """Assign every sample to a split using percentage counts rounded up."""
    percentages = np.array(
        [training_percent, validation_percent, test_percent], dtype=float
    )
    if np.any(percentages < 0) or not math.isclose(percentages.sum(), 100.0):
        raise ValueError("training, validation, and test percentages must sum to 100")
    counts = np.ceil(sample_count * percentages / 100.0).astype(int)
    counts[0] = sample_count - counts[1] - counts[2]
    if counts[0] < 1:
        raise ValueError("rounded validation and test splits leave no training samples")

    assignments = np.repeat(("train", "validation", "test"), counts)
    rng.shuffle(assignments)
    return assignments


def write_sample_splits(
    data_log: Path,
    split_manifest: Path,
    output_root: Path,
    assignments: np.ndarray,
) -> None:
    """Write a manifest and separate CSV for each generated sample split."""
    frame = np.loadtxt(data_log, delimiter=",", skiprows=1)
    frame = np.atleast_2d(frame)
    columns = [f"param_{i}" for i in range(frame.shape[1] - 2)] + ["Ej_MHz", "Ec_MHz"]
    measured = pd.DataFrame(frame, columns=columns)
    if len(measured) != len(assignments):
        raise RuntimeError("sample split count does not match measured data")
    measured.insert(0, "split", assignments)
    split_root = output_root / "training_data"
    split_root.mkdir(parents=True, exist_ok=True)
    measured.to_csv(split_manifest, index=False)
    for split_name in ("train", "validation", "test"):
        measured.loc[measured["split"] == split_name].drop(columns="split").to_csv(
            split_root / f"{split_name}_samples.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="training_run")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducibility seed; omitted means a fresh random seed",
    )
    parser.add_argument("--training-percent", type=float, default=70.0)
    parser.add_argument("--validation-percent", type=float, default=10.0)
    parser.add_argument("--test-percent", type=float, default=20.0)
    parser.add_argument(
        "--include-boundary-points",
        action="store_true",
        help="Replace the first samples with all low/high parameter corners",
    )
    parser.add_argument(
        "--keep-visualization",
        action="store_true",
        help="Keep Palace Paraview and diagnostic image files",
    )
    parser.add_argument("--mpi-procs", type=int, default=default_mpi_procs())
    parser.add_argument("--palace-bin", default=DEFAULT_PALACE_PATH)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not math.isclose(
        args.training_percent + args.validation_percent + args.test_percent, 100.0
    ):
        parser.error("training, validation, and test percentages must sum to 100")
    if not 1 <= args.mpi_procs <= max_mpi_procs():
        parser.error(
            f"--mpi-procs must be between 1 and {max_mpi_procs()} "
            "(90% of logical CPUs, rounded up)"
        )

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
        retain_visualization=args.keep_visualization,
    )
    seed = args.seed
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    rng = np.random.default_rng(seed)
    print(f"Sampling seed: {seed}")
    samples = latin_hypercube_samples(rng, BOUNDS, args.samples)
    if args.include_boundary_points:
        corners = boundary_samples(BOUNDS)
        if len(samples) < len(corners):
            parser.error("--samples must be at least 16 with boundary points enabled")
        samples[:len(corners)] = corners
    assignments = split_sample_indices(
        rng,
        args.samples,
        args.training_percent,
        args.validation_percent,
        args.test_percent,
    )
    for index, sample in enumerate(samples):
        update_qiskit_geometry(design, sample, PARAM_NAMES)
        ej_mhz, ec_mhz = evaluator.evaluate(design, sample, f"sample_{index:04d}")
        surrogate.log_and_append_sample(sample, [ej_mhz, ec_mhz])
        print(f"Sample {index + 1}/{args.samples}: Ej={ej_mhz:.3f} MHz, Ec={ec_mhz:.3f} MHz")

    print(f"Training data: {data_log}")
    print(f"Samples available: {len(surrogate.X_train)}")
    split_manifest = output_root / "training_data" / "sample_splits.csv"
    write_sample_splits(data_log, split_manifest, output_root, assignments)
    print(f"Sample splits: {split_manifest}")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "sampling_seed": seed,
        "sample_count": args.samples,
        "training_percent": args.training_percent,
        "validation_percent": args.validation_percent,
        "test_percent": args.test_percent,
        "mpi_procs": args.mpi_procs,
        "palace_bin": args.palace_bin,
        "retain_visualization": args.keep_visualization,
        "include_boundary_points": args.include_boundary_points,
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip() or None,
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()