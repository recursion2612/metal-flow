"""Utility to recompute Ec_MHz in active_learning_log.csv from raw Palace terminal-C.csv files."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.cad import extract_transmon_c_sigma
from src.palace import E_CHARGE, H_PLANCK


def recompute_samples(data_dir: Path, training_data_dir: Path) -> pd.DataFrame:
    """Scan palace_runs for terminal-C.csv, recalculate Ec, and update CSV logs."""
    palace_runs = sorted(data_dir.glob("palace_runs/sample_*"))
    if not palace_runs:
        sys.exit(f"Error: No sample_* folders found in {data_dir / 'palace_runs'}")

    log_csv = training_data_dir / "active_learning_log.csv"
    if not log_csv.exists():
        sys.exit(f"Error: {log_csv} not found")

    df = pd.read_csv(log_csv)
    print(f"Found {len(df)} rows in {log_csv.name} and {len(palace_runs)} simulation folders.")

    corrected_ec = []
    corrected_csig = []

    for i, run_dir in enumerate(palace_runs):
        c_csv = run_dir / "outputFiles/terminal-C.csv"
        if not c_csv.exists():
            c_csv = run_dir / "terminal-C.csv"
        if not c_csv.exists():
            print(f"Warning: {c_csv} not found, keeping original Ec for sample {i}")
            corrected_ec.append(df.loc[i, "Ec_MHz"])
            continue

        raw = pd.read_csv(c_csv, index_col=0)
        mat = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        c_sigma = extract_transmon_c_sigma(mat)
        ec = (E_CHARGE**2 / (2.0 * c_sigma * H_PLANCK)) * 1e-6
        corrected_ec.append(ec)
        corrected_csig.append(c_sigma * 1e15)

    old_ec = df["Ec_MHz"].copy()
    df["Ec_MHz"] = corrected_ec[: len(df)]
    df.to_csv(log_csv, index=False)
    print(f"Successfully updated {log_csv}")
    print(f"  Old Ec: min={old_ec.min():.2f} MHz, max={old_ec.max():.2f} MHz, mean={old_ec.mean():.2f} MHz")
    print(f"  New Ec: min={df['Ec_MHz'].min():.2f} MHz, max={df['Ec_MHz'].max():.2f} MHz, mean={df['Ec_MHz'].mean():.2f} MHz")
    if corrected_csig:
        print(f"  C_sigma: min={min(corrected_csig):.2f} fF, max={max(corrected_csig):.2f} fF")

    # Update split files if present
    for split_name in ("train_samples.csv", "validation_samples.csv", "test_samples.csv"):
        split_path = training_data_dir / split_name
        if split_path.exists():
            sdf = pd.read_csv(split_path)
            p_cols = [c for c in sdf.columns if c.startswith("param_")]
            merged = sdf.drop(columns=["Ec_MHz"], errors="ignore").merge(
                df[p_cols + ["Ec_MHz"]], on=p_cols, how="left"
            )
            merged.to_csv(split_path, index=False)
            print(f"Updated split file: {split_name}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute true Ec_MHz in active_learning_log.csv from raw Palace terminal-C.csv files."
    )
    parser.add_argument(
        "--training-run-dir",
        type=str,
        required=True,
        help="Path to training_run directory (containing 'data/' and 'training_data/')",
    )
    args = parser.parse_args()

    run_dir = Path(args.training_run_dir)
    data_dir = run_dir / "data"
    training_data_dir = run_dir / "training_data"
    recompute_samples(data_dir, training_data_dir)


if __name__ == "__main__":
    main()

