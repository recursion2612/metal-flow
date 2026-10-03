# Troubleshooting Guide & FAQ

This guide resolves common runtime, environment, and solver issues encountered when operating the pipeline.

---

## Environment & Dependency Issues

### 1. `ModuleNotFoundError: No module named 'qiskit_metal'`
- **Cause**: Active Python interpreter does not have `quantum-metal` installed, or Python version is `>=3.13`.
- **Resolution**: Activate the designated virtual environment:
  ```bash
  source "${QUANTUM_DESIGN_ENV:-../../quantum_design_env}/.venv/bin/activate"
  ```
  Ensure Python is 3.11 or 3.12:
  ```bash
  python -c "import sys; assert sys.version_info[:2] in [(3, 11), (3, 12)]"
  ```

### 2. `mpirun: command not found` or `gmsh: command not found`
- **Cause**: OpenMPI or Gmsh system binaries are missing from your system `PATH`.
- **Resolution**:
  - **Linux (Ubuntu/Debian)**: `sudo apt-get install -y openmpi-bin gmsh`
  - **macOS (Homebrew)**: `brew install open-mpi gmsh`
  - **Docker**: Run via container (`./setup_env.sh --docker && ./run_container.sh ...`) where all binaries are pre-installed.

---

## Palace & Solver Issues

### 3. `ValueError: Palace MPI processes must be between 1 and X on this machine`
- **Cause**: The value passed to `--mpi-procs` exceeds the 90% core allocation ceiling.
- **Resolution**: Omit `--mpi-procs` to allow the pipeline to automatically calculate the safe maximum for your CPU, or pass a value $\le \lceil 0.90 \times \text{cores} \rceil$.

### 4. `FileNotFoundError: SQDMetal did not produce terminal-C.csv`
- **Cause**: Palace solver execution failed or exited prematurely.
- **Resolution**: Inspect the solver output log:
  ```bash
  tail -n 100 <output-root>/data/palace_runs/sample_XXXX/outputFiles/out.log
  ```
  Check `palace.stderr.log` for geometry self-intersections or linear solver divergence.

### 5. `Error in plotting: 'Data array (V) not present in this dataset.'`
- **Cause**: Non-fatal visualization postprocessing notice emitted by SQDMetal after electrostatics solve.
- **Resolution**: Safe to ignore. The simulation outputs and Maxwell capacitance matrix are correctly extracted from `terminal-C.csv`.

---

## Storage & Performance Issues

### 6. Rapid Disk Space Consumption
- **Cause**: Paraview 3D visualization files (`.vtu`) consume 70–80 MB per sample.
- **Resolution**: Do not pass `--keep-visualization` during production runs. The pipeline automatically deletes field meshes after capacitance matrix extraction by default.

### 7. Simulation Stalling or Slowing Down
- **Cause**: Running multiple concurrent generators against the same machine or disk.
- **Resolution**: Palace already maximizes CPU utilization via OpenMPI. Run generation sequentially and check CPU utilization using `top` or `htop`.

---

## Physical Extraction & Pipeline Data Issues

### 8. $E_c$ Measured at ~19.5 MHz Instead of Target ~320 MHz
- **Cause**: Reading the Maxwell capacitance matrix from Terminal 0 (the macroscopic $2.8\text{ mm} \times 2.0\text{ mm}$ ground plane chip, with self-capacitance $\approx 1.0\text{ pF}$) instead of the differential transmon pads (Terminals 1 & 2, $\approx 60\text{ fF}$).
- **Resolution**: Handled in the latest `src/cad.py` via `extract_transmon_c_sigma()`, which computes:
  $$C_\Sigma = C_{12} + \frac{C_{1,g} \cdot C_{2,g}}{C_{1,g} + C_{2,g}}$$
  For previous simulation runs, execute:
  ```bash
  python recompute_samples.py --training-run-dir <path/to/run>
  ```
  to recalculate true $E_c$ across all raw `terminal-C.csv` files in seconds without re-simulating.

### 9. Sample Split Count Mismatch During Incremental Runs
- **Cause**: Appending new samples into an existing data directory when splitting is restricted to the newly sampled count.
- **Resolution**: `generate_samples.py` automatically consolidates and partitions the total combined pool of older and new samples across `train_samples.csv`, `validation_samples.csv`, and `test_samples.csv`.
