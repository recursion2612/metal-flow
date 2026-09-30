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
