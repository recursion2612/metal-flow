# Quantum Optimization Pipeline Guide

This guide provides operational details, mathematical formulations, and reference configurations for the CDAC Quantum Optimization Pipeline.

---

## 1. Pipeline Overview

The pipeline automates the simulation and optimization cycle for planar transmon qubits through four main phases:

```
[ Latin Hypercube Sampling ] ---> [ Gmsh 3D Meshing ] ---> [ AWS Palace Solves ]
                                                                   |
                                                                   v
                                                      [ Train / Val / Test CSVs ]
                                                                   |
                                                                   v
                                                      [ PhysicsNeMo Surrogate ]
                                                                   |
                                                                   v
                                                      [ GA Optimization Engine ]
```

1. **Sample Generation (`generate_samples.py`)**: Explores parametric qubit design space via randomized Latin Hypercube Sampling (LHS), builds conformal 3D meshes in Gmsh, and extracts Maxwell capacitance matrices using AWS Palace.
2. **Surrogate Training (`train_surrogate.py`)**: Fits a custom graph neural network surrogate in NVIDIA PhysicsNeMo using log-transformed targets $\\log(E_j)$ and features $\\log(L_j)$.
3. **Model Evaluation**: Employs validation-based early stopping and evaluates prediction error on an untouched 20% holdout test partition.
4. **Genetic Optimization (`optimize.py`)**: Executes tournament selection, simulated binary crossover (SBX), and polynomial mutation to discover optimal geometries targeting desired qubit Hamiltonian frequencies.

---

## 2. Environment Setup

The pipeline requires **Python 3.11 or 3.12**.

### Option A: Containerized Setup (Docker - Recommended)

Build the minimal Docker image:

```bash
./setup_env.sh --docker
```

Run test suite or pipeline commands inside the container:

```bash
# Run tests
./run_container.sh pytest -q

# Run sample generation
./run_container.sh python generate_samples.py --samples 50

# Interactive shell
./run_container.sh
```

To bind-mount a host Palace executable:
```bash
export PALACE_BIN="/path/to/palace"
./run_container.sh python generate_samples.py --palace-bin "$PALACE_BIN" ...
```

### Option B: Native Virtual Environment

Run the setup script:

```bash
./setup_env.sh
source "${QUANTUM_DESIGN_ENV:-../../quantum_design_env}/.venv/bin/activate" 2>/dev/null || \
    source "./quantum_design_env/.venv/bin/activate"
```

For NVIDIA GPU acceleration:
```bash
./setup_env.sh --cuda
```

### Lean Dependency Profile
To ensure fast installations and prevent disk exhaustion, heavy interactive libraries (PySide6, Jupyter, Ansys backends, Streamlit, torchvision) are omitted:
- **Core Runtime**: `numpy`, `pandas`, `torch`, `nvidia-physicsnemo`, `quantum-metal[mesh]`, `SQDMetal` (minimal runtime: `mph`, `pyvista`), `gmsh`, and `pytest`.
- **System Tools**: `mpirun` (OpenMPI), `gmsh`, and `palace`.

Verify runtime imports:
```bash
python -c "import physicsnemo, qiskit_metal, SQDMetal, gmsh; print('Imports verified successfully')"
python -c "import torch; print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
```

---

## 3. Resource Management & MPI Allocation

AWS Palace solves large finite-element linear systems in parallel using MPI. To preserve system stability, the pipeline reserves at least 10% of logical CPU cores for host OS background tasks:

$$\\text{MPI Ranks} = \\max\\left(1, \\left\\lceil 0.90 \\times \\text{logical\\_cpu\\_count} \\right\\rceil\\right)$$

On an 11-core system, Palace automatically defaults to 10 ranks.

Check machine rank limit:
```bash
python -c "from src.palace import max_mpi_procs; print(f'Safe MPI rank limit: {max_mpi_procs()}')"
```

MPI parallelizes each individual finite-element solve. Parameter samples are evaluated sequentially.

---

## 4. Sample Generation (`generate_samples.py`)

### Parameter Bounds

| Parameter | Key | Range | Unit |
| :--- | :--- | :--- | :--- |
| Transmon Pad Width | `Q1.pad_width` | 300 to 600 | $\\mu\\text{m}$ |
| Transmon Pad Height | `Q1.pad_height` | 20 to 80 | $\\mu\\text{m}$ |
| Ground Pocket Gap | `Q1.pad_gap` | 10 to 40 | $\\mu\\text{m}$ |
| Josephson Inductance | `lj` | 6.0 to 14.0 | $\\text{nH}$ |

### Sampling & Partitioning
- **Latin Hypercube Sampling**: Stratifies parameter intervals equally for optimal space-filling coverage.
- **Randomized Seeds**: Generates fresh entropy per invocation by default; use `--seed <int>` for deterministic reproduction.
- **Boundary Sampling**: Pass `--include-boundary-points` to populate the first 16 samples with every combination of minimum/maximum parameter bounds.
- **Ceiling Holdout Splitting**: Splits data into 70% train, 10% validation, and 20% test (holdout counts rounded up using `ceil`).

```bash
python generate_samples.py \
    --output-root training_run \
    --samples 150 \
    --palace-bin "$PALACE_BIN"
```

### Generated Files
- `training_data/active_learning_log.csv`: Consolidated dataset containing `param_0`..`param_3`, `Ej_MHz`, and `Ec_MHz`.
- `training_data/train_samples.csv`: 70% partition for surrogate model fitting.
- `training_data/validation_samples.csv`: 10% partition for model selection.
- `training_data/test_samples.csv`: 20% holdout partition for independent evaluation.
- `training_data/sample_splits.csv`: Combined manifest tracking assigned split per row.
- `run_metadata.json`: Audit log containing timestamps, git commit, seeds, and runtime versions.

---

## 5. Surrogate Modeling (`train_surrogate.py`)

### Architecture & Physics Transforms
The surrogate model learns electromagnetic relationships from scalar geometries:
- **Log-Space Targets**: Fits $\\log(E_j)$ to linearize the inverse relationship $E_j \\propto 1/L_j$.
- **Log-Space Inputs**: Feeds $\\log(L_j)$ into the graph network to improve gradient stability.
- **Node Type Encodings**: Parameter identity features allow the network to distinguish geometric coordinates from circuit inductance.
- **Monte Carlo Dropout**: Regularizes training and provides predictive uncertainty during genetic algorithm runs.
- **Early Stopping**: Restores model weights from the epoch with minimal validation loss.

```bash
python train_surrogate.py \
    --data-log training_run/training_data/train_samples.csv \
    --test-log training_run/training_data/test_samples.csv \
    --checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --epochs 2000 \
    --early-stopping-patience 200 \
    --evaluation-output training_run/training_data/checkpoints/training_evaluation.json
```

### Outputs
- `nemo_surrogate.mdlus`: PhysicsNeMo model architecture and weights.
- `nemo_surrogate.state.pt`: Normalization statistics, optimizer states, and version tags.
- `training_evaluation.json`: Comprehensive report containing MSE, RMSE, MAE, relative accuracy within 5%, precision, recall, and F1 scores.

---

## 6. Genetic Optimization (`optimize.py`)

Searches the continuous design space to hit target Hamiltonian energies ($E_j^*, E_c^*$).

### Optimization Process
1. Initializes a population of continuous parameter vectors within fabrication limits.
2. Evaluates candidates using the frozen PhysicsNeMo surrogate.
3. Ranks candidates via quadratic cost function:
   $$\\text{Cost} = \\left(\\frac{E_j - E_j^*}{E_j^*}\\right)^2 + \\left(\\frac{E_c - E_c^*}{E_c^*}\\right)^2$$
4. Applies tournament selection, Simulated Binary Crossover (SBX), and polynomial mutation ($p_m = 0.25$).
5. Automatically clones top elites into the next generation.
6. Writes winning geometry and parameters to `optimization_result.json`.

```bash
python optimize.py \
    --output-root optimization_run \
    --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_run/training_data/train_samples.csv \
    --population 12 \
    --generations 10 \
    --palace-bin "$PALACE_BIN"
```

Add `--use-palace` to validate the final winning design via a live Palace simulation solve.

---

## 7. Verification & Quality Checks

Run the complete test suite:
```bash
python -m pytest -q
```

Validate syntax, type hygiene, and formatting:
```bash
python -m compileall -q pipeline.py generate_samples.py optimize.py train_surrogate.py src tests
git diff --check
```

---

## 8. Disk & Process Management

- Intermediate 3D Paraview field meshes (`.vtu`) are deleted automatically after capacitance extraction, preserving disk space. Pass `--keep-visualization` only when visual field debugging is required.
- Do not run concurrent generators targeting the same output directory.
- Verify available storage before launching large simulation batches:
  ```bash
  df -h .
  ```

---

## 9. Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| `Missing qiskit_metal` | Incorrect Python environment | Activate the Python 3.11 virtual environment (`source quantum_design_env/.venv/bin/activate`). |
| `mpirun: command not found` | OpenMPI not installed | Install OpenMPI (`openmpi-bin` on Linux, `brew install open-mpi` on macOS) or use Docker. |
| `MPI rank rejection error` | `--mpi-procs` exceeds 90% of cores | Omit `--mpi-procs` to use automatic machine default. |
| `terminal-C.csv missing` | Palace solve failure or geometry self-intersection | Inspect `out.log` in `data/palace_runs/sample_XXXX/outputFiles/`. |
| `Data array (V) not present` | Non-fatal Paraview field warning from Palace | Safe to ignore; capacitance value is extracted from `terminal-C.csv`. |
