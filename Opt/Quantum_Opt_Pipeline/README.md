# Quantum Optimization Pipeline

A hybrid CAD and machine learning framework for superconducting transmon qubit design optimization. The pipeline integrates **Qiskit Metal** parameterization, **SQDMetal** + **AWS Palace** finite-element capacitance simulation, an **NVIDIA PhysicsNeMo** surrogate neural network, and a **real-valued Genetic Algorithm (GA)**.

---

## Quick Start Guide

Execute all commands from the repository root (`Opt/Quantum_Opt_Pipeline`).

### Step 1: Environment Setup

Choose between containerized execution (Docker) or a local virtual environment:

```bash
# Option A: Containerized (Docker - recommended for reproducibility)
./setup_env.sh --docker
./run_container.sh pytest -q

# Option B: Native Virtual Environment (Python 3.11/3.12)
./setup_env.sh
source "${QUANTUM_DESIGN_ENV:-../../quantum_design_env}/.venv/bin/activate" 2>/dev/null || \
    source "./quantum_design_env/.venv/bin/activate"
```
- **What to expect**: Prepares a lean environment with `torch`, `physicsnemo`, `quantum-metal`, `SQDMetal`, and `gmsh` in 1–2 minutes. Unnecessary bloatware (PySide6/Qt, Jupyter, Ansys) is excluded.

---

### Step 2: Generate Training Samples via Palace

```bash
export PALACE_BIN="/path/to/palace"

python generate_samples.py \
    --output-root training_run \
    --samples 150 \
    --palace-bin "$PALACE_BIN"
```
- **What to expect**: Explores the 4D parameter box via Latin Hypercube Sampling. Reserves 10% of CPU cores for host OS stability and runs Palace across the remaining cores via OpenMPI (~10–15s per sample).
- **Outputs**: Produces `active_learning_log.csv` and auto-splits data into 70% `train_samples.csv`, 10% `validation_samples.csv`, and 20% `test_samples.csv`. Transient 3D field files (`.vtu`) are deleted automatically to save disk space.

---

### Step 3: Train PhysicsNeMo Surrogate

```bash
python train_surrogate.py \
    --data-log training_run/training_data/train_samples.csv \
    --test-log training_run/training_data/test_samples.csv \
    --checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --epochs 2000 \
    --early-stopping-patience 200 \
    --evaluation-output training_run/training_data/checkpoints/training_evaluation.json
```
- **What to expect**: Fits the PhysicsNeMo GNN on $\\log(E_j)$ and $\\log(L_j)$ in ~15–45s. Stops automatically when validation loss plateaus and restores the best weights.
- **Outputs**: Writes `nemo_surrogate.mdlus`, `nemo_surrogate.state.pt`, and `training_evaluation.json` (reporting relative accuracy within 5%, MAE, and independent test score).

---

### Step 4: Run Genetic Algorithm Optimization

```bash
python optimize.py \
    --output-root optimization_run \
    --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_run/training_data/train_samples.csv \
    --population 12 \
    --generations 10 \
    --palace-bin "$PALACE_BIN"
```
- **What to expect**: Evaluates 120 candidate designs in ~2–5 seconds using the neural surrogate for real-time scoring.
- **Outputs**: Writes winning parameters ($pad\\_width$, $pad\\_height$, $pad\\_gap$, $L_j$) and predicted frequencies to `optimization_run/optimization_result.json`. Add `--use-palace` to run a live Palace simulation on the final winner.

---

### Step 5: Run Automated Tests

```bash
python -m pytest -q
```
- **What to expect**: Runs 15 unit tests in ~3 seconds, verifying genetic operators, math transforms, MPI bounds, and script syntax.

---

## Architecture Flow

```
[ Parameter Bounds ] ---> [ Latin Hypercube Sampling ]
                                    |
                                    v
                          [ Qiskit Metal Design ]
                                    |
                                    v
                            [ Gmsh 3D Mesh ]
                                    |
                                    v
                       [ AWS Palace EM Simulation ]
                                    |
                                    v
                      [ 70/10/20 Train/Val/Test Split ]
                                    |
                                    v
                     [ PhysicsNeMo Surrogate (Log-Scale) ]
                                    |
                                    v
                    [ Genetic Algorithm Optimization ]
```

---

## Documentation

Detailed operational guides, mathematical formulations, and reference manuals are organized by topic in [`docs/`](docs/):

| Document | Topic | Description |
| :--- | :--- | :--- |
| **[Quick Start Guide](docs/quickstart.md)** | Step-by-Step | Complete end-to-end tutorial with detailed phase expectations and outputs. |
| **[Environment Setup](docs/setup.md)** | Setup & Dependencies | Docker build & run commands, native virtualenv, and lean dependency profile. |
| **[MPI Resource Policy](docs/mpi_policy.md)** | Resource Management | $\\lceil 0.90 \\times \\text{cores} \\rceil$ allocation reserving 10% for OS background tasks. |
| **[Sample Generation](docs/sampling.md)** | Design Exploration | Parameter bounds, Latin Hypercube Sampling, and 70/10/20 data partitioning. |
| **[Surrogate Modeling](docs/surrogate.md)** | Neural Network | PhysicsNeMo GNN architecture, log-scale physics transforms, and training. |
| **[Genetic Optimization](docs/optimization.md)** | Design Search | Real-valued GA engine, objective cost function, operators, and Palace validation. |
| **[Troubleshooting Guide](docs/troubleshooting.md)** | Diagnostics & FAQ | Solutions for missing libraries, solver errors, and runtime issues. |
| **[Next Version Roadmap](docs/roadmap.md)** | Research Roadmap | Plans for mesh-aware graph neural networks and physics-informed losses. |
