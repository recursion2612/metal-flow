# Quantum Optimization Pipeline

A hybrid CAD and machine learning pipeline for superconducting transmon qubit design optimization. The framework integrates **Qiskit Metal** parameterization, **SQDMetal** + **AWS Palace** finite-element capacitance simulation, an **NVIDIA PhysicsNeMo** surrogate neural network, and a **real-valued Genetic Algorithm (GA)**.

---

## Quick Start

Execute all commands from the pipeline directory (`Opt/Quantum_Opt_Pipeline`).

### 1. Environment Setup

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

### 2. Full Workflow Example

```bash
# Set Palace executable path (if not in PATH)
export PALACE_BIN="/path/to/palace"

# Step 1: Generate training samples via Palace simulations
python generate_samples.py \
    --output-root training_run \
    --samples 150 \
    --palace-bin "$PALACE_BIN"

# Step 2: Train PhysicsNeMo surrogate model with early stopping
python train_surrogate.py \
    --data-log training_run/training_data/train_samples.csv \
    --test-log training_run/training_data/test_samples.csv \
    --checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --epochs 2000 \
    --early-stopping-patience 200 \
    --evaluation-output training_run/training_data/checkpoints/training_evaluation.json

# Step 3: Run surrogate-assisted genetic optimization
python optimize.py \
    --output-root optimization_run \
    --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_run/training_data/train_samples.csv \
    --population 12 \
    --generations 10 \
    --palace-bin "$PALACE_BIN"

# Step 4: Run test suite
python -m pytest -q
```

---

## Architecture

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

- **PhysicsNeMo Surrogate**: Custom graph neural network architecture with Monte Carlo dropout uncertainty estimation. Trains on log-scaled targets $\\log(E_j)$ and features $\\log(L_j)$ to accurately model non-linear electromagnetic responses.
- **Automated Data Splitting**: Partitions samples into 70% training, 10% validation (model selection/early stopping), and 20% holdout test.
- **Resource Management**: Allocates Palace MPI ranks as `ceil(0.9 * CPU_cores)`, reserving at least 10% of logical CPU cores for host OS stability.
- **Storage Optimization**: Automatically removes transient Paraview field files after capacitance matrix extraction, reducing disk footprint by ~70%.
- **Portable & Containerized**: Zero hardcoded local paths; fully packaged Docker workflow and lean dependency profile.

---

## Command Reference

### Sample Generation (`generate_samples.py`)
Generates transmon geometries, meshes via Gmsh, and runs Palace capacitance simulations.
```bash
python generate_samples.py \
    --output-root <dir> \
    --samples <int> \
    [--training-percent 70] [--validation-percent 10] [--test-percent 20] \
    [--include-boundary-points] \
    [--keep-visualization] \
    [--palace-bin <path>]
```

### Surrogate Training (`train_surrogate.py`)
Trains the PhysicsNeMo neural network on measured data with early stopping.
```bash
python train_surrogate.py \
    --data-log <train_csv> \
    --checkpoint <output_mdlus> \
    [--test-log <test_csv>] \
    [--epochs 2000] \
    [--early-stopping-patience 200] \
    [--evaluation-output <eval_json>]
```

### Genetic Optimization (`optimize.py`)
Optimizes qubit geometry parameters ($pad\_width$, $pad\_height$, $pad\_gap$, $L_j$) using tournament selection, SBX crossover, and polynomial mutation scored by the trained surrogate.
```bash
python optimize.py \
    --output-root <dir> \
    --model-checkpoint <checkpoint_path> \
    --model-data-log <train_csv> \
    [--population 12] \
    [--generations 10] \
    [--use-palace] \
    [--palace-bin <path>]
```

---

## Documentation

Detailed documentation is organized by topic in the [`docs/`](docs/) directory:

- **[Quick Start Guide](docs/quickstart.md)**: Step-by-step walkthrough detailing commands and expected outputs at each phase.
- **[Environment Setup](docs/setup.md)**: Containerized Docker setup, virtual environment, and lean dependency profile.
- **[MPI Resource Policy](docs/mpi_policy.md)**: Core allocation rules, 10% OS capacity reservation, and scaling.
- **[Sample Generation](docs/sampling.md)**: Design bounds, Latin Hypercube Sampling, boundary box coverage, and 70/10/20 splitting.
- **[Surrogate Modeling](docs/surrogate.md)**: PhysicsNeMo GNN architecture, log-scale physics transforms, dropout uncertainty, and training.
- **[Genetic Optimization](docs/optimization.md)**: Real-valued GA engine, objective function, tournament selection, and operators.
- **[Troubleshooting Guide](docs/troubleshooting.md)**: Common failure modes, root causes, and resolutions.
- **[Next Version Roadmap](docs/roadmap.md)**: Mesh-aware graph networks, physics-informed losses, and future plans.

---

## Testing

Run the automated test suite:
```bash
python -m pytest -q
```
