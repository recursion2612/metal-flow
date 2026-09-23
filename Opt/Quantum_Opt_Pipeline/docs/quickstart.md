# Quick Start Guide

This guide walks you through running the complete transmon optimization workflow from start to finish, including **what to expect at each step**.

Execute all commands from the repository root (`Opt/Quantum_Opt_Pipeline`).

---

## Step 1: Environment Setup

Choose between containerized execution (Docker) or a local virtual environment:

```bash
# Option A: Containerized Execution (Docker)
./setup_env.sh --docker
./run_container.sh pytest -q

# Option B: Native Virtual Environment (Python 3.11/3.12)
./setup_env.sh
source "${QUANTUM_DESIGN_ENV:-../../quantum_design_env}/.venv/bin/activate" 2>/dev/null || \
    source "./quantum_design_env/.venv/bin/activate"
```

### What to Expect
- **Duration**: ~1–3 minutes.
- **Output**: Prepares a lean Python environment with `torch`, `physicsnemo`, `quantum-metal`, `SQDMetal`, and `gmsh`.
- **Verification**: Confirms external tools (`mpirun`, `gmsh`, `palace`) are found or warns if missing.

---

## Step 2: Sample Generation via AWS Palace

Generate training data by running electromagnetic capacitance simulations across Latin Hypercube sampled geometries:

```bash
# Set path to Palace executable (or ensure 'palace' is on PATH)
export PALACE_BIN="/path/to/palace"

python generate_samples.py \
    --output-root training_run \
    --samples 150 \
    --palace-bin "$PALACE_BIN"
```

### What to Expect
- **Duration**: ~10–15 seconds per sample (~25–35 minutes for 150 samples on modern multi-core CPUs).
- **Core Allocation**: Automatically reserves 10% of logical CPU cores for host OS stability and runs Palace across the remaining cores via OpenMPI.
- **Artifacts Created**:
  - `training_run/training_data/active_learning_log.csv`: Full measured dataset.
  - `training_run/training_data/train_samples.csv`: 70% partition (105 samples).
  - `training_run/training_data/validation_samples.csv`: 10% partition (15 samples).
  - `training_run/training_data/test_samples.csv`: 20% holdout partition (30 samples).
  - `training_run/run_metadata.json`: Audit log containing timestamp, git commit, seed, and Python version.
- **Storage**: Transient 3D Paraview field data (`.vtu`) is automatically pruned, keeping total disk usage under ~11 GB for 150 samples.

---

## Step 3: Surrogate Model Training

Fit the NVIDIA PhysicsNeMo graph neural network surrogate on the measured training partition:

```bash
python train_surrogate.py \
    --data-log training_run/training_data/train_samples.csv \
    --test-log training_run/training_data/test_samples.csv \
    --checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --epochs 2000 \
    --early-stopping-patience 200 \
    --evaluation-output training_run/training_data/checkpoints/training_evaluation.json
```

### What to Expect
- **Duration**: ~15–45 seconds on CPU or GPU.
- **Model Training**: The network trains on $\log(E_j)$ targets and $\log(L_j)$ features with Monte Carlo dropout.
- **Early Stopping**: Halts training automatically when validation loss stops improving and restores the best checkpoint.
- **Evaluation Output**: Evaluates both internal validation and the untouched 20% holdout test partition (`test_samples.csv`), reporting MAE in MHz and relative accuracy within 5%.
- **Artifacts Created**:
  - `training_data/checkpoints/nemo_surrogate.mdlus`: Model weights.
  - `training_data/checkpoints/nemo_surrogate.state.pt`: Normalization statistics.
  - `training_data/checkpoints/training_evaluation.json`: Full evaluation report.

---

## Step 4: Genetic Algorithm Optimization

Run real-valued Genetic Algorithm optimization to find the best transmon parameters matching target qubit frequencies:

```bash
python optimize.py \
    --output-root optimization_run \
    --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_run/training_data/train_samples.csv \
    --population 100 \
    --generations 20 \
    --palace-bin "$PALACE_BIN"
```

### What to Expect
- **Duration**: ~2–5 seconds (surrogate predictions evaluate in milliseconds).
- **Optimization Process**: Evaluates 100 candidate designs per generation using tournament selection, SBX crossover, and polynomial mutation. Stops early when the best candidate reaches within 5% of target parameters (or up to 20 max generations), simultaneously deleting old generation data to keep disk usage zero.
- **Artifacts Created**:
  - `optimization_run/optimization_result.json`: Summary containing the winning parameters (`pad_width`, `pad_height`, `pad_gap`, and $L_j$), predicted $[E_j, E_c]$ values, and final objective cost.
- **Optional Live Palace Check**: Add `--use-palace` to run an actual Palace finite-element simulation on the winning design to verify surrogate accuracy.

---

## Step 5: Test Suite Verification

Run the automated test suite to ensure all unit tests and environment assertions pass:

```bash
python -m pytest -q
```

### What to Expect
- **Duration**: ~3–5 seconds.
- **Result**: `17 passed` confirming math transforms, genetic operators, MPI rules, and script integrity.

---

## Next Steps

- For in-depth configuration options: see the [Pipeline Guide Index](README.md).
- For parameter ranges and sampling controls: see [Sample Generation](sampling.md).
- For GNN architecture and loss curves: see [Surrogate Modeling](surrogate.md).
- For GA objective formulations: see [Genetic Optimization](optimization.md).
