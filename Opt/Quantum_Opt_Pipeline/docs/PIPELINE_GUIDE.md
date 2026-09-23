# Quantum Optimization Pipeline Guide

This guide is the detailed reference for the CDAC quantum optimization pipeline.
The repository README contains the short Quick Start.
Future architecture work is tracked in [NEXT_VERSION.md](NEXT_VERSION.md).

## 1. Pipeline Overview

The pipeline has four operational phases:

1. **Sample generation**: Qiskit Metal creates or updates the transmon geometry.
   SQDMetal generates a mesh and runs Palace for each parameter sample.
2. **Surrogate training**: PhysicsNeMo trains a two-output model for `Ej` and
   `Ec` from measured Palace rows.
3. **Independent evaluation**: A frozen checkpoint can be scored against a CSV
   that was not used for fitting.
4. **Optimization**: A genetic algorithm scores candidates with the frozen
   surrogate and optionally sends the final candidate to Palace.

A normal production workflow is:

```text
Latin-hypercube samples -> Palace measurements -> split CSVs
                                      |
                                      v
                         train -> validate -> test
                                      |
                                      v
                         frozen surrogate -> GA -> optional Palace check
```

## 2. Environment Setup

The project requires Python `>=3.11,<3.13`. The recommended environment is the
existing `quantum_design_env/.venv` beside this project.

```bash
export CDAC_ROOT="$(cd ../.. && pwd)"
export PIPELINE_ROOT="$CDAC_ROOT/Opt/Quantum_Opt_Pipeline"
export ENV_ROOT="$CDAC_ROOT/quantum_design_env"
python3.11 -m venv "$ENV_ROOT/.venv"
source "$ENV_ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
```

Install CPU dependencies:

```bash
python -m pip install -r "$PIPELINE_ROOT/requirements-cpu.txt"
python -m pip install -e "$ENV_ROOT/quantum-metal[mesh]"
python -m pip install -e "$ENV_ROOT/SQDMetal"
python -m pip install -e "$PIPELINE_ROOT" --no-deps
```

For an NVIDIA machine, use the project CUDA requirements instead of the CPU
requirements when selecting PyTorch:

```bash
python -m pip install -r "$PIPELINE_ROOT/requirements.txt"
```

Verify the important imports:

```bash
python -c "import physicsnemo, qiskit_metal, gmsh; print('imports ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The runtime also needs Palace, `mpirun`, and Gmsh:

```bash
export PALACE_BIN="/absolute/path/to/palace"
"$PALACE_BIN" --help >/dev/null
mpirun --version
gmsh --version
```

## 3. MPI Resource Policy

The default Palace rank count is calculated as:

```text
ceil(0.90 * logical_cpu_count)
```

There is no fixed 15-rank cap. On an 11-logical-core machine the default is 10
MPI ranks. The policy is intended to leave approximately 10% of logical CPU
capacity for normal operating-system tasks.

Check the value:

```bash
python -c "from src.palace import max_mpi_procs; print(max_mpi_procs())"
```

Override it explicitly only when needed:

```bash
python generate_samples.py --mpi-procs 10 ...
```

The value must be a positive integer no greater than the machine-aware limit.
MPI parallelizes each Palace solve; samples themselves are processed
sequentially by the generator.

## 4. Generating Samples

Basic command:

```bash
python generate_samples.py \
    --output-root training_run \
    --samples 150 \
    --palace-bin "$PALACE_BIN"
```

Each sample is a four-parameter vector:

| Parameter       |         Range | Unit |
| --------------- | ------------: | ---- |
| `Q1.pad_width`  |    300 to 600 | um   |
| `Q1.pad_height` |      20 to 80 | um   |
| `Q1.pad_gap`    |      10 to 40 | um   |
| `lj`            | 6e-9 to 14e-9 | H    |

### Sampling behavior

Samples use randomized Latin-hypercube sampling. Each parameter receives one
sample in every equal-width stratum, which gives better marginal coverage than
independent uniform random draws.

The default seed is generated from NumPy entropy at runtime. Therefore two runs
without `--seed` produce different sample plans. Use an explicit seed only when
reproducing a run:

```bash
python generate_samples.py --samples 150 --seed 123456 ...
```

Boundary coverage is optional. It replaces the first 16 samples with all low/high
corners of the four-dimensional design box:

```bash
python generate_samples.py \
    --samples 150 \
    --include-boundary-points \
    ...
```

At least 16 samples are required when boundary coverage is enabled.

### Train, validation, and test splits

The default percentages are:

```text
training:   70%
validation: 10%
test:       20%
```

Validation and test counts are rounded up with `ceil`. The training count is the
remaining count, so all samples are assigned exactly once. For 150 samples this
produces 105 training, 15 validation, and 30 test samples.

Override the split:

```bash
python generate_training_data.py \
    --training-percent 70 \
    --validation-percent 10 \
    --test-percent 20 \
    ...
```

The percentages must sum to 100. The assignment is shuffled with the same
runtime random generator used for the sample plan.

### Generated files

The generator writes:

```text
training_run/training_data/active_learning_log.csv
training_run/training_data/sample_splits.csv
training_run/training_data/train_samples.csv
training_run/training_data/validation_samples.csv
training_run/training_data/test_samples.csv
training_run/run_metadata.json
training_run/data/palace_runs/sample_XXXX/
```

`active_learning_log.csv` is the combined measured dataset. Each row contains
`param_0` through `param_3`, `Ej_MHz`, and `Ec_MHz`.

`sample_splits.csv` adds a `split` column to the measured rows. The three
split-specific CSVs remove that column and are ready for model tooling.

`run_metadata.json` records the UTC timestamp, Python executable and version,
sampling seed, split percentages, MPI count, Palace path, visualization policy,
boundary coverage setting, and git revision when available.

Each Palace sample directory normally contains a mesh, Palace configuration,
solver logs, and `outputFiles/terminal-C.csv`. Paraview fields and diagnostic
PNG files are removed after the capacitance result is read, unless
`--keep-visualization` is passed.

Keep visualization output only when needed for inspection:

```bash
python generate_training_data.py --keep-visualization ...
```

The visualization files are large. A previous run used roughly 70-80 MB per
sample with Paraview output retained.

## 5. Training the Surrogate

The model predicts both `Ej` and `Ec`; `Ej` is not calculated analytically in
the GA.

The training implementation uses:

- Log-space `Ej` targets (`log_ej_v1`)
- Log-space `lj` input feature (`log_lj_v1`)
- Typed parameter-node features so the network distinguishes geometry from `lj`
- Two-output regression for `[Ej, Ec]`
- Dropout layers in edge, node, and readout blocks
- Monte Carlo dropout at prediction time for uncertainty estimates
- AdamW optimization with weight decay
- Validation-loss early stopping and best-weight restoration

Train from a dedicated training CSV and evaluate an untouched test CSV:

```bash
python train_surrogate.py \
    --data-log training_run/training_data/train_samples.csv \
    --test-log training_run/training_data/test_samples.csv \
    --checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --epochs 2000 \
    --early-stopping-patience 200 \
    --evaluation-output training_run/training_data/checkpoints/training_evaluation.json \
    --test-output training_run/training_data/checkpoints/test_evaluation.json
```

Important: `--test-log` is evaluated only after fitting and must not be included
in `--data-log`. The current CLI performs an internal validation split from the
training CSV. The generated `validation_samples.csv` is available as a separate
artifact for external evaluation or future split-aware training workflows.

Optional controls:

```bash
--validation-split 0.1
--accuracy-tolerance-percent 5
--evaluation-seed 123456
--ej-threshold-mhz 22000 --ec-threshold-mhz 400
```

Omit `--evaluation-seed` for a fresh validation split each run. Use it only for
reproducibility.

### Training outputs

The checkpoint directory contains:

```text
nemo_surrogate.mdlus
nemo_surrogate.state.pt
nemo_surrogate.evaluation.json
training_evaluation.json       # when --evaluation-output is supplied
test_evaluation.json            # when --test-output is supplied
```

The state file contains normalization ranges, optimizer state, transform
versions, and model version. Incompatible older checkpoints are ignored rather
than loaded silently.

### Metrics

The report includes:

- MSE, RMSE, and MAE
- Separate `Ej` and `Ec` MAE in MHz
- Percentage within the configured relative-error tolerance
- Precision, recall, and F1 using optional target thresholds
- Training and validation sample counts
- Actual epochs run before early stopping
- Independent test metrics when `--test-log` is provided

## 6. Optimization

Run prediction-only optimization with a trained checkpoint:

```bash
python optimize.py \
    --output-root optimization_run \
    --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_run/training_data/train_samples.csv \
    --population 12 \
    --generations 10 \
    --palace-bin "$PALACE_BIN"
```

What happens:

1. Qiskit Metal creates the starting transmon design.
2. The GA creates a random population inside the four bounds.
3. The frozen surrogate predicts `Ej` and `Ec` for each candidate.
4. The cost function ranks the candidates.
5. Tournament selection, SBX crossover, and polynomial mutation create the next
   generation.
6. The best candidate is written to `optimization_result.json`.
7. With default options, the result is surrogate-only and no Palace evaluation
   is run during optimization.

Request final Palace validation explicitly:

```bash
python optimize.py \
    --output-root optimization_run \
    --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_run/training_data/train_samples.csv \
    --use-palace \
    --palace-bin "$PALACE_BIN"
```

`--train-surrogate` is retained for compatibility but training remains a
separate offline phase. The optimizer writes a manifest containing parameters,
predicted metrics, optional measured metrics, cost, population size, generation
count, and model paths.

## 7. Testing and Quality Checks

Run the complete tests:

```bash
python -m pytest -q
```

Run syntax and whitespace checks:

```bash
python -m compileall -q pipeline.py generate_samples.py optimize.py train_surrogate.py src tests
git diff --check
```

The tests cover bounds, MPI policy, Latin-hypercube strata, split rounding,
boundary corners, target transforms, dropout behavior, geometry updates,
capacitance parsing, and the workflow smoke path.

## 8. Disk and Process Management

Before a large run:

```bash
df -h .
du -sh training_run 2>/dev/null || true
```

Use the default visualization cleanup to keep storage manageable. Do not start a
second generator against the same output root. If a run is interrupted, inspect
the CSV row count and sample directories before resuming; the generator does not
currently deduplicate an already completed sample automatically.

The expensive step is Palace generation. Training on a few hundred CSV rows is
usually much faster than producing those rows. Preserve the CSV and checkpoint
if the mesh and Paraview artifacts are no longer needed.

## 9. Troubleshooting

### Missing `qiskit_metal`

Use the Python executable from `quantum_design_env/.venv` and install the local
`quantum-metal[mesh]` project. The pipeline requires Python 3.11 or 3.12.

### Missing Palace, MPI, or Gmsh

Check `PALACE_BIN`, `mpirun --version`, and `gmsh --version`. Pass an absolute
Palace path with `--palace-bin` to avoid PATH ambiguity.

### MPI rank rejection

The requested rank count is above 90% of detected logical CPUs rounded up. Lower
`--mpi-procs`, or omit it and use the calculated default.

### Missing `terminal-C.csv`

Inspect the sample's `outputFiles/out.log`. A missing capacitance result stops the
run because the sample cannot be trusted.

### Repeated Paraview plotting warning

`Error in plotting: 'Data array (V) not present in this dataset.'` is a
visualization warning from the Palace postprocessor. The capacitance result can
still be valid when `terminal-C.csv` exists and the solver log shows convergence.
Use the default visualization cleanup if those plots are not needed.

### Low validation or test accuracy

Check the split files, sample ranges, target distributions, and independent test
metrics. Increase Palace samples with Latin-hypercube or boundary coverage
before increasing epochs indefinitely. Use the saved run metadata to reproduce
the environment and sample plan.
