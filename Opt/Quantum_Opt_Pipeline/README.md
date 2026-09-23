# Nemo + Palace Genetic Optimization

This pipeline uses Qiskit Metal geometry, SQDMetal's AWS Palace capacitance backend, an NVIDIA PhysicsNeMo `Module` surrogate, and a real-valued genetic algorithm.

## Quick Start

Run these commands from the repository's `Opt/Quantum_Opt_Pipeline` directory.
The full command reference and feature details are in
[docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md).
Future architecture and research improvements are tracked in
[docs/NEXT_VERSION.md](docs/NEXT_VERSION.md).
The setup script installs all Python and local CAD dependencies; Palace itself
is an external runtime configured through `PALACE_BIN`.

```bash
# 1. Set up the compatible environment and install dependencies
./setup_environment.sh
source ../../quantum_design_env/.venv/bin/activate

# 2. Configure Palace and run a small smoke sample
export PALACE_BIN="/absolute/path/to/palace"
python generate_samples.py \
    --output-root training_run \
    --samples 12 \
    --palace-bin "$PALACE_BIN"

# 3. Train and evaluate the surrogate
python train_surrogate.py \
    --data-log training_run/training_data/train_samples.csv \
    --test-log training_run/training_data/test_samples.csv \
    --checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --epochs 2000 \
    --early-stopping-patience 200 \
    --evaluation-output training_run/training_data/checkpoints/training_evaluation.json

# 4. Run frozen-surrogate genetic optimization
python optimize.py \
    --output-root optimization_run \
    --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_run/training_data/train_samples.csv \
    --palace-bin "$PALACE_BIN"

# 5. Run the automated checks
python -m pytest -q
```

The sample phase runs one Palace simulation per sample and can be the longest
step. The training phase writes the model and metrics. The optimization phase
uses the frozen model for GA scoring and writes
`optimization_run/optimization_result.json`; Palace is called for the final
design only when `--use-palace` is supplied.

## Requirements and installation

For setup options, GPU selection, external Palace installation, and runtime
details, see [docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md). The Quick Start
above is the complete short setup path.

The project requires Python `>=3.11,<3.13`. When no usable NVIDIA GPU is
detected, `requirements-cpu.txt` selects the official PyTorch CPU wheel index.
When `nvidia-smi -L` succeeds, the standard requirements install is used so a
CUDA-capable PyTorch wheel can be selected.

The package is installed as `nvidia-physicsnemo` and imported as `physicsnemo`:

```bash
python -c "import physicsnemo; print(physicsnemo.__version__)"
```

For NVIDIA GPU training on a Linux dev node, use a CUDA-compatible PyTorch
environment or the NVIDIA PhysicsNeMo container. Verify the selected runtime
before training:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Install Palace with Spack, or build it using the [Palace installation
guide](https://awslabs.github.io/palace/dev/install/), then expose it through
`PATH` or set `PALACE_BIN`. The command-line `--palace-bin` option has highest
precedence. The node must also provide `mpirun` and `gmsh` on `PATH`.

```bash
spack install palace
export PALACE_BIN="$(spack location -i palace)/bin/palace"
export PATH="$(dirname "$PALACE_BIN"):$PATH"
which mpirun
mpirun --version
gmsh --version
"$PALACE_BIN" --help >/dev/null
```

The optimizer selects MPS, CUDA, or CPU automatically. On a CPU-only Linux
node it uses CPU PyTorch; macOS can use CPU or Apple MPS when available.

## Required integration

The default entry point creates a Qiskit Metal `DesignPlanar`, updates `Q1`, and uses SQDMetal to generate the Gmsh mesh and Palace configuration. SQDMetal derives material, ground, and terminal physical attributes from the generated geometry.

For lower-level use with a compatible Qiskit Metal layer-stack implementation, the direct exporter is available:

```python
from src.cad import export_qiskit_metal_gmsh
export_qiskit_metal_gmsh(design, "design.msh")
```

The generated `.msh` must contain populated `$Nodes` and `$Elements` sections.
For the local `DesignPlanar` implementation, use the SQDMetal runner below; it owns the Palace physical-group mapping.

## Training and optimization flow

1. Generate a randomized Latin-hypercube design inside the four fabrication bounds.
2. Use Monte Carlo dropout predictions from the dropout-enabled surrogate when it is trained and uncertainty is below `1.5`.
3. In active-learning mode, run Palace for uncertain candidates, append measured `Ej` and `Ec` to `training_data/active_learning_log.csv`, and train Nemo after each generation.
4. In prediction-only mode, use the previously trained model for scoring and run Palace only for the final selected design. Select three aspirants per tournament and retain the lowest-cost candidate. Two elites survive each generation.
5. Apply SBX crossover and polynomial mutation. The default mutation probability is `1 / 4 = 0.25` per parameter, a standard starting point for four continuous genes; every child is clipped to its bounds.
6. Save the PhysicsNeMo model to `training_data/checkpoints/nemo_surrogate.mdlus` and optimizer/normalization state to `nemo_surrogate.state.pt`.
7. Select the final winner from measured Palace samples, not only a surrogate prediction.
8. Train `Ej` in log space with log-scaled `lj` features while preserving learned two-output prediction.

Example invocation:

```python
from pipeline import run_pipeline

result = run_pipeline(design=design, output_root="run_01", palace_bin="/absolute/path/to/palace")
```

Complete default run:

```bash
python optimize.py \
    --output-root run_01 \
    --model-checkpoint training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_data/active_learning_log.csv \
    --palace-bin "$PALACE_BIN"
```

The normal optimization workflow uses Nemo predictions to score every GA
population. Train and validate the surrogate separately with
`train_surrogate.py`, then provide its checkpoint and matching data log
above. The optimizer loads that final model, does not call `fit()`, and sends
only the final Nemo-selected design to Palace for ground-truth simulation.
The result manifest contains both predicted and Palace-measured metrics.

To deliberately run the older active-learning behavior, opt in explicitly:

```bash
python optimize.py \
    --output-root run_01 \
    --train-surrogate \
    --use-palace \
    --palace-bin "$PALACE_BIN"
```

`--use-palace` additionally enables Palace evaluations for high-uncertainty
candidates during the generations. The final selected candidate is always
measured by Palace. `--train-surrogate` enables retraining after each
generation and should be used together with `--use-palace`.

## Results

The run returns a Python dictionary and writes `run_01/optimization_result.json` containing the best parameters, measured `Ej`/`Ec`, cost, sample count, mutation rate, training CSV path, and PhysicsNeMo checkpoint path.

## Training the surrogate

Training is a separate offline preparation step. Each Palace result is appended
to `training_data/active_learning_log.csv`; `train_surrogate.py` validates
the measured data, trains the PhysicsNeMo module, and saves the final
checkpoint. The normal optimizer then reloads both the measured CSV and the
`.mdlus` model without retraining.

### Complete training run from new Palace samples

Run all commands from `Opt/Quantum_Opt_Pipeline`. By default, the pipeline uses
90% of the machine's logical CPUs for Palace MPI ranks, rounded up to the next
whole number. There is no fixed 15-process cap; pass `--mpi-procs` only when a
different value is required.

```bash
python generate_samples.py \
    --output-root training_run \
    --samples 12 \
    --mpi-procs 4 \
    --palace-bin "$PALACE_BIN"

python train_surrogate.py \
    --data-log training_run/training_data/active_learning_log.csv \
    --checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --epochs 2000 \
    --early-stopping-patience 200 \
    --validation-split 0.2 \
    --evaluation-output training_run/training_data/checkpoints/training_evaluation.json

python optimize.py \
    --output-root optimization_run \
    --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_run/training_data/active_learning_log.csv \
    --population 12 \
    --generations 10 \
    --mpi-procs 4 \
    --palace-bin "$PALACE_BIN"
```

The first command performs the expensive Palace simulations using a randomized
Latin-hypercube sample plan and creates the
CSV used by training. The second command trains and evaluates the surrogate.
Training restores the best validation checkpoint when early stopping triggers.
Pass `--keep-visualization` to preserve Paraview fields and diagnostic images;
they are removed by default after `terminal-C.csv` is read to limit disk use.
The final command runs prediction-only genetic optimization and measures its
selected design with Palace. Use at least 12 samples for a cold-start
population; 40-80 samples is a more useful first model.

### Files produced

The generator writes one consolidated CSV for surrogate training:

```text
training_run/training_data/active_learning_log.csv
```

Each generation also assigns samples to `train`, `validation`, and `test`
splits. The defaults are 70%, 10%, and 20%; validation and test counts are
rounded up to whole samples, with the remaining samples assigned to training.
The assignments and separate CSVs are written to:

```text
training_run/training_data/sample_splits.csv
training_run/training_data/train_samples.csv
training_run/training_data/validation_samples.csv
training_run/training_data/test_samples.csv
training_run/run_metadata.json
```

Override the percentages with `--training-percent`, `--validation-percent`,
and `--test-percent`; they must sum to 100. Sampling uses a fresh random seed
by default; pass `--seed` when an exactly reproducible sample plan is needed.
Pass `--include-boundary-points` to replace the first 16 samples with every
low/high corner of the four-parameter design box.

Its columns are `param_0`, `param_1`, `param_2`, `param_3`, `Ej_MHz`, and
`Ec_MHz`. The parameter columns correspond, in order, to
`Q1.pad_width` (um), `Q1.pad_height` (um), `Q1.pad_gap` (um), and `lj` (H).
The target columns are Palace-derived energies in MHz. Each Palace sample also
has a directory under `training_run/data/palace_runs/sample_XXXX/` containing
the generated mesh, Palace configuration, logs, and `terminal-C.csv`.

After training, the checkpoint directory contains:

```text
training_run/training_data/checkpoints/nemo_surrogate.mdlus
training_run/training_data/checkpoints/nemo_surrogate.state.pt
training_run/training_data/checkpoints/nemo_surrogate.evaluation.json
```

The optimization command writes `optimization_run/optimization_result.json`.
Generated run directories and model files are ignored by Git; copy or archive
them separately if they are needed on another machine.

To train or resume the model directly from an existing Palace log, run this
from `Opt/Quantum_Opt_Pipeline`:

```bash
python train_surrogate.py \
    --data-log training_data/active_learning_log.csv \
    --checkpoint training_data/checkpoints/nemo_surrogate.mdlus \
    --epochs 2000 \
    --early-stopping-patience 200 \
    --validation-split 0.2 \
    --accuracy-tolerance-percent 5 \
    --ej-threshold-mhz 22000 \
    --ec-threshold-mhz 400 \
    --evaluation-output training_data/checkpoints/training_evaluation.json
```

To score an untouched test split with the frozen checkpoint, add
`--test-log training_run/training_data/test_samples.csv`. The report then
includes `independent_test` metrics; the test CSV is never used for fitting.

Training prints the model status and validation MAE for `Ej` and `Ec`. A fresh
random seed is used for validation selection by default; pass
`--evaluation-seed` when reproducible evaluation splits are needed. The
checkpoint also stores an evaluation report beside the model as
`nemo_surrogate.evaluation.json`, including sample counts, train/validation
metrics, accuracy within the configured relative tolerance, precision, recall,
F1, split, seed, epoch count, and actual epochs run. Precision/recall/F1 classify each target as
above or below its configured threshold. If thresholds are omitted, the
training-partition median is used. Set `--validation-split 0` to disable the
holdout evaluation when all samples are needed for training diagnostics.

Five samples are the code minimum, but that is only enough to prove the loop runs. For four continuous design variables, use:

- `12` Palace samples: cold-start bootstrap for one population.
- `40-80` samples: useful first surrogate for a smooth local design region.
- `100-300` samples: recommended production range when the geometry response is nonlinear or the search bounds are broad.

Do not choose the final count by rule alone. Hold out 20% of measured samples and require validation error on both `Ej` and `Ec` to be below the accuracy needed by `compute_cost`. Continue Palace sampling when MC-dropout uncertainty remains above the active-learning threshold.

Each Palace run also writes:

- `data/palace_runs/gen_XX_ind_YY/gen_XX_ind_YY.msh`
- `data/palace_runs/gen_XX_ind_YY/gen_XX_ind_YY.json`
- `data/palace_runs/gen_XX_ind_YY/outputFiles/terminal-C.csv`
- `data/palace_runs/gen_XX_ind_YY/outputFiles/out.log`

A missing mesh, failed Palace process, missing capacitance output, or malformed result now stops the run instead of producing fabricated training data.
