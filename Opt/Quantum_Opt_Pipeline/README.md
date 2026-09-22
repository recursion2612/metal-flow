# Nemo + Palace Genetic Optimization

This pipeline uses Qiskit Metal geometry, SQDMetal's AWS Palace capacitance backend, an NVIDIA PhysicsNeMo `Module` surrogate, and a real-valued genetic algorithm.

## Requirements and installation

Use Python 3.11. The training-only step needs Python, PyTorch, NumPy, pandas,
and PhysicsNeMo. The Palace data-generation and optimization steps additionally
need MPI, Gmsh, Palace, Qiskit Metal, and SQDMetal.

From the repository root, configure a virtual environment and install the
local projects. `REPO_ROOT` is derived from the checkout, so no user-specific
absolute path is required:

```bash
REPO_ROOT="$(pwd)"
python3.11 -m venv "$REPO_ROOT/.venv"
source .venv/bin/activate
python -m pip install --upgrade pip

# Select CUDA-capable PyTorch when an NVIDIA GPU is available; otherwise use CPU wheels.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    python -m pip install -r "$REPO_ROOT/Opt/Quantum_Opt_Pipeline/requirements.txt"
else
    python -m pip install -r "$REPO_ROOT/Opt/Quantum_Opt_Pipeline/requirements-cpu.txt"
fi
python -m pip install -e "$REPO_ROOT/quantum_design_env/quantum-metal[mesh]"
python -m pip install -e "$REPO_ROOT/quantum_design_env/SQDMetal"
python -m pip install -e "$REPO_ROOT/Opt/Quantum_Opt_Pipeline" --no-deps
```

Then enter the pipeline directory for all commands below:

```bash
cd Opt/Quantum_Opt_Pipeline
```

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
from src.cad_adapter import export_qiskit_metal_gmsh
export_qiskit_metal_gmsh(design, "design.msh")
```

The generated `.msh` must contain populated `$Nodes` and `$Elements` sections.
For the local `DesignPlanar` implementation, use the SQDMetal runner below; it owns the Palace physical-group mapping.

## Training and optimization flow

1. Generate an initial population uniformly inside the four fabrication bounds.
2. Use Monte Carlo dropout predictions when the surrogate is trained and uncertainty is below `1.5`.
3. In active-learning mode, run Palace for uncertain candidates, append measured `Ej` and `Ec` to `training_data/active_learning_log.csv`, and train Nemo after each generation.
4. In prediction-only mode, use the previously trained model for scoring and run Palace only for the final selected design. Select three aspirants per tournament and retain the lowest-cost candidate. Two elites survive each generation.
5. Apply SBX crossover and polynomial mutation. The default mutation probability is `1 / 4 = 0.25` per parameter, a standard starting point for four continuous genes; every child is clipped to its bounds.
6. Save the PhysicsNeMo model to `training_data/checkpoints/nemo_surrogate.mdlus` and optimizer/normalization state to `nemo_surrogate.state.pt`.
7. Select the final winner from measured Palace samples, not only a surrogate prediction.

Example invocation:

```python
from main import run_pipeline

result = run_pipeline(design=design, output_root="run_01", palace_bin="/absolute/path/to/palace")
```

Complete default run:

```bash
python run_optimization.py \
    --output-root run_01 \
    --model-checkpoint training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_data/active_learning_log.csv \
    --palace-bin "$PALACE_BIN"
```

The normal optimization workflow uses Nemo predictions to score every GA
population. Train and validate the surrogate separately with
`train_nemo_surrogate.py`, then provide its checkpoint and matching data log
above. The optimizer loads that final model, does not call `fit()`, and sends
only the final Nemo-selected design to Palace for ground-truth simulation.
The result manifest contains both predicted and Palace-measured metrics.

To deliberately run the older active-learning behavior, opt in explicitly:

```bash
python run_optimization.py \
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
to `training_data/active_learning_log.csv`; `train_nemo_surrogate.py` validates
the measured data, trains the PhysicsNeMo module, and saves the final
checkpoint. The normal optimizer then reloads both the measured CSV and the
`.mdlus` model without retraining.

### Complete training run from new Palace samples

Run all commands from `Opt/Quantum_Opt_Pipeline`. Palace supports at most 15
MPI processes, so keep `--mpi-procs` at 15 or below:

```bash
python generate_training_data.py \
    --output-root training_run \
    --samples 12 \
    --seed 42 \
    --mpi-procs 4 \
    --palace-bin "$PALACE_BIN"

python train_nemo_surrogate.py \
    --data-log training_run/training_data/active_learning_log.csv \
    --checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --epochs 150 \
    --validation-split 0.2 \
    --evaluation-output training_run/training_data/checkpoints/training_evaluation.json

python run_optimization.py \
    --output-root optimization_run \
    --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_run/training_data/active_learning_log.csv \
    --population 12 \
    --generations 10 \
    --mpi-procs 4 \
    --palace-bin "$PALACE_BIN"
```

The first command performs the expensive Palace simulations and creates the
CSV used by training. The second command trains and evaluates the surrogate.
The final command runs prediction-only genetic optimization and measures its
selected design with Palace. Use at least 12 samples for a cold-start
population; 40-80 samples is a more useful first model.

### Files produced

The generator writes one consolidated CSV for surrogate training:

```text
training_run/training_data/active_learning_log.csv
```

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
python train_nemo_surrogate.py \
    --data-log training_data/active_learning_log.csv \
    --checkpoint training_data/checkpoints/nemo_surrogate.mdlus \
    --epochs 150 \
    --validation-split 0.2 \
    --evaluation-seed 42 \
    --accuracy-tolerance-percent 5 \
    --ej-threshold-mhz 22000 \
    --ec-threshold-mhz 400 \
    --evaluation-output training_data/checkpoints/training_evaluation.json
```

Training prints the model status and validation MAE for `Ej` and `Ec`. The
checkpoint also stores an evaluation report beside the model as
`nemo_surrogate.evaluation.json`, including sample counts, train/validation
metrics, accuracy within the configured relative tolerance, precision, recall,
F1, split, seed, and epoch count. Precision/recall/F1 classify each target as
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
