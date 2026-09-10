# Nemo + Palace Genetic Optimization

This pipeline uses Qiskit Metal geometry, SQDMetal's AWS Palace capacitance backend, an NVIDIA PhysicsNeMo `Module` surrogate, and a real-valued genetic algorithm.

## Install PhysicsNeMo

The package is installed as `nvidia-physicsnemo` and imported as `physicsnemo`:

```bash
python -m pip install -r Opt/Quantum_Opt_Pipeline/requirements.txt
python -c "import physicsnemo; print(physicsnemo.__version__)"
```

For NVIDIA GPU training, use a CUDA-compatible PyTorch environment or the NVIDIA PhysicsNeMo container. macOS can use the model on CPU if the package dependencies install successfully, but it cannot use CUDA.

Install the local CAD packages from the repository root:

```bash
python -m pip install -e "quantum_design_env/quantum-metal[mesh]"
python -m pip install -e quantum_design_env/SQDMetal
```

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
3. Run Palace for uncertain candidates, append measured `Ej` and `Ec` to `training_data/active_learning_log.csv`, and train Nemo after each generation.
4. Select three aspirants per tournament and retain the lowest-cost candidate. Two elites survive each generation.
5. Apply SBX crossover and polynomial mutation. The default mutation probability is `1 / 4 = 0.25` per parameter, a standard starting point for four continuous genes; every child is clipped to its bounds.
6. Save the PhysicsNeMo model to `training_data/checkpoints/nemo_surrogate.mdlus` and optimizer/normalization state to `nemo_surrogate.state.pt`.
7. Select the final winner from measured Palace samples, not only a surrogate prediction.

Example invocation:

```python
from main import run_pipeline

result = run_pipeline(design=design, output_root="run_01", palace_bin="/path/to/palace")
```

Complete default run:

```bash
cd Opt/Quantum_Opt_Pipeline
python run_optimization.py --output-root run_01 --palace-bin /path/to/palace
```

## Results

The run returns a Python dictionary and writes `run_01/optimization_result.json` containing the best parameters, measured `Ej`/`Ec`, cost, sample count, mutation rate, training CSV path, and PhysicsNeMo checkpoint path.

## Training the surrogate

Training is active learning, not a standalone synthetic-data pretraining step. Each Palace result is appended to `training_data/active_learning_log.csv`; after each generation, `fit()` trains the PhysicsNeMo module for 150 epochs and saves the checkpoint. A new process reloads both the measured CSV and the `.mdlus` model.

To train or resume the model directly from an existing Palace log:

```bash
cd Opt/Quantum_Opt_Pipeline
/Users/akhshatkampassi/Documents/CDAC/quantum_design_env/.venv/bin/python train_nemo_surrogate.py \
    --data-log training_data/active_learning_log.csv \
    --checkpoint training_data/checkpoints/nemo_surrogate.mdlus \
    --epochs 150
```

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
