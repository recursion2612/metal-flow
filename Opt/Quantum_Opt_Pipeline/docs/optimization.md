# Genetic Algorithm Optimization

The `optimize.py` script executes a real-valued Genetic Algorithm (GA) to discover transmon qubit geometry and circuit parameters that match target Hamiltonian energy levels ($E_j^{\text{target}}, E_c^{\text{target}}$).

---

## Objective Function

Candidates are scored using a normalized quadratic relative-error cost function:

$$
\text{Cost}(x) = \left(\frac{\hat{E}_j(x) - E_j^{\text{target}}}{E_j^{\text{target}}}\right)^2 + \left(\frac{\hat{E}_c(x) - E_c^{\text{target}}}{E_c^{\text{target}}}\right)^2
$$

Where $\hat{E}_j$ and $\hat{E}_c$ are predicted in real-time by the frozen PhysicsNeMo surrogate. The global optimum is $\text{Cost}=0$.

---

## Genetic Operators

The optimization engine uses real-valued continuous genetic operators:

1. **Tournament Selection**: Chooses the fittest individual among 3 randomly selected candidates.
2. **Elitism**: Clones the top 2 fittest candidates directly into the next generation without modification.
3. **Simulated Binary Crossover (SBX)**: Recombines parent parameters continuously with distribution index $\eta_c=2.0$.
4. **Polynomial Mutation**: Applies continuous perturbation with distribution index $\eta_m=20.0$ and mutation probability $p_m = 1/4 = 0.25$ per gene. Children are strictly clipped to fabrication bounds.

---

## Execution Modes

### Mode 1: Fast Frozen-Surrogate Optimization (Default)

Evaluates all GA candidates strictly using the neural surrogate:

```bash
python optimize.py \
    --output-root optimization_run \
    --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_run/training_data/train_samples.csv \
    --population 12 \
    --generations 10 \
    --palace-bin "$PALACE_BIN"
```

### Mode 2: Live Palace Validation (`--use-palace`)

Runs the full optimization with the surrogate, then automatically sends the final winning candidate to AWS Palace for full-wave finite-element verification:

```bash
python optimize.py \
    --output-root optimization_run \
    --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus \
    --model-data-log training_run/training_data/train_samples.csv \
    --population 12 \
    --generations 10 \
    --use-palace \
    --palace-bin "$PALACE_BIN"
```

---

## Output Manifest

Results are written to `<output-root>/optimization_result.json`:

```json
{
  "best_cost": 0.000341,
  "best_params": [
    485.23,
    42.10,
    24.85,
    9.15e-9
  ],
  "param_names": [
    "Q1.pad_width",
    "Q1.pad_height",
    "Q1.pad_gap",
    "lj"
  ],
  "predicted_Ej_MHz": 18240.5,
  "predicted_Ec_MHz": 248.3,
  "measured_Ej_MHz": 18235.1,
  "measured_Ec_MHz": 248.1,
  "result_source": "palace_simulation",
  "population_size": 12,
  "generations": 10
}
```
