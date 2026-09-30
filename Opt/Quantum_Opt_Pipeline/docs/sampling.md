# Sample Generation & Design Exploration

The `generate_samples.py` script automates parametric exploration of the transmon qubit design space, generating geometries, meshing them in Gmsh, running AWS Palace simulations, and writing partitioned training datasets.

---

## 4-Parameter Design Space

The pipeline explores four continuous geometric and circuit variables:

| Index | Parameter Key | Description | Range | Unit |
| :---: | :--- | :--- | :---: | :---: |
| 0 | `Q1.pad_width` | Transmon capacitor pad width | 300.0 to 600.0 | µm |
| 1 | `Q1.pad_height` | Transmon capacitor pad height | 20.0 to 80.0 | µm |
| 2 | `Q1.pad_gap` | Gap between pad and ground pocket | 10.0 to 40.0 | µm |
| 3 | `lj` | Josephson junction linear inductance | 6.0 to 14.0 | nH |

Target Hamiltonian energies derived from simulation:
- **$E_j$ (Josephson Energy)**: Calculated in MHz from Josephson inductance $L_j$.
- **$E_c$ (Charging Energy)**: Calculated in MHz from Maxwell self-capacitance $C_\Sigma$.

---

## Sampling Techniques

### 1. Randomized Latin Hypercube Sampling (LHS)
To ensure optimal space-filling properties across the 4-dimensional hypercube, samples are selected using Latin Hypercube Sampling. Each dimension is divided into $N$ equal-probability strata, ensuring uniform marginal coverage without clustering.

```bash
python generate_samples.py --output-root run_01 --samples 150
```

### 2. Randomized Entropy Seeds
By default, each invocation draws fresh entropy from NumPy's `SeedSequence` to produce distinct sample plans. For deterministic reproduction, pass an integer seed:

```bash
python generate_samples.py --output-root run_01 --samples 150 --seed 42
```

### 3. Boundary Corner Sampling
To prevent surrogate extrapolation errors when the Genetic Algorithm searches parameter boundaries, pass `--include-boundary-points`. This replaces the first 16 samples ($2^4$) with every permutation of minimum and maximum parameter bounds:

```bash
python generate_samples.py --output-root run_01 --samples 150 --include-boundary-points
```

---

## Automated 70/10/20 Partitioning

The generator automatically divides collected samples into three distinct CSV datasets:
- **Training Set (70%)**: Used for fitting surrogate model weights.
- **Validation Set (10%)**: Used for monitoring loss and triggering early stopping.
- **Test Set (20%)**: Held out untouched for final independent model evaluation.

Holdout sample counts are rounded up using `ceil`, and remaining samples are assigned to training. For 150 samples, this produces exactly:
- `105` training samples
- `15` validation samples
- `30` test samples

Custom percentage overrides:
```bash
python generate_samples.py \
    --output-root run_01 \
    --samples 150 \
    --training-percent 70 \
    --validation-percent 10 \
    --test-percent 20
```

---

## Output Files

The generator writes structured data under the designated output directory:

```text
<output-root>/
├── run_metadata.json                          # Audit log (timestamp, git SHA, seed, python version)
├── training_data/
│   ├── active_learning_log.csv                # Complete consolidated dataset
│   ├── sample_splits.csv                      # Manifest recording split assignment per sample
│   ├── train_samples.csv                      # 70% training partition
│   ├── validation_samples.csv                 # 10% validation partition
│   └── test_samples.csv                       # 20% independent test partition
└── data/
    └── palace_runs/
        └── sample_0000/
            ├── sample_0000.msh                # Gmsh 3D conformal mesh
            ├── sample_0000.json               # Palace configuration
            ├── palace.stdout.log              # Solver stdout log
            └── outputFiles/
                ├── terminal-C.csv             # Extracted Maxwell capacitance matrix
                └── out.log                    # Iterative PCG convergence log
```

Transient 3D Paraview field data (`.vtu`) is automatically pruned after capacitance extraction to conserve disk space. To retain visualization fields, pass `--keep-visualization`.
