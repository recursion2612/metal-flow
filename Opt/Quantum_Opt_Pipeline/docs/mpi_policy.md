# Resource Management & MPI Allocation Policy

AWS Palace solves large-scale 3D finite-element electromagnetic systems in parallel using the Message Passing Interface (MPI). To prevent compute resource exhaustion and maintain host operating system responsiveness, the pipeline enforces an automated machine-aware core reservation policy.

---

## 10% Core Reservation Rule

The pipeline reserves **at least 10% of total logical CPU cores** for host OS background tasks and uses the remaining cores for Palace MPI execution:

$$\\text{MPI Ranks} = \\max\\left(1, \\left\\lceil 0.90 \\times \\text{logical\\_cpu\\_count} \\right\\rceil\\right)$$

There is no arbitrary hard cap (e.g. 15 cores); the limit scales dynamically with any machine:

| Machine Logical Cores | Palace MPI Processes | Reserved OS Cores |
| :---: | :---: | :---: |
| 4 cores | 4 | 0 |
| 8 cores | 8 | 0 |
| 10 cores | 9 | 1 |
| 11 cores (e.g., Apple M-series) | 10 | 1 |
| 16 cores | 15 | 1 |
| 32 cores | 29 | 3 |
| 64 cores | 58 | 6 |

---

## Python API

You can inspect the calculated limit directly via the `src.palace` module:

```python
from src.palace import max_mpi_procs, default_mpi_procs

# Safe default for the current machine
print(f"Default MPI ranks: {default_mpi_procs()}")

# Evaluate safe ranks for a specific CPU count
print(f"Safe limit for 11 cores: {max_mpi_procs(11)}")  # Output: 10
```

---

## Command-Line Overrides

By default, omitting `--mpi-procs` automatically uses the machine default:

```bash
# Automatically utilizes safe 90% core allocation
python generate_samples.py --output-root run_01 --samples 50
```

If you explicitly provide `--mpi-procs`, the argument is validated against the machine limit:

```bash
# Valid override
python generate_samples.py --output-root run_01 --samples 50 --mpi-procs 8

# Invalid: Attempting to use more than 90% of cores raises an error
python generate_samples.py --output-root run_01 --samples 50 --mpi-procs 11
# ValueError: Palace MPI processes must be between 1 and 10 on this machine
```

---

## Execution Model

- **Intra-Sample Parallelism**: MPI parallelizes the matrix assembly, preconditioning, and PCG linear solve within each Palace simulation.
- **Sequential Sample Evaluation**: The outer parameter loop in `generate_samples.py` evaluates designs sequentially to ensure deterministic log generation and avoid resource contention.
