# Environment Setup & Dependencies

The pipeline supports both **Containerized (Docker)** and **Native Virtual Environment** execution with zero hardcoded machine paths.

---

## Prerequisites

- **Python**: `>=3.11,<3.13` (strictly Python 3.11 or 3.12 required for `quantum-metal` and `SQDMetal`).
- **External Binaries**: OpenMPI (`mpirun`), Gmsh (`gmsh`), and AWS Palace (`palace`).

---

## Option A: Containerized Execution (Docker - Recommended)

Docker provides an isolated, reproducible environment pre-configured with Linux OpenMPI, Gmsh, and all required Python packages.

### 1. Build Container Image

Run from the repository root:

```bash
./setup_env.sh --docker
```

This compiles a lean image named `quantum-opt-pipeline:latest` based on Debian Bookworm.

### 2. Run Commands

Use `run_container.sh` to execute commands inside the container while mounting your local workspace:

```bash
# Execute unit test suite
./run_container.sh pytest -q

# Run sample generation
./run_container.sh python generate_samples.py --output-root run_01 --samples 50

# Launch interactive container bash shell
./run_container.sh
```

### 3. Forwarding External Palace

If you have a host-compiled Palace binary, forward it into the container via `PALACE_BIN`:

```bash
export PALACE_BIN="/path/to/palace"
./run_container.sh python generate_samples.py --palace-bin "$PALACE_BIN" ...
```

---

## Option B: Native Virtual Environment

### 1. Run Setup Script

```bash
./setup_env.sh
```

For NVIDIA GPU acceleration:

```bash
./setup_env.sh --cuda
```

### 2. Activate Environment

```bash
source "${QUANTUM_DESIGN_ENV:-../../quantum_design_env}/.venv/bin/activate" 2>/dev/null || \
    source "./quantum_design_env/.venv/bin/activate"
```

### 3. Verify Installation

```bash
python -c "import physicsnemo, qiskit_metal, SQDMetal, gmsh; print('Runtime packages imported successfully')"
python -c "import torch; print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
```

---

## Dependency Optimization (Lean Profile)

To prevent disk bloat, useless packages commonly bundled in broad quantum development environments are strictly excluded from installation:

### Excluded Packages (~2.5 GB+ Saved)
- **GUI Libraries**: `PySide6`, `QtPy`, `QDarkStyle`, `shiboken6` (the pipeline executes headless Qiskit Metal `DesignPlanar`).
- **Jupyter Stack**: `IPython`, `ipykernel`, `ipywidgets`, `jupyter`, `jupyterlab`, `nbconvert`, `nbsphinx`.
- **Proprietary CAD**: `pyaedt`, `pyedb`, `pyEPR-quantum` (Ansys HFSS/Q3D backends; we use AWS Palace).
- **Web Servers**: `streamlit`, `uvicorn`, `starlette`, `fastapi`, `websockets`.
- **Unused AI & Plotting**: `torchvision`, `timm`, `xarray`, `zarr`, `datashader`, `schemdraw`, `scikit-rf`.

### Retained Core Runtime
- `numpy>=1.24`, `pandas>=2.0`, `torch>=2.1`, `nvidia-physicsnemo>=2.2,<3`
- `quantum-metal[mesh]>=0.7.4` (provides Qiskit Metal geometry and Gmsh mesher)
- `SQDMetal` (minimal runtime: `mph`, `pyvista`)
- `pytest>=8`

---

## External Solver Setup

Palace can be compiled from source using the [AWS Palace Installation Guide](https://awslabs.github.io/palace/dev/install/) or installed via Spack:

```bash
spack install palace
export PALACE_BIN="$(spack location -i palace)/bin/palace"
export PATH="$(dirname "$PALACE_BIN"):$PATH"

# Verify tool availability
which mpirun
mpirun --version
gmsh --version
"$PALACE_BIN" --help >/dev/null
```
