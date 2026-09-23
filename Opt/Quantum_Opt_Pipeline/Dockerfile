# Multi-platform containerized environment for CDAC Quantum Optimization Pipeline
# Designed for headless, automated execution with minimal dependency footprint.
# Excludes bloated dependencies (PySide6/Qt, Jupyter, Ansys, Streamlit, torchvision).

FROM python:3.11-slim-bookworm

LABEL maintainer="CDAC Quantum Optimization Team"
LABEL description="PhysicsNeMo surrogate and Qiskit Metal + Palace optimization environment"

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV QUANTUM_DESIGN_ENV=/opt/quantum_design_env
ENV PALACE_BIN=/usr/local/bin/palace
ENV PATH="/usr/local/bin:${PATH}"

# Install core system packages for OpenMPI, Gmsh, OpenGL headless meshing, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    gmsh \
    libgl1 \
    libglu1-mesa \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Setup lean quantum_design_env containing Qiskit Metal (quantum-metal) and SQDMetal
# Cloned directly from upstream repositories into /opt/quantum_design_env (no local paths)
WORKDIR /opt/quantum_design_env

RUN git clone --depth 1 https://github.com/qiskit-community/qiskit-metal.git /opt/quantum_design_env/quantum-metal && \
    git clone --depth 1 https://github.com/sqdlab/SQDMetal.git /opt/quantum_design_env/SQDMetal

# Install quantum-metal with only the [mesh] extra (Gmsh meshing, excludes [gui] and [ansys])
RUN pip install --no-cache-dir -e "/opt/quantum_design_env/quantum-metal[mesh]"

# Install SQDMetal with --no-deps to avoid pulling quantum-metal[full], PySide6, and Ansys
# Then install only the runtime packages required by SQDMetal Palace simulation (mph, pyvista)
RUN pip install --no-cache-dir --no-deps -e /opt/quantum_design_env/SQDMetal && \
    pip install --no-cache-dir mph pyvista

# Setup workspace and install pipeline dependencies
WORKDIR /workspace

# Copy requirements and pyproject.toml first for layer caching
COPY requirements-cpu.txt requirements.txt pyproject.toml ./
ARG REQUIREMENTS=requirements-cpu.txt
RUN pip install --no-cache-dir -r ${REQUIREMENTS} && \
    pip install --no-cache-dir "pytest>=8"

# Copy the rest of the repository
COPY . /workspace

# Install the pipeline package in editable mode without reinstalling dependencies
RUN pip install --no-cache-dir --no-deps -e .

# Default working directory
WORKDIR /workspace

# Verify environment imports on container build
RUN python -c "import torch, physicsnemo, qiskit_metal, SQDMetal, gmsh; print('Container runtime imports verified successfully')"

# Default entry point runs unit tests; pass custom command to override
CMD ["pytest", "-q"]
