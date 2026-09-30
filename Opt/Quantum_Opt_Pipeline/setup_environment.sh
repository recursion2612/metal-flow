#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
USE_CUDA=0
USE_DOCKER=0

# Parse options
for arg in "$@"; do
    case "$arg" in
        --cuda)
            USE_CUDA=1
            ;;
        --docker|--container)
            USE_DOCKER=1
            ;;
        --help|-h)
            printf "%s\n" "Usage: ./setup_environment.sh [--docker] [--cuda]"
            printf "%s\n" "  --docker    Build and configure containerized Docker environment"
            printf "%s\n" "  --cuda      Use CUDA-enabled dependencies instead of CPU wheel index"
            printf "%s\n" "  --help      Show this help message"
            exit 0
            ;;
        *)
            printf "%s\n" "Unknown option: $arg" >&2
            printf "%s\n" "Usage: ./setup_environment.sh [--docker] [--cuda]" >&2
            exit 2
            ;;
    esac
done

REQUIREMENTS_FILE="$SCRIPT_DIR/requirements-cpu.txt"
REQUIREMENTS_ARG="requirements-cpu.txt"
if [ "$USE_CUDA" -eq 1 ]; then
    REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
    REQUIREMENTS_ARG="requirements.txt"
fi

# -----------------------------------------------------------------------------
# Containerized Setup Mode (--docker / --container)
# -----------------------------------------------------------------------------
if [ "$USE_DOCKER" -eq 1 ]; then
    if ! command -v docker >/dev/null 2>&1; then
        printf "%s\n" "Error: Docker executable not found on PATH." >&2
        printf "%s\n" "Please install Docker or run without --docker for native virtualenv setup." >&2
        exit 1
    fi

    IMAGE_NAME=${DOCKER_IMAGE:-"quantum-opt-pipeline:latest"}
    printf "Building container image: %s\n" "$IMAGE_NAME"
    docker build \
        -t "$IMAGE_NAME" \
        --build-arg REQUIREMENTS="$REQUIREMENTS_ARG" \
        "$SCRIPT_DIR"

    printf "\nContainerized environment successfully built:\n"
    printf "  Image: %s\n" "$IMAGE_NAME"
    printf "  Run commands via: ./run_container.sh <command>\n"
    printf "  Example:          ./run_container.sh pytest -q\n"
    printf "  Interactive:      ./run_container.sh\n"
    exit 0
fi

# -----------------------------------------------------------------------------
# Native Virtualenv Setup Mode
# -----------------------------------------------------------------------------
# Portably resolve quantum_design_env directory without hardcoded local paths
if [ -n "${QUANTUM_DESIGN_ENV:-}" ] && [ -d "$QUANTUM_DESIGN_ENV" ]; then
    ENV_ROOT="$QUANTUM_DESIGN_ENV"
elif [ -d "$SCRIPT_DIR/../../quantum_design_env" ]; then
    ENV_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../quantum_design_env" && pwd)
elif [ -d "$SCRIPT_DIR/../quantum_design_env" ]; then
    ENV_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../quantum_design_env" && pwd)
elif [ -d "$SCRIPT_DIR/quantum_design_env" ]; then
    ENV_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/quantum_design_env" && pwd)
else
    ENV_ROOT="${QUANTUM_DESIGN_ENV:-"$SCRIPT_DIR/quantum_design_env"}"
    printf "quantum_design_env not found locally. Bootstrapping into %s...\n" "$ENV_ROOT"
    mkdir -p "$ENV_ROOT"
    if [ ! -d "$ENV_ROOT/quantum-metal" ]; then
        git clone --depth 1 https://github.com/qiskit-community/qiskit-metal.git "$ENV_ROOT/quantum-metal"
    fi
    if [ ! -d "$ENV_ROOT/SQDMetal" ]; then
        git clone --depth 1 https://github.com/sqdlab/SQDMetal.git "$ENV_ROOT/SQDMetal"
    fi
fi

VENV_PATH=${VENV_PATH:-"$ENV_ROOT/.venv"}
PYTHON_BIN=${PYTHON_BIN:-python3.11}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf "%s\n" "Python executable not found: $PYTHON_BIN" >&2
    printf "%s\n" "Set PYTHON_BIN to Python 3.11 or 3.12 and run again." >&2
    exit 1
fi

case "$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" in
    3.11|3.12) ;;
    *)
        printf "%s\n" "Python 3.11 or 3.12 is required." >&2
        exit 1
        ;;
esac

mkdir -p "$ENV_ROOT"
"$PYTHON_BIN" -m venv "$VENV_PATH"
VENV_PYTHON="$VENV_PATH/bin/python"

printf "Upgrading pip and installing core dependencies...\n"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"

printf "Installing lean CAD dependencies (excluding useless GUI/notebook/Ansys packages)...\n"
# Install quantum-metal with only [mesh] extra (excludes [gui], [ansys], [full])
"$VENV_PYTHON" -m pip install -e "$ENV_ROOT/quantum-metal[mesh]"

# Install SQDMetal with --no-deps to prevent pulling quantum-metal[full], PySide6, or Ansys
"$VENV_PYTHON" -m pip install -e "$ENV_ROOT/SQDMetal" --no-deps
"$VENV_PYTHON" -m pip install mph pyvista

# Install pipeline package in editable mode
"$VENV_PYTHON" -m pip install -e "$SCRIPT_DIR" --no-deps

printf "\nEnvironment ready:\n%s\n" "$VENV_PATH"
printf "Activate it with:\nsource \"%s/bin/activate\"\n" "$VENV_PATH"

printf "\nChecking external runtime tools:\n"
for tool in mpirun gmsh; do
    if command -v "$tool" >/dev/null 2>&1; then
        printf "  %s: found\n" "$tool"
    else
        printf "  %s: missing (required for Palace runs)\n" "$tool"
    fi
done
if [ -n "${PALACE_BIN:-}" ] && [ -x "$PALACE_BIN" ]; then
    printf "  Palace: found at %s\n" "$PALACE_BIN"
else
    printf "%s\n" "  Palace: set PALACE_BIN to the executable path (installed separately)"
fi
