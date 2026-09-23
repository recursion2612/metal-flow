#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CDAC_ROOT=${CDAC_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)}
ENV_ROOT=${ENV_ROOT:-"$CDAC_ROOT/quantum_design_env"}
VENV_PATH=${VENV_PATH:-"$ENV_ROOT/.venv"}
PYTHON_BIN=${PYTHON_BIN:-python3.11}
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements-cpu.txt"

if [ "${1:-}" = "--cuda" ]; then
    REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
    shift
fi

if [ "$#" -gt 0 ]; then
    printf '%s\n' "Usage: ./setup_environment.sh [--cuda]" >&2
    exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf '%s\n' "Python executable not found: $PYTHON_BIN" >&2
    printf '%s\n' "Set PYTHON_BIN to Python 3.11 or 3.12 and run again." >&2
    exit 1
fi

case "$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" in
    3.11|3.12) ;;
    *)
        printf '%s\n' "Python 3.11 or 3.12 is required." >&2
        exit 1
        ;;
esac

mkdir -p "$ENV_ROOT"
"$PYTHON_BIN" -m venv "$VENV_PATH"
VENV_PYTHON="$VENV_PATH/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"
"$VENV_PYTHON" -m pip install -e "$ENV_ROOT/quantum-metal[mesh]"
"$VENV_PYTHON" -m pip install -e "$ENV_ROOT/SQDMetal"
"$VENV_PYTHON" -m pip install -e "$SCRIPT_DIR" --no-deps

printf '\nEnvironment ready:\n%s\n' "$VENV_PATH"
printf 'Activate it with:\nsource "%s/bin/activate"\n' "$VENV_PATH"
printf '\nChecking external runtime tools:\n'
for tool in mpirun gmsh; do
    if command -v "$tool" >/dev/null 2>&1; then
        printf '  %s: found\n' "$tool"
    else
        printf '  %s: missing (required for Palace runs)\n' "$tool"
    fi
done
if [ -n "${PALACE_BIN:-}" ] && [ -x "$PALACE_BIN" ]; then
    printf '  Palace: found at %s\n' "$PALACE_BIN"
else
    printf '%s\n' '  Palace: set PALACE_BIN to the executable path (installed separately)'
fi
