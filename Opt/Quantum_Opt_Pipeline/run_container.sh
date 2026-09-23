#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IMAGE_NAME=${DOCKER_IMAGE:-"quantum-opt-pipeline:latest"}

if ! command -v docker >/dev/null 2>&1; then
    printf "%s\n" "Error: Docker executable not found on PATH." >&2
    exit 1
fi

# Build volume and environment arguments
DOCKER_ARGS="-v \"$SCRIPT_DIR:/workspace\" -w /workspace"

# Mount external Palace executable if provided and exists
if [ -n "${PALACE_BIN:-}" ] && [ -x "$PALACE_BIN" ]; then
    if [ "$(uname -s)" = "Darwin" ] && file "$PALACE_BIN" 2>/dev/null | grep -q "Mach-O"; then
        printf "Notice: Host PALACE_BIN (%s) is a macOS binary and cannot run inside Linux container.\n" "$PALACE_BIN" >&2
        printf "Inside the container, ensure a Linux Palace binary is mounted or available on PATH.\n" >&2
    else
        DOCKER_ARGS="$DOCKER_ARGS -v \"$PALACE_BIN:/usr/local/bin/palace:ro\" -e PALACE_BIN=/usr/local/bin/palace"
    fi
elif [ -n "${PALACE_BIN:-}" ]; then
    DOCKER_ARGS="$DOCKER_ARGS -e PALACE_BIN=$PALACE_BIN"
fi

# Forward MPI procs setting if present
if [ -n "${MPI_PROCS:-}" ]; then
    DOCKER_ARGS="$DOCKER_ARGS -e MPI_PROCS=$MPI_PROCS"
fi

# If interactive terminal is available, use -it
INTERACTIVE_FLAG=""
if [ -t 0 ] && [ -t 1 ]; then
    INTERACTIVE_FLAG="-it"
fi

if [ "$#" -eq 0 ]; then
    eval "docker run --rm $INTERACTIVE_FLAG $DOCKER_ARGS \"$IMAGE_NAME\" /bin/bash"
else
    eval "docker run --rm $INTERACTIVE_FLAG $DOCKER_ARGS \"$IMAGE_NAME\" \"\$@\""
fi
