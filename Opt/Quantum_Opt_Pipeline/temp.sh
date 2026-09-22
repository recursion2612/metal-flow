#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PALACE_BIN="${PALACE_BIN:-palace}"
OUTPUT_DIR="${OUTPUT_DIR:-${TMPDIR:-/tmp}/cdac-palace/smoke}"

mkdir -p "${OUTPUT_DIR}/outputFiles"
cd "${SCRIPT_DIR}"
"${PALACE_BIN}" -np "${MPI_PROCS:-1}" -nt "${PALACE_THREADS:-1}" smoke.json \
	| tee "${OUTPUT_DIR}/outputFiles/out.log"
