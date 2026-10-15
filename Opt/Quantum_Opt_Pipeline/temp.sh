#!/usr/bin/env bash
TARGET_DIR="${1:-./optimization_run/data/palace_runs/gen_19_elite_00}"
PALACE_BIN="${PALACE_BIN:-palace}"
cd "$TARGET_DIR" || exit 1
"$PALACE_BIN" -np 18 -nt 1 gen_19_elite_00.json | tee outputFiles/out.log
