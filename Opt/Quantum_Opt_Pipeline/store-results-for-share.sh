#!/usr/bin/env bash

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Storing Sim results in results/"


mkdir -p $SRC_DIR/results

cp -r $SRC_DIR/training_run/ $SRC_DIR/results/
cp -r $SRC_DIR/optimization_run/ $SRC_DIR/results/
zip -r results.zip $SRC_DIR/results/
