#!/usr/bin/bash

SRC_DIR=/home/aashay2/metal-flow/Opt/Quantum_Opt_Pipeline

echo "Storing Sim results in results/"


mkdir -p $SRC_DIR/results

cp -r $SRC_DIR/training_run/ $SRC_DIR/results/
cp -r $SRC_DIR/optimization_run/ $SRC_DIR/results/
zip -r results.zip $SRC_DIR/results/
