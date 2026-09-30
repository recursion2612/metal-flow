#!/usr/bin/bash

SRC_DIR=/home/aashay2/metal-flow/Opt/Quantum_Opt_Pipeline

rm  $SRC_DIR/training_run/run_metadata.json
rm -r $SRC_DIR/training_run/data/*

rm  $SRC_DIR/optimization_run/optimization_result.json
rm -r $SRC_DIR/optimization_run/data/*
rm $SRC_DIR/training_run/training_data/*.csv
rm $SRC_DIR/training_run/training_data/checkpoints/*
