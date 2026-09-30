#!/usr/bin/bash


python generate_samples.py --output-root training_run --samples 500 --palace-bin "$PALACE_BIN" --mpi-procs 18

python train_surrogate.py --data-log training_run/training_data/train_samples.csv --test-log training_run/training_data/test_samples.csv  --checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus --epochs 2000 --early-stopping-patience 200 --evaluation-output training_run/training_data/checkpoints/training_evaluation.json

python optimize.py --output-root optimization_run --model-checkpoint training_run/training_data/checkpoints/nemo_surrogate.mdlus --model-data-log training_run/training_data/train_samples.csv --population 100 --generations 20  --palace-bin "$PALACE_BIN" --mpi-procs 18
