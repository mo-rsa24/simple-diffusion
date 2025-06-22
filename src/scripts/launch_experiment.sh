#!/usr/bin/env bash

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment) EXP_ID="$2"; shift ;;
        --run) RUN_ID="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --use_wandb) USE_WANDB="--use_wandb" ;;
        --use_tensorboard) USE_TB="--use_tensorboard" ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# Check required args
if [[ -z "$EXP_ID" || -z "$RUN_ID" || -z "$DATASET" ]]; then
    echo "Missing required args: --experiment --run --dataset"
    exit 1
fi

# Activate conda
source ~/.bashrc
conda activate cxr

pwd=$(pwd)
echo "Running from: $pwd"
# Submit job
sbatch src/scripts/run_job.slurm "$EXP_ID" "$RUN_ID" "$DATASET" "$USE_WANDB" "$USE_TB"