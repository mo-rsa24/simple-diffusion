#!/usr/bin/env bash

# Print each command (optional for debugging)
# set -x

# Initialize optional flags
USE_WANDB=""
USE_TB=""

# Parse CLI arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment) EXP_ID="$2"; shift ;;
        --run) RUN_ID="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --use_wandb) USE_WANDB="true" ;;
        --use_tensorboard) USE_TB="true" ;;
        *) echo "❌ Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$EXP_ID" || -z "$RUN_ID" || -z "$DATASET" ]]; then
    echo "❌ Missing required arguments. Usage:"
    echo "   ./run_experiment.sh --experiment <EXP_ID> --run <RUN_ID> --dataset <DATASET> [--use_wandb] [--use_tensorboard]"
    exit 1
fi

# Optional: Move to project root if script is nested
cd "$(dirname "$0")/../.." || exit 1

# Set PYTHONPATH to ensure src is accessible
export PYTHONPATH=$(pwd)

# Launch with debugpy
echo "🚀 Launching experiment with debugpy..."
python3 -m debugpy --listen 5678 --wait-for-client src/run.py \
  --experiment_id "$EXP_ID" \
  --run_id "$RUN_ID" \
  --dataset "$DATASET" \
  ${USE_WANDB:+--use_wandb} \
  ${USE_TB:+--use_tensorboard}

# python3 src/run.py \
#   --experiment_id "$EXP_ID" \
#   --run_id "$RUN_ID" \
#   --dataset "$DATASET" \
#   ${USE_WANDB:+--use_wandb} \
#   ${USE_TB:+--use_tensorboard}
