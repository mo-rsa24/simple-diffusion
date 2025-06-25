#!/usr/bin/env bash

# Print each command (optional for debugging)
# set -x

# Initialize optional flags
USE_WANDB=""
USE_TB=""
TASK=""
COLOR=""
NUMBER=""

# Parse CLI arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment|-e) EXP_ID="$2"; shift ;;
        --run|-r) RUN_ID="$2"; shift ;;
        --dataset|-d) DATASET="$2"; shift ;;
        --task|-t) TASK="$2"; shift ;;
        --color|-c) COLOR="$2"; shift ;;
        --number|-n) NUMBER="$2"; shift ;;
        --use_wandb) USE_WANDB="true" ;;
        --use_tensorboard) USE_TB="true" ;;
        *) echo "❌ Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$EXP_ID" || -z "$RUN_ID" || -z "$DATASET" ]]; then
    echo "❌ Missing required arguments. Usage:"
    echo "   ./run_experiment.sh --experiment <EXP_ID> --run <RUN_ID> --dataset <DATASET> [--task <TASK>] [--color <COLOR>] [--number <NUM>] [--use_wandb] [--use_tensorboard]"
    exit 1
fi

# Optional: Move to project root if script is nested
cd "$(dirname "$0")/../.." || exit 1

# Set PYTHONPATH to ensure src is accessible
export PYTHONPATH=$(pwd)

# Construct Python command
PY_CMD="python3 -m debugpy --listen 5678 --wait-for-client src/run.py \
  --experiment_id \"$EXP_ID\" \
  --run_id \"$RUN_ID\" \
  --dataset \"$DATASET\""

if [[ -n "$TASK" ]]; then
    PY_CMD+=" --task \"$TASK\""
fi
if [[ -n "$COLOR" ]]; then
    PY_CMD+=" --color \"$COLOR\""
fi
if [[ -n "$NUMBER" ]]; then
    PY_CMD+=" --number \"$NUMBER\""
fi
if [[ "$USE_WANDB" == "true" ]]; then
    PY_CMD+=" --use_wandb"
fi
if [[ "$USE_TB" == "true" ]]; then
    PY_CMD+=" --use_tensorboard"
fi

echo "🚀 Launching experiment with debugpy..."
eval $PY_CMD

# For non-debugpy runs, you could use:
# python3 src/run.py ...
