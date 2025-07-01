#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL      # Notify on job completion and fail
#SBATCH --mail-user=iammrrobot24@gmail.com  # Set your email address
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=${BASE_DIR}/logs/slurm-%j.out
#SBATCH --error=${BASE_DIR}/logs/slurm-%j.err

# --- Receive Args ---
EXP_ID="${EXP_ID:-$1}"
RUN_ID="${RUN_ID:-$2}"
DATASET="${DATASET:-$3}"
TASK="${TASK:-$4}"
COLOR="${COLOR:-$5}"
NUMBER="${NUMBER:-$6}"
GEN_MODEL="${GEN_MODEL:-$7}"
USE_WANDB="${USE_WANDB:-$8}"
USE_TB="${USE_TB:-$9}"
RESUME="${RESUME:-}"
BASE_DIR="${BASE_DIR:?}"

# --- Activate environment ---
source ~/.bashrc
mamba activate cxr

# (Inside run_job.slurm)
echo "Changing directory to project root..."
cd /home-mscluster/mmolefe/Playground/PhD/simple-diffusion || { echo "Failed to cd to project root!"; exit 1; }
echo "Working directory: $(pwd)"
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"

# Determine checkpoint path and resume flag
CKPT_DIR="$BASE_DIR/checkpoints"
LOG_DIR="$BASE_DIR/logs"
LATEST_CKPT=$(ls -1 "$CKPT_DIR"/epoch_*.ckpt 2>/dev/null | sort | tail -n 1)
if [[ -n "$RESUME" || -n "$LATEST_CKPT" ]]; then
    RESUME_FLAG="--resume"
else
    RESUME_FLAG=""
fi

# --- Build Python Command ---
PY_CMD="python3 src/run.py \
  --experiment_id \"$EXP_ID\" \
  --run_id \"$RUN_ID\" \
  --dataset \"$DATASET\""

if [[ -n "$TASK" && "$TASK" != "None" ]]; then
    PY_CMD+=" --task \"$TASK\""
fi
if [[ -n "$COLOR" && "$COLOR" != "None" ]]; then
    PY_CMD+=" --color \"$COLOR\""
fi
if [[ -n "$NUMBER" && "$NUMBER" != "None" ]]; then
    PY_CMD+=" --number \"$NUMBER\""
fi
if [[ -n "$GEN_MODEL" && "$GEN_MODEL" != "None" ]]; then
    PY_CMD+=" --gen_model \"$GEN_MODEL\""
fi
if [[ "$USE_WANDB" == "true" ]]; then
    PY_CMD+=" --use_wandb"
fi
if [[ "$USE_TB" == "true" ]]; then
    PY_CMD+=" --use_tensorboard"
fi
PY_CMD+=" --checkpoint-dir $CKPT_DIR --log-dir $LOG_DIR"
if [[ -n "$RESUME_FLAG" ]]; then
    PY_CMD+=" $RESUME_FLAG"
fi
if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    PY_CMD+=" --array_index $SLURM_ARRAY_TASK_ID"
fi

echo "Launching: $PY_CMD"
eval $PY_CMD
