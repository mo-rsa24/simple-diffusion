#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL      # Notify on job completion and fail
#SBATCH --mail-user=iammrrobot24@gmail.com  # Set your email address
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# --- Receive Args ---
EXP_ID="$1"
RUN_ID="$2"
DATASET="$3"
TASK="$4"
COLOR="$5"
NUMBER="$6"
GEN_MODEL="$7"
USE_WANDB="$8"
USE_TB="$9"

# --- Activate environment ---
source ~/.bashrc
mamba activate cxr

# (Inside run_job.slurm)
echo "Changing directory to project root..."
cd /home-mscluster/mmolefe/Playground/PhD/simple-diffusion || { echo "Failed to cd to project root!"; exit 1; }
echo "Working directory: $(pwd)"
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"


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

echo "Launching: $PY_CMD"
eval $PY_CMD
