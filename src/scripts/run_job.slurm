#!/bin/bash
#SBATCH --job-name=tb_diffusion
#SBATCH --output=/gluster/mmolefe/PhD/simple-diffusion/jobs/%x_%j.out
#SBATCH --error=/gluster/mmolefe/PhD/simple-diffusion/jobs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=biggpu

EXP_ID=$1
RUN_ID=$2
DATASET=$3
USE_WANDB=$4
USE_TB=$5

echo "✅ Running experiment=$EXP_ID, run=$RUN_ID, dataset=$DATASET"
cd $SLURM_SUBMIT_DIR
export PYTHONPATH="$(pwd)"

python3 src/run.py \
  --experiment_id "$EXP_ID" \
  --run_id "$RUN_ID" \
  --dataset "$DATASET" \
  $USE_WANDB \
  $USE_TB