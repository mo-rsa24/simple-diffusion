#!/bin/bash
# scripts/submit_experiment.sh

# --- Parse Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment) EXPERIMENT_ID="$2"; shift ;;
        --run) RUN_ID="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --task) TASK="$2"; shift ;;

        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Check Required Args ---
if [[ -z "$EXPERIMENT_ID" || -z "$RUN_ID" || -z "$DATASET" || -z "$TASK" ]]; then
    echo "❌ Usage: ./submit_experiment.sh --experiment 1 --run 3 --dataset cleaned --task TB"
    exit 1
fi


# --- Partition Selection Logic ---
choose_partition() {
    for p in biggpu bigbatch stampede; do
        if sinfo -h -p "$p" > /dev/null 2>&1; then
            echo "$p"
            return
        fi
    done
    echo "❌ No suitable partition found." >&2
    exit 1
}

PARTITION=$(choose_partition)

# --- Logging Paths ---
BASE_LOG_DIR="/gluster/mmolefe/PhD/Super-Position Of Medical Imaging Diffusion Models For Disease Discovery/logs"
LOG_DIR="${BASE_LOG_DIR}/experiment_${EXPERIMENT_ID}/run_${RUN_ID}"
mkdir -p "$LOG_DIR"

# --- Submit Job ---
sbatch \
  --job-name="task_E${EXPERIMENT_ID}_R${RUN_ID}" \
  --partition="$PARTITION" \
  --output="${LOG_DIR}/output.log" \
  --error="${LOG_DIR}/error.log" \
  --export=ALL,EXPERIMENT_ID=$EXPERIMENT_ID,RUN_ID=$RUN_ID,DATASET=$DATASET,TASK=$TASK  \
  slurm/submit_experiment.slurm
