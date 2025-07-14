#!/usr/bin/env bash

# ===----------------------------------------------------------------------===//
#
#  run_experiment.sh
#
#  Purpose:
#  This script provides a way to run an experiment locally, simulating the
#  environment of a SLURM job for debugging purposes. It accepts the same
#  arguments as `launch_experiment.sh` but executes the Python script
#  directly with `debugpy`, allowing you to attach a debugger from an IDE
#  like PyCharm.
#
#  Usage:
#  ./run_experiment.sh -e <exp_id> -r <run_id> -d <dataset> [options]
#  ./run_experiment.sh --dry-run -e <exp_id> ... # To see the command without running
#
# ===----------------------------------------------------------------------===//


# === Shell Script Best Practices ===
set -euo pipefail


# === Pretty Print Helpers ===
NC='\033[0m' # No Color
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
BOLD='\033[1m'
DIM='\033[2m'

info()     { echo -e "\n${CYAN}ℹ️  $*${NC}"; }
success()  { echo -e "${GREEN}✅ $*${NC}"; }
warn()     { echo -e "${YELLOW}⚠️  $*${NC}"; }
error()    { echo -e "\n${RED}❌ $*${NC}"; exit 1; }
print_kv() { printf "${BOLD}%-20s${NC} %s\n" "$1:" "$2"; }
#-------------------------------------------------------------------------------


# === Banner ===
echo -e "${BOLD}${CYAN}\n🚀 LOCAL EXPERIMENT RUNNER (with debugpy)${NC}\n"


# --- Argument Parsing ---
# Initialize variables to avoid unbound variable errors with `set -u`
USE_WANDB=""
USE_TB=""
TASK=""
COLOR=""
NUMBER=""
GEN_MODEL=""
RESUME=""
EXP_ID=""
RUN_ID=""
DATASET=""
DRY_RUN="" # <-- Added for dry-run functionality

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment|-e) EXP_ID="$2"; shift ;;
        --run|-r) RUN_ID="$2"; shift ;;
        --dataset|-d) DATASET="$2"; shift ;;
        --task|-t) TASK="$2"; shift ;;
        --color|-c) COLOR="$2"; shift ;;
        --number|-n) NUMBER="$2"; shift ;;
        --gen_model|-g) GEN_MODEL="$2"; shift ;;
        --resume) RESUME="true" ;;
        --use_wandb) USE_WANDB="true" ;;
        --use_tensorboard) USE_TB="true" ;;
        --dry-run) DRY_RUN="true" ;; # <-- Handle the dry-run flag
        *) error "Unknown parameter: $1" ;;
    esac
    shift
done

info "Parsing experiment arguments..."

# --- Required Args Check ---
if [[ -z "$EXP_ID" || -z "$RUN_ID" || -z "$DATASET" ]]; then
    error "Missing required arguments: --experiment, --run, --dataset"
fi

# --- Project & Directory Setup ---
# This section mimics the directory structure that SLURM jobs would use,
# ensuring consistency between local runs and cluster runs.
JOB_NAME="${EXP_ID}_${RUN_ID}_${TASK:-local_debug}"
BASE_DIR="./jobs/${JOB_NAME}" # Use a local jobs directory

info "--- DIRECTORY SETUP ---"
mkdir -p "$BASE_DIR/checkpoints" "$BASE_DIR/logs" "$BASE_DIR/results"
print_kv "Base Directory" "$BASE_DIR"
echo "-----------------------"

# --- Summary Table ---
info "--- EXPERIMENT SUMMARY ---"
print_kv "Experiment" "$EXP_ID"
print_kv "Run ID" "$RUN_ID"
print_kv "Dataset" "$DATASET"
print_kv "Task" "${TASK:-<none>}"
print_kv "Color" "${COLOR:-<none>}"
print_kv "Number" "${NUMBER:-<none>}"
print_kv "Generate Model" "${GEN_MODEL:-<none>}"
print_kv "Use WandB" "${USE_WANDB:-false}"
print_kv "Use TensorBoard" "${USE_TB:-false}"
print_kv "Resume" "${RESUME:-false}"
echo "--------------------------"


# --- Build Python Command using an Array ---
# Using a bash array is safer and avoids issues with quoting special characters.
info "--- BUILDING PYTHON COMMAND ---"
CKPT_DIR="$BASE_DIR/checkpoints"
LOG_DIR="$BASE_DIR/logs"

# Start with the base command, conditionally adding debugpy
PY_ARGS=("python3")
if [[ "$DRY_RUN" != "true" ]]; then
    PY_ARGS+=("-m" "debugpy" "--listen" "0.0.0.0:5678" "--wait-for-client")
fi

PY_ARGS+=(
    "src/run.py"
    --experiment_id "$EXP_ID"
    --run_id "$RUN_ID"
    --dataset "$DATASET"
    --checkpoint-dir "$CKPT_DIR"
    --log-dir "$LOG_DIR"
)

# Conditionally add optional arguments
if [[ -n "$TASK" && "$TASK" != "None" ]]; then
    PY_ARGS+=(--task "$TASK")
fi
if [[ -n "$COLOR" && "$COLOR" != "None" ]]; then
    PY_ARGS+=(--color "$COLOR")
fi
if [[ -n "$NUMBER" && "$NUMBER" != "None" ]]; then
    PY_ARGS+=(--number "$NUMBER")
fi
if [[ -n "$GEN_MODEL" && "$GEN_MODEL" != "None" ]]; then
    PY_ARGS+=(--gen_model "$GEN_MODEL")
fi
if [[ "$USE_WANDB" == "true" ]]; then
    PY_ARGS+=(--use_wandb)
fi
if [[ "$USE_TB" == "true" ]]; then
    PY_ARGS+=(--use_tensorboard)
fi
if [[ "$RESUME" == "true" ]]; then
   PY_ARGS+=(--resume)
fi
echo "-----------------------------"


# --- Launch Python Script ---
if [[ "$DRY_RUN" == "true" ]]; then
    info "--- DRY RUN MODE ---"
    warn "The following command would be executed (without debugpy):"
    echo ""
    # Use printf to safely quote and print each argument
    printf "%q " "${PY_ARGS[@]}"
    echo -e "\n"
    success "Dry run complete. No script was executed."
else
    info "--- LAUNCHING SCRIPT WITH DEBUGGER ---"
    warn "Script is now waiting for a debugger to attach on port 5678..."
    info "Executing command: ${PY_ARGS[*]}"
    echo "--------------------------------------"
    echo ""

    # Execute the command. The script will pause here until you attach your debugger.
    exec "${PY_ARGS[@]}"
fi

