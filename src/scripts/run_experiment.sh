#!/usr/bin/env bash

# === Pretty Print Helpers ===
NC='\033[0m' # No Color
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
BOLD='\033[1m'
DIM='\033[2m'
DATE_STAMP=$(date +"%Y-%m-%d %H:%M:%S")
LOG_FILE="run_experiment.log"

info()    { echo -e "${CYAN}ℹ️  [$DATE_STAMP]$NC $*"; echo "[$DATE_STAMP] $*" >> "$LOG_FILE"; }
success() { echo -e "${GREEN}✅ [$DATE_STAMP]$NC $*"; echo "[$DATE_STAMP] $*" >> "$LOG_FILE"; }
warn()    { echo -e "${YELLOW}⚠️  [$DATE_STAMP]$NC $*"; echo "[$DATE_STAMP] $*" >> "$LOG_FILE"; }
error()   { echo -e "${RED}❌ [$DATE_STAMP]$NC $*"; echo "[$DATE_STAMP] $*" >> "$LOG_FILE"; }
print_kv() { printf "${BOLD}  %-16s${NC} %s\n" "$1:" "$2"; }

echo -e "${BOLD}${CYAN}\n🚀 RUNNING PYTHON EXPERIMENT${NC}\n"

# Initialize optional flags
USE_WANDB=""
USE_TB=""
TASK=""
COLOR=""
NUMBER=""

# Parse CLI arguments
info "Parsing command line arguments..."
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
        *) error "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

info "Experiment Arguments:"
print_kv "Experiment" "$EXP_ID"
print_kv "Run ID" "$RUN_ID"
print_kv "Dataset" "$DATASET"
print_kv "Task" "${TASK:-<none>}"
print_kv "Color" "${COLOR:-<none>}"
print_kv "Number" "${NUMBER:-<none>}"
print_kv "Use WandB" "${USE_WANDB:-<none>}"
print_kv "Use TensorBoard" "${USE_TB:-<none>}"

# Validate required arguments
if [[ -z "$EXP_ID" || -z "$RUN_ID" || -z "$DATASET" ]]; then
    error "Missing required arguments. Usage:"
    echo "   ./run_experiment.sh --experiment <EXP_ID> --run <RUN_ID> --dataset <DATASET> [--task <TASK>] [--color <COLOR>] [--number <NUM>] [--use_wandb] [--use_tensorboard]"
    exit 1
fi

# Optional: Move to project root if script is nested
info "Changing directory to project root..."
cd "$(dirname "$0")/../.." || { error "Failed to change directory to project root!"; exit 1; }
success "Working directory: $(pwd)"

# Set PYTHONPATH to ensure src is accessible
export PYTHONPATH=$(pwd)
info "PYTHONPATH set to: $PYTHONPATH"

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

info "Final experiment launch command:"
echo -e "${CYAN}$PY_CMD${NC}" | tee -a "$LOG_FILE"

# --- Launch Experiment ---
echo -e "${BOLD}${GREEN}🚀 Launching experiment with debugpy...${NC}"
eval $PY_CMD

# For non-debugpy runs, you could use:
# python3 src/run.py ...
