#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL      # Notify on job completion and fail
#SBATCH --mail-user=iammrrobot24@gmail.com  # Set your email address
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=${BASE_DIR}/logs/slurm-%j.out
#SBATCH --error=${BASE_DIR}/logs/slurm-%j.err
#SBATCH --exclude=mscluster45

# === Pretty Print Helpers ===
# Define colors and helper functions for logging.
NC='\033[0m' # No Color
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'

info()     { echo -e "\n${CYAN}ℹ️  $*${NC}"; }
success()  { echo -e "${GREEN}✅ $*${NC}"; }
warn()     { echo -e "${YELLOW}⚠️  $*${NC}"; }
print_kv() { printf "${BOLD}%-20s${NC} %s\n" "$1:" "$2"; }
#-------------------------------------------------------------------------------

# === Traceability and Logging ===
# This section prints crucial information to the log file at the start of the job.
info "--- JOB METADATA ---"
print_kv "Job ID" "${SLURM_JOB_ID}"
print_kv "Job Name" "${SLURM_JOB_NAME}"
print_kv "Submission Host" "${SLURM_SUBMIT_HOST}"
print_kv "Execution Node" "$(hostname)"
print_kv "Allocated CPUs" "${SLURM_CPUS_PER_TASK}"
print_kv "Allocated Memory" "${SLURM_MEM_PER_NODE}MB"
echo "--------------------"



# --- Receive Args ---
# These variables are expected to be exported by the launch script.
EXP_ID="${EXP_ID:-$1}"
RUN_ID="${RUN_ID:-$2}"
DATASET="${DATASET:-$3}"
TASK="${TASK:-$4}"
COLOR="${COLOR:-$5}"
NUMBER="${NUMBER:-$6}"
GEN_MODEL="${GEN_MODEL:-$7}"
USE_WANDB="${USE_WANDB:-$8}"
USE_TB="${USE_TB:-$9}"
PROFILE="${PROFILE:-default}" # ✨ Capture the profile variable, defaulting to "default"
RESUME="${RESUME:-}"
BASE_DIR="${BASE_DIR:?}" # Fails if BASE_DIR is not set



# --- Activate Environment & Log Details ---
info "--- ENVIRONMENT SETUP ---"
info "Activating Mamba environment: cxr"
source ~/.bashrc
mamba activate cxr
success "Environment activated."
echo ""
print_kv "Python version" "$(python --version)"
info "Conda environment details:"
echo "-------------------------"


# --- Project Setup ---
info "--- PROJECT SETUP ---"
info "Changing directory to project root..."
cd /home-mscluster/mmolefe/Playground/PhD/simple-diffusion || { echo "Failed to cd to project root!"; exit 1; }
print_kv "Working directory" "$(pwd)"
export PYTHONPATH=$(pwd)
print_kv "PYTHONPATH set to" "$PYTHONPATH"
echo "---------------------"

# --- Resume Logic ---
info "--- RESUME LOGIC ---"
CKPT_DIR="$BASE_DIR/checkpoints"
LOG_DIR="$BASE_DIR/logs"
RESUME_FLAG=""
# Check for the resume flag from the launcher OR if a checkpoint already exists
if [[ "$RESUME" == "true" || -n $(find "$CKPT_DIR" -name "*.ckpt" -print -quit) ]]; then
    warn "Resume flag is set or checkpoints found. Enabling resume."
    RESUME_FLAG="--resume"
else
    info "No resume flag or existing checkpoints. Starting a fresh run."
fi
echo "------------------"

# --- Build Python Command using an Array ---
info "--- BUILDING PYTHON COMMAND ---"
PY_ARGS=(
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
# ✨ Add the profile argument to the python command
if [[ -n "$PROFILE" && "$PROFILE" != "None" ]]; then
    PY_ARGS+=(--profile "$PROFILE")
fi
if [[ "$USE_WANDB" == "true" ]]; then
    PY_ARGS+=(--use_wandb)
fi
if [[ "$USE_TB" == "true" ]]; then
    PY_ARGS+=(--use_tensorboard)
fi
if [[ -n "$RESUME_FLAG" ]]; then
    PY_ARGS+=("$RESUME_FLAG")
fi
if [[ -n "${SLURM_ARRAY_TASK_ID-}" ]]; then # Note: Added default empty value for set -u
    PY_ARGS+=(--array_index "$SLURM_ARRAY_TASK_ID")
fi
echo "-----------------------------"


# --- Launch Python Script ---
info "--- LAUNCHING SCRIPT ---"
info "Executing command: python3 ${PY_ARGS[*]}"
echo "--------------------------"
echo ""

# The `exec` command replaces the shell process with the python process.
# This is slightly more efficient. The `2>&1` redirects stderr to stdout,
# ensuring errors appear chronologically in the main log file.
exec python3 "${PY_ARGS[@]}" 2>&1

success "--- JOB COMPLETE ---"
# Note: This line will only be reached if `exec` is removed.
# With `exec`, the script terminates when the python process finishes.

