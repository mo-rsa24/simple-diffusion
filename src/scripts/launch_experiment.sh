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
LOG_FILE="launch_experiment.log"

info()    { echo -e "${CYAN}ℹ️  [$DATE_STAMP]$NC $*"; echo "[$DATE_STAMP] $*" >> "$LOG_FILE"; }
success() { echo -e "${GREEN}✅ [$DATE_STAMP]$NC $*"; echo "[$DATE_STAMP] $*" >> "$LOG_FILE"; }
warn()    { echo -e "${YELLOW}⚠️  [$DATE_STAMP]$NC $*"; echo "[$DATE_STAMP] $*" >> "$LOG_FILE"; }
error()   { echo -e "${RED}❌ [$DATE_STAMP]$NC $*"; echo "[$DATE_STAMP] $*" >> "$LOG_FILE"; }

print_kv() { printf "${BOLD}  %-16s${NC} %s\n" "$1:" "$2"; }

# === Banner ===
echo -e "${BOLD}${CYAN}\n🚀 LAUNCHING SLURM EXPERIMENT RUNNER${NC}\n"

# --- Argument Parsing ---
USE_WANDB=""
USE_TB=""
TASK=""
COLOR=""
NUMBER=""
GEN_MODEL=""
JOB_NAME=""
OUTPUT_DIR=""
ERROR_DIR=""
PARTITION=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment|-e) EXP_ID="$2"; shift ;;
        --run|-r) RUN_ID="$2"; shift ;;
        --dataset|-d) DATASET="$2"; shift ;;
        --task|-t) TASK="$2"; shift ;;
        --color|-c) COLOR="$2"; shift ;;
        --number|-n) NUMBER="$2"; shift ;;
        --gen_model|-g) GEN_MODEL="$2"; shift ;;
        --use_wandb) USE_WANDB="true" ;;
        --use_tensorboard) USE_TB="true" ;;
        --job-name) JOB_NAME="$2"; shift ;;
        --output) OUTPUT_DIR="$2"; shift ;;
        --error) ERROR_DIR="$2"; shift ;;
        --partition) PARTITION="$2"; shift ;;
        *) error "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

info "Parsing experiment arguments..."

# --- Required Args Check ---
if [[ -z "$EXP_ID" || -z "$RUN_ID" || -z "$DATASET" ]]; then
    error "Missing required arguments: --experiment --run --dataset"
    exit 1
fi

# --- Partition Smart Selection ---
if [[ -z "$PARTITION" ]]; then
    info "No partition specified; auto-selecting from available SLURM partitions (biggpu → bigbatch → stampede, idle nodes preferred)..."
    # Filter sinfo for idle partitions and pick in priority order
    PARTITION=""
    for p in biggpu bigbatch stampede; do
        PARTITION_CANDIDATE=$(sinfo -h -o "%P %t" | grep -E "^$p" | grep "idle" | awk '{print $1}' | head -n 1 | sed 's/\*//g')
        if [[ -n "$PARTITION_CANDIDATE" ]]; then
            PARTITION="$PARTITION_CANDIDATE"
            success "Selected partition: $PARTITION"
            break
        fi
    done
    if [[ -z "$PARTITION" ]]; then
        error "Could not find an available idle partition (biggpu, bigbatch, stampede)."
        exit 1
    fi
else
    success "User-specified partition: $PARTITION"
fi

# --- Job Name Default ---
if [[ -z "$JOB_NAME" ]]; then
    JOB_NAME="${EXP_ID}_r${RUN_ID}"
    info "Job name not specified. Using default: $JOB_NAME"
fi

# --- Output/Error Dir Defaults ---
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="/gluster/mmolefe/PhD/simple-diffusion/jobs/${JOB_NAME}"
    info "Output directory not specified. Using default: $OUTPUT_DIR"
fi
if [[ -z "$ERROR_DIR" ]]; then
    ERROR_DIR="/gluster/mmolefe/PhD/simple-diffusion/jobs/${JOB_NAME}"
    info "Error directory not specified. Using default: $ERROR_DIR"
fi
mkdir -p "$OUTPUT_DIR" "$ERROR_DIR"

# --- Summary Table ---
echo -e "\n${DIM}────────── Experiment Launch Summary ──────────${NC}"
print_kv "Experiment" "$EXP_ID"
print_kv "Run ID" "$RUN_ID"
print_kv "Dataset" "$DATASET"
print_kv "Task" "${TASK:-<none>}"
print_kv "Color" "${COLOR:-<none>}"
print_kv "Number" "${NUMBER:-<none>}"
print_kv "Generate Model" "${GEN_MODEL:-<none>}"
print_kv "Partition" "$PARTITION"
print_kv "Job Name" "$JOB_NAME"
print_kv "Output Dir" "$OUTPUT_DIR"
print_kv "Error Dir" "$ERROR_DIR"
print_kv "Use WandB" "${USE_WANDB:-<none>}"
print_kv "Use TensorBoard" "${USE_TB:-<none>}"
echo -e "${DIM}───────────────────────────────────────────────${NC}\n"

# --- Final SLURM Command Construction ---
SBATCH_CMD="sbatch --job-name=\"$JOB_NAME\" \
    --partition=\"$PARTITION\" \
    --output=\"$OUTPUT_DIR/%x_%j.out\" \
    --error=\"$ERROR_DIR/%x_%j.err\" \
    src/scripts/run_job.slurm \
    \"$EXP_ID\" \"$RUN_ID\" \"$DATASET\" \"$TASK\" \"$COLOR\" \"$NUMBER\" \"GEN_MODEL\" \"$USE_WANDB\" \"$USE_TB\""

info "Submitting job with the following command:"
echo -e "${CYAN}$SBATCH_CMD${NC}" | tee -a "$LOG_FILE"

JOB_SUBMIT_OUTPUT=$(eval $SBATCH_CMD 2>&1)
if [[ $? -eq 0 ]]; then
    success "Job submitted successfully! SLURM output:\n$JOB_SUBMIT_OUTPUT"
else
    error "Job submission failed! SLURM error:\n$JOB_SUBMIT_OUTPUT"
fi

success "Experiment launch complete."
