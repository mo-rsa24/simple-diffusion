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
PROFILE="default"
JOB_NAME=""
PARTITION=""
RESUME=""
DRY_RUN=""
LOSS_MASK=""
DIGIT_COLOR_LABEL=""
BBOX_COLOR_LABEL=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment|-e) EXP_ID="$2"; shift ;;
        --run|-r) RUN_ID="$2"; shift ;;
        --dataset|-d) DATASET="$2"; shift ;;
        --task|-t) TASK="$2"; shift ;;
        --color|-c) COLOR="$2"; shift ;;
        --number|-n) NUMBER="$2"; shift ;;
        --gen_model|-g) GEN_MODEL="$2"; shift ;;
        --profile|-p) PROFILE="$2"; shift ;;
        --loss_mask) LOSS_MASK="$2"; shift ;;
        --digit_color_label) DIGIT_COLOR_LABEL="$2"; shift ;;
        --bbox_color_label) BBOX_COLOR_LABEL="$2"; shift ;;
        --resume) RESUME="true" ;;
        --use_wandb) USE_WANDB="true" ;;
        --use_tensorboard) USE_TB="true" ;;
        --job-name) JOB_NAME="$2"; shift ;;
        --partition) PARTITION="$2"; shift ;;
        --dry-run) DRY_RUN="true" ;;
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
    JOB_NAME="${EXP_ID}_run_${RUN_ID}_${TASK:-default}"
    info "Job name not specified. Using default: $JOB_NAME"
fi

#BASE_DIR="/gluster/mmolefe/PhD/simple-diffusion/${JOB_NAME}"
BASE_DIR="/gluster/mmolefe/PhD/simple-diffusion/${EXP_ID}/run_${RUN_ID}/${TASK}/${GEN_MODEL}"
mkdir -p "$BASE_DIR/checkpoints" "$BASE_DIR/logs" "$BASE_DIR/results"

# --- Check for existing checkpoints to infer resume ---
CKPT_DIR="$BASE_DIR/checkpoints"
if [[ "$RESUME" != "true" && -n $(ls -1 "$CKPT_DIR"/epoch_*.ckpt 2>/dev/null | head -n 1) ]]; then
    RESUME="true"
    info "Found checkpoints in $CKPT_DIR - enabling resume"
fi

# --- Summary Table ---
echo -e "\n${DIM}────────── Experiment Launch Summary ──────────${NC}"
print_kv "Experiment" "$EXP_ID"
print_kv "Run ID" "$RUN_ID"
print_kv "Dataset" "$DATASET"
print_kv "Task" "${TASK:-<none>}"
print_kv "Color" "${COLOR:-<none>}"
print_kv "Number" "${NUMBER:-<none>}"
print_kv "Generate Model" "${GEN_MODEL:-<none>}"
print_kv "Profile" "${PROFILE:-<none>}"
print_kv "Loss Mask" "${LOSS_MASK:-<none>}"
print_kv "Digit Color Label" "${DIGIT_COLOR_LABEL:-<none>}"
print_kv "Bbox Color Label" "${BBOX_COLOR_LABEL:-<none>}"
print_kv "Partition" "$PARTITION"
print_kv "Job Name" "$JOB_NAME"
print_kv "Base Dir" "$BASE_DIR"
print_kv "Use WandB" "${USE_WANDB:-<none>}"
print_kv "Use TensorBoard" "${USE_TB:-<none>}"
echo -e "${DIM}───────────────────────────────────────────────${NC}\n"

# --- Final SLURM Command Construction ---
SBATCH_CMD="sbatch --job-name=\"$JOB_NAME\" \
    --partition=\"$PARTITION\" \
    --output=\"$BASE_DIR/logs/slurm-%j.out\" \
    --error=\"$BASE_DIR/logs/slurm-%j.err\" \
    --export=ALL,BASE_DIR=$BASE_DIR,EXP_ID=$EXP_ID,RUN_ID=$RUN_ID,DATASET=$DATASET,TASK=$TASK,COLOR=$COLOR,NUMBER=$NUMBER,GEN_MODEL=$GEN_MODEL,PROFILE=$PROFILE,USE_WANDB=$USE_WANDB,USE_TB=$USE_TB,RESUME=$RESUME,LOSS_MASK=$LOSS_MASK,DIGIT_COLOR_LABEL=$DIGIT_COLOR_LABEL,BBOX_COLOR_LABEL=$BBOX_COLOR_LABEL \
    src/scripts/run_job.slurm"

# --- Execute or Simulate Submission ---
if [[ "$DRY_RUN" == "true" ]]; then
    info "DRY RUN MODE: Validating SLURM directives..."

    # Create the test command by replacing sbatch with sbatch --test-only
    TEST_CMD="${SBATCH_CMD/sbatch/sbatch --test-only}"

    echo -e "${DIM}Running SLURM validation command:${NC}"
    echo -e "${YELLOW}$TEST_CMD${NC}"

    # Execute the test command and capture output
    TEST_OUTPUT=$(eval $TEST_CMD 2>&1)

    if [[ $? -eq 0 && "$TEST_OUTPUT" == *"Job valid"* ]]; then
        success "SLURM directives are valid."
    else
        error "SLURM directive validation failed:"
        echo -e "${RED}$TEST_OUTPUT${NC}"
    fi

    warn "\nThis was only a test. The full command to be executed is:"
    echo -e "${CYAN}$SBATCH_CMD${NC}\n"
    success "✅ DRY RUN complete. No job was submitted."

else
    info "Submitting job to SLURM..."
    echo -e "${CYAN}$SBATCH_CMD${NC}" | tee -a "$LOG_FILE"
    JOB_SUBMIT_OUTPUT=$(eval $SBATCH_CMD 2>&1)
    if [[ $? -eq 0 ]]; then
        success "Job submitted successfully! SLURM output:\n$JOB_SUBMIT_OUTPUT"
    else
        error "Job submission failed! SLURM error:\n$JOB_SUBMIT_OUTPUT"
    fi
fi

success "Experiment launch script finished."


#./src/scripts/launch_experiment.sh --experiment "mnist" --run "1" --dataset "MNIST" --task "generate" --gen_model "vae" --profile sanity --partition bigbatch