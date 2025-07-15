#!/usr/bin/env bash

# ===----------------------------------------------------------------------===//
#
#  debug_experiment.sh
#
#  Purpose:
#  This script is a convenience wrapper for running a specific debugging
#  configuration through the main launch_experiment.sh script.
#
#  It pre-fills a set of common experiment parameters and automatically
#  enables the --dry-run flag. This allows you to quickly test the entire
#  setup, including SLURM directive validation and Python command
#  construction, without submitting an actual job.
#
#  Usage:
#  ./debug_experiment.sh
#
# ===----------------------------------------------------------------------===//

# Find the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Call the main launch script with the predefined debug parameters
# and the --dry-run flag.
"$SCRIPT_DIR/launch_experiment.sh" \
    --experiment "comp_mnist" \
    --run "99" \
    --dataset "MNIST_COMPOSABLE" \
    --task "generate" \
    --color "fg" \
    --number "8" \
    --gen_model "vanilla" \
    --dry-run

