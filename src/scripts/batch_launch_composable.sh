#!/usr/bin/env bash
# Launch a grid of composable diffusion experiments on MNIST_COMPOSABLE.
# Usage: ./batch_launch_composable.sh [--partition <name>]

PARTITION="bigbatch"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --partition) PARTITION="$2"; shift ;;
        *) echo "Unknown arg $1"; exit 1 ;;
    esac
    shift
done

MODELS=(comp_unet cascaded guided moe classifier_guided)
VARIANTS=(fg bg)
DIGITS=("" 3 8)
RUN=1
for model in "${MODELS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    for digit in "${DIGITS[@]}"; do
      EXP="mnist_${variant}_${model}_${digit:-all}"
      CMD="./src/scripts/launch_experiment.sh -e ${EXP}_e${RUN} -r $RUN -d MNIST_COMPOSABLE -t generate -c $variant -g $model"
      if [[ -n "$digit" ]]; then
        CMD+=" -n $digit"
      fi
      CMD+=" --partition $PARTITION"
      echo "$CMD"
      eval $CMD
      RUN=$((RUN+1))
    done
  done
done