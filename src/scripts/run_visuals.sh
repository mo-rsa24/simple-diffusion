#!/bin/bash

EXPERIMENT=$1
RUN=$2
DATASET=$3

BASE_DIR="./outputs/experiment_${EXPERIMENT}/run_${RUN}"
CHECKPOINT="${BASE_DIR}/ema_epoch*.pt"
CONFIG="config.yaml"

# Forward & Reverse Trajectories
echo "📉 Generating forward & reverse trajectories..."
python scripts/run_trajectories.py --config ${CONFIG} --experiment ${EXPERIMENT} --run ${RUN} --dataset ${DATASET}

# Noise Schedule
echo "📊 Plotting noise schedule..."
python scripts/run_noise_schedule.py --config ${CONFIG}

# Feature Maps
echo "🧠 Visualizing feature maps..."
python scripts/run_feature_maps.py --config ${CONFIG} --experiment ${EXPERIMENT} --run ${RUN}

# t-SNE & UMAP
echo "🔍 Projecting latent features..."
python scripts/run_projection.py --config ${CONFIG} --experiment ${EXPERIMENT} --run ${RUN} --method tsne
python scripts/run_projection.py --config ${CONFIG} --experiment ${EXPERIMENT} --run ${RUN} --method umap