#!/bin/bash
# ODE Initialization Training Script
# Based on CausVid's ODE pretraining approach (https://github.com/tianweiy/CausVid)
# 
# This script trains the causal generator using ODE regression to provide
# a good initialization before DMD distillation training.
#
# After this completes, use the checkpoint for DMD training by updating:
# configs/self_forcing_dmd.yaml -> generator_ckpt: <path_to_ode_checkpoint>

set -e  # Exit on error

#########################################
# Configuration Variables
#########################################

# Distributed Training Setup
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
RDZV_ID=${RDZV_ID:-5235}

# Training Configuration
CONFIG_PATH=${CONFIG_PATH:-"configs/ode_init.yaml"}
LOGDIR=${LOGDIR:-"/storage/latent-forcing/ode_init"}

# WandB Configuration
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-""}
DISABLE_WANDB=${DISABLE_WANDB:-""}

# Data Paths
ODE_PAIRS_PATH=${ODE_PAIRS_PATH:-"/storage/latent-forcing/ode-data/ode_pairs_lmdb"}

# Model Path
WAN_MODEL_PATH=${WAN_MODEL_PATH:-"wan_models/Wan2.1-T2V-1.3B"}

#########################################
# Pre-flight Checks
#########################################

echo "=========================================="
echo "ODE Initialization Training Script"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Nodes: ${NUM_NODES}"
echo "  - GPUs per node: ${NUM_GPUS_PER_NODE}"
echo "  - Total GPUs: $((NUM_NODES * NUM_GPUS_PER_NODE))"
echo "  - Config: ${CONFIG_PATH}"
echo "  - Log directory: ${LOGDIR}"
echo "  - ODE pairs dataset: ${ODE_PAIRS_PATH}"
echo ""

# Check if ODE pairs dataset exists
if [ ! -d "${ODE_PAIRS_PATH}" ]; then
    echo "WARNING: ODE pairs dataset not found at ${ODE_PAIRS_PATH}"
    echo ""
    echo "To generate ODE pairs dataset:"
    echo "  1. Run: torchrun --nproc_per_node 8 scripts/generate_ode_pairs.py \\"
    echo "            --output_folder data/ode_pairs \\"
    echo "            --caption_path prompts/vidprom_filtered_extended.txt"
    echo ""
    echo "  2. Create LMDB: python utils/create_lmdb.py \\"
    echo "                    --data_path data/ode_pairs \\"
    echo "                    --lmdb_path data/ode_pairs_lmdb"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if WAN model exists
if [ ! -d "${WAN_MODEL_PATH}" ]; then
    echo "ERROR: WAN model not found at ${WAN_MODEL_PATH}"
    echo "Please download the Wan2.1-T2V-1.3B model first."
    exit 1
fi

# Check if config exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config file not found at ${CONFIG_PATH}"
    exit 1
fi

echo "Pre-flight checks passed!"
echo ""

#########################################
# Training
#########################################

echo "=========================================="
echo "Starting ODE Initialization Training"
echo "=========================================="
echo ""
echo "Training will save checkpoints to: ${LOGDIR}"
echo "Recommended iterations: 1000-2000 (CausVid uses ~1.5K dataset)"
echo "Checkpoints saved every 100 iterations"
echo ""

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export WORLD_SIZE=$((NUM_NODES * NUM_GPUS_PER_NODE))
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}

# Create output directory
mkdir -p ${LOGDIR}

# Set wandb directory
export WANDB_DIR=$PWD/wandb

# Run ODE initialization training
torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --rdzv_id=${RDZV_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    train.py \
    --config_path ${CONFIG_PATH} \
    --logdir ${LOGDIR} \
    ${WANDB_SAVE_DIR:+--wandb-save-dir $WANDB_SAVE_DIR} \
    ${DISABLE_WANDB}

echo ""
echo "=========================================="
echo "ODE Initialization Training Completed!"
echo "=========================================="
echo ""

# Find the latest checkpoint
LATEST_CHECKPOINT=$(find ${LOGDIR} -name "model.pt" -type f | sort | tail -1)

if [ -n "${LATEST_CHECKPOINT}" ]; then
    echo "Latest checkpoint saved at:"
    echo "  ${LATEST_CHECKPOINT}"
    echo ""
    echo "To use this checkpoint for DMD training:"
    echo "  1. Edit configs/self_forcing_dmd.yaml"
    echo "  2. Update the line: generator_ckpt: ${LATEST_CHECKPOINT}"
    echo "  3. Run: bash train_latent_forcing.sh"
    echo ""
else
    echo "Warning: No checkpoints found in ${LOGDIR}"
    echo "Training may have failed or not saved any checkpoints."
fi

echo "All checkpoints are located in: ${LOGDIR}"
echo ""

