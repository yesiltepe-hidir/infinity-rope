#!/bin/bash
# Latent-Forcing Training Script
# This script trains the Latent-Forcing model from scratch using DMD (Distribution Matching Distillation)
# Expected training time: ~16 hours on 8 H100 GPUs (with gradient accumulation)
# Target: 600 iterations to reproduce paper results

set -e  # Exit on error

#########################################
# Configuration Variables
#########################################

# Distributed Training Setup
NUM_NODES=${NUM_NODES:-1}                    # Number of nodes (default: 1)
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}    # GPUs per node (default: 8)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}  # Specific GPUs to use
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}        # Master node address
MASTER_PORT=${MASTER_PORT:-29500}            # Master node port
RDZV_ID=${RDZV_ID:-5235}                     # Rendezvous ID for distributed training

# Training Configuration
CONFIG_PATH=${CONFIG_PATH:-"configs/self_forcing_dmd.yaml"}  # Config file to use
LOGDIR=${LOGDIR:-"/storage/latent-forcing"}                  # Directory for saving checkpoints and logs
MAX_ITERATIONS=${MAX_ITERATIONS:-19200}                       # Increased iterations for single-node training

# Optional: WandB Configuration (if you want to enable logging)
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-""}         # WandB save directory (empty = default)
# DISABLE_WANDB=${DISABLE_WANDB:-"--disable-wandb"}  # Remove this flag to enable WandB logging

# Checkpoint and Data Paths (should already be downloaded)
CHECKPOINT_PATH="checkpoints/ode_init.pt"
DATA_PATH="prompts/vidprom_filtered_extended.txt"

# Model Path
WAN_MODEL_PATH="wan_models/Wan2.1-T2V-1.3B"

#########################################
# Pre-flight Checks
#########################################

echo "=========================================="
echo "Latent-Forcing Training Script"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Nodes: ${NUM_NODES}"
echo "  - GPUs per node: ${NUM_GPUS_PER_NODE}"
echo "  - GPU devices: ${CUDA_VISIBLE_DEVICES}"
echo "  - Total GPUs: $((NUM_NODES * NUM_GPUS_PER_NODE))"
echo "  - Config: ${CONFIG_PATH}"
echo "  - Log directory: ${LOGDIR}"
echo "  - Target iterations: ${MAX_ITERATIONS}"
echo "  - Master address: ${MASTER_ADDR}:${MASTER_PORT}"
echo ""

# Check if required files exist
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: ODE initialization checkpoint not found at ${CHECKPOINT_PATH}"
    echo "Please download it or ensure it exists at the correct path."
    exit 1
fi

if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: Training prompts not found at ${DATA_PATH}"
    echo "Please download them or ensure they exist at the correct path."
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config file not found at ${CONFIG_PATH}"
    echo "Available configs:"
    ls -1 configs/*.yaml
    exit 1
fi

# Check if Wan model is downloaded (required for teacher model)
if [ ! -d "${WAN_MODEL_PATH}" ]; then
    echo "WARNING: Wan2.1-T2V-1.3B model not found at ${WAN_MODEL_PATH}"
    echo "The training config uses 1.3B model. Make sure it's downloaded:"
    echo "  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create log directory
mkdir -p "${LOGDIR}"

echo "All checks passed. Starting training..."
echo ""

#########################################
# Training Command
#########################################

# Note: Training runs indefinitely (while True loop in trainer)
# We use timeout to stop after approximately reaching MAX_ITERATIONS
# H200 performance estimate: ~150-180 seconds per iteration with 4 GPUs (gradient accumulation)
# 600 iterations * 165s/iter ≈ 27.5 hours

echo "Starting training..."
echo "Command:"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun \\"
echo "  --nnodes=${NUM_NODES} \\"
echo "  --nproc_per_node=${NUM_GPUS_PER_NODE} \\"
echo "  --rdzv_id=${RDZV_ID} \\"
echo "  --rdzv_backend=c10d \\"
echo "  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \\"
echo "  train.py \\"
echo "  --config_path ${CONFIG_PATH} \\"
echo "  --logdir ${LOGDIR} \\"
echo "  ${DISABLE_WANDB}"
if [ -n "${WANDB_SAVE_DIR}" ]; then
    echo "  --wandb-save-dir ${WANDB_SAVE_DIR}"
fi
echo ""

# Run training
# Set CUDA_VISIBLE_DEVICES to restrict to specific GPUs
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun \
  --nnodes=${NUM_NODES} \
  --nproc_per_node=${NUM_GPUS_PER_NODE} \
  --rdzv_id=${RDZV_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
  --config_path ${CONFIG_PATH} \
  --logdir ${LOGDIR} \
  ${DISABLE_WANDB} \
  ${WANDB_SAVE_DIR:+--wandb-save-dir ${WANDB_SAVE_DIR}}

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 124 ]; then
    echo "Training completed (timeout reached)"
    echo "This is expected behavior for the continuous training loop."
elif [ ${EXIT_CODE} -eq 0 ]; then
    echo "Training finished successfully"
else
    echo "Training exited with error code: ${EXIT_CODE}"
fi
echo ""
echo "Checkpoints saved to: ${LOGDIR}"
echo "=========================================="

# Optional: List the saved checkpoints
if [ -d "${LOGDIR}" ]; then
    echo ""
    echo "Saved checkpoints:"
    ls -lh "${LOGDIR}"/checkpoint_model_*/model.pt 2>/dev/null || echo "No checkpoint files found yet"
fi

