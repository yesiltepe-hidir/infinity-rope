#!/bin/bash
# Progressive Training Script for ODE Initialization
# This script iteratively trains with different mla_attn_layers configurations
# Starting from [5,6] and progressively adding more layers up to [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]

set -e  # Exit on error

#########################################
# Configuration Variables
#########################################

# Base configuration
CONFIG_PATH="configs/ode_init.yaml"
BASE_LOGDIR="logs/progressive"
TRAIN_SCRIPT="train_ode_init.sh"

# Layer configuration
START_LAYER=5
END_LAYER=28
LAYER_STEP=2  # Add 2 layers each iteration

#########################################
# Helper Functions
#########################################

# Function to create layer range string
create_layer_range() {
    local start=$1
    local end=$2
    local layers=""
    
    for ((i=start; i<=end; i++)); do
        if [ -n "$layers" ]; then
            layers="${layers},${i}"
        else
            layers="${i}"
        fi
    done
    
    echo "$layers"
}

# Function to update config file
update_config() {
    local layer_range=$1
    local config_file=$2
    local generator_ckpt=$3
    
    echo "Updating config file: ${config_file}"
    echo "Setting mla_attn_layers to: ${layer_range}"
    if [ -n "$generator_ckpt" ]; then
        echo "Setting generator_ckpt to: ${generator_ckpt}"
    fi
    
    # Create a temporary file for the updated config
    local temp_config=$(mktemp)
    
    # Update the config file
    sed "s/^mla_attn_layers:.*/mla_attn_layers: '${layer_range}'/" "$config_file" > "$temp_config"
    sed -i "s/^  mla_attn_layers:.*/  mla_attn_layers: '${layer_range}'/" "$temp_config"
    
    # Update generator_ckpt if provided
    if [ -n "$generator_ckpt" ]; then
        sed -i "s|^generator_ckpt:.*|generator_ckpt: ${generator_ckpt}|" "$temp_config"
    fi
    
    # Replace the original config with the updated one
    mv "$temp_config" "$config_file"
    
    echo "Config updated successfully"
}

# Function to run training for a specific layer range
run_training() {
    local layer_range=$1
    local start_layer=$2
    local end_layer=$3
    local logdir="${BASE_LOGDIR}/${start_layer}-${end_layer}"
    local generator_ckpt="${logdir}/checkpoint_model_$(printf "%06d" $MAX_STEP)/model.pt"
    
    echo ""
    echo "=========================================="
    echo "Training with layers: [${layer_range}]"
    echo "Log directory: ${logdir}"
    echo "=========================================="
    echo ""
    
    # Create log directory
    mkdir -p "$logdir"
    
    # Update config file
    update_config "$layer_range" "$CONFIG_PATH"
    
    # Run training
    echo "Starting training..."
    LOGDIR="$logdir" bash "$TRAIN_SCRIPT"

    update_config "$layer_range" "$CONFIG_PATH" "$generator_ckpt"
    
    echo ""
    echo "Training completed for layers [${layer_range}]"
    echo "Checkpoints saved to: ${logdir}"
    echo ""
}

#########################################
# Main Execution
#########################################

echo "=========================================="
echo "Progressive ODE Initialization Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Start layer: ${START_LAYER}"
echo "  - End layer: ${END_LAYER}"
echo "  - Layer step: ${LAYER_STEP}"
echo "  - Base log directory: ${BASE_LOGDIR}"
echo "  - Config file: ${CONFIG_PATH}"
echo ""

# Create base log directory
echo "Creating log directory: ${BASE_LOGDIR}"
mkdir -p "$BASE_LOGDIR"

# Check if required files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found at ${CONFIG_PATH}"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found at ${TRAIN_SCRIPT}"
    exit 1
fi

echo "Pre-flight checks passed!"
echo ""

# Store original config for restoration
ORIGINAL_CONFIG=$(mktemp)
cp "$CONFIG_PATH" "$ORIGINAL_CONFIG"

# Read max_step from config once
MAX_STEP=$(grep "^max_step:" "$CONFIG_PATH" | sed 's/max_step: *//' | sed 's/ *#.*//' | tr -d ' ')
echo "Using max_step: ${MAX_STEP}"

# Set generator_ckpt to null for the first iteration
echo "Setting generator_ckpt to null for progressive training..."
sed -i "s|^generator_ckpt:.*|generator_ckpt: null|" "$CONFIG_PATH"

# Progressive training loop
current_end=$((START_LAYER + 1))  # Start with 5,6

while [ $current_end -le $END_LAYER ]; do
    # Create layer range string
    layer_range=$(create_layer_range $START_LAYER $current_end)
    
    # Run training for this layer range
    run_training "$layer_range" "$START_LAYER" "$current_end"
    
    # Move to next iteration (add 2 more layers)
    current_end=$((current_end + LAYER_STEP))
done

# Restore original config
echo "Restoring original config file..."
cp "$ORIGINAL_CONFIG" "$CONFIG_PATH"
rm "$ORIGINAL_CONFIG"

echo ""
echo "=========================================="
echo "Progressive Training Completed!"
echo "=========================================="
echo ""
echo "All training sessions completed:"
echo "  - Logs saved to: ${BASE_LOGDIR}"
echo "  - Layer ranges trained:"
for ((end=START_LAYER+1; end<=END_LAYER; end+=LAYER_STEP)); do
    echo "    * ${START_LAYER}-${end}: layers [$(create_layer_range $START_LAYER $end)]"
done
echo ""
echo "Latest checkpoints are available in their respective subdirectories."
echo ""

# Show final directory structure
echo "Directory structure:"
find "$BASE_LOGDIR" -type d | sort
echo ""