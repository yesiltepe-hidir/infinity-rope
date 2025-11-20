# python inference.py \
#     --config_path configs/self_forcing_dmd.yaml \
#     --checkpoint_path checkpointss/ema_model.pt \
#     --output_folder videos/interactive \
#     --data_path prompts/5.txt \
#     --use_ema \
#     --num_output_frames 168


#!/bin/bash

# Script to run inference for all prompts in generated_prompts_with_time.txt
# Each prompt runs on a different GPU in parallel, with max processes = number of GPU_IDS

# Specify which GPUs to use (space-separated list)
GPU_IDS=(0 1)  # Adjust this to match your available GPUs
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: ${GPU_IDS[*]}"
echo "Number of GPUs: $NUM_GPUS"

# Input file with prompts
PROMPTS_FILE="prompts/extended_prompts.txt"
OUTPUT_FOLDER="videos/extended_prompts"

# Number of seeds/samples to generate per prompt
NUM_SEEDS=3  # Adjust this to generate multiple variations per prompt

# Function to sanitize filename (matching Python's sanitize_filename)
sanitize_filename() {
    local text="$1"
    local max_length=100
    
    # Replace invalid characters with underscores
    text=$(echo "$text" | sed 's/[<>:"/\\|?*]/_/g')
    # Replace multiple spaces/underscores with single underscore
    text=$(echo "$text" | sed 's/[[:space:]_]\+/_/g')
    # Remove leading/trailing underscores and dots
    text=$(echo "$text" | sed 's/^[_.]*//;s/[_.]*$//')
    # Truncate to max_length
    if [ ${#text} -gt $max_length ]; then
        text="${text:0:$max_length}"
    fi
    echo "$text"
}

# Function to extract prompt from line (matches inference.py: everything before ';')
extract_prompt() {
    local line="$1"
    # Get first part before ';' (this matches how inference.py extracts the prompt)
    local prompt="${line%%;*}"
    echo "$prompt"
}

# Create temporary directory for prompt files
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Cleanup function
cleanup() {
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Read prompts and create individual prompt files, also track sanitized prompts
prompt_idx=0
declare -a SANITIZED_PROMPTS
while IFS= read -r prompt_line; do
    if [ -n "$prompt_line" ]; then
        echo "$prompt_line" > "$TEMP_DIR/prompt_${prompt_idx}.txt"
        
        # Extract prompt and sanitize it
        raw_prompt=$(extract_prompt "$prompt_line")
        sanitized_prompt=$(sanitize_filename "$raw_prompt")
        SANITIZED_PROMPTS[$prompt_idx]="$sanitized_prompt"
        
        ((prompt_idx++))
    fi
done < "$PROMPTS_FILE"

TOTAL_PROMPTS=$prompt_idx
echo "Found $TOTAL_PROMPTS prompts to process"

# Base seed (each seed will be base_seed + seed_idx)
BASE_SEED=0

# Round-robin GPU assignment with proper waiting
next_gpu_idx=0
active_jobs=0

# Process prompts and seeds
for ((i=0; i<TOTAL_PROMPTS; i++)); do
    sanitized_prompt="${SANITIZED_PROMPTS[$i]}"
    
    # Process each seed for this prompt
    for ((seed_idx=0; seed_idx<NUM_SEEDS; seed_idx++)); do
        seed=$((BASE_SEED + seed_idx))
        output_path="$OUTPUT_FOLDER/${sanitized_prompt}-${seed}.mp4"
        
        # Check if output file already exists
        if [ -f "$output_path" ]; then
            echo "Skipping prompt $i, seed $seed: Output file already exists: $output_path"
            continue
        fi
        
        # Wait if all GPUs are busy
        while [ $active_jobs -ge $NUM_GPUS ]; do
            wait -n  # Wait for any background job to finish
            ((active_jobs--))
        done
        
        # Assign GPU using round-robin
        GPU_ID=${GPU_IDS[$next_gpu_idx]}
        next_gpu_idx=$(((next_gpu_idx + 1) % NUM_GPUS))
        
        echo "Processing prompt $i, seed $seed on GPU $GPU_ID: $output_path"
        
        # Run inference in background
        (
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            python inference.py \
                --config_path configs/self_forcing_dmd.yaml \
                --checkpoint_path checkpoints/ema_model.pt \
                --output_folder "$OUTPUT_FOLDER" \
                --data_path "$TEMP_DIR/prompt_${i}.txt" \
                --use_ema \
                --num_output_frames 240 \
                --num_samples 1 \
                --seed $seed
        ) &
        
        ((active_jobs++))
    done
done

# Wait for all remaining processes
wait
echo "All prompts processed!"