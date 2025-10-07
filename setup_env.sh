#!/bin/bash

# Setup script for latent-forcing environment
# Creates a Python 3.10 virtual environment using uv

set -e  # Exit on any error

ENV_NAME="latent-forcing"
ENV_PATH=".venv"
PYTHON_VERSION="3.10"

echo "Setting up latent-forcing environment..."
echo "Environment name: $ENV_NAME"
echo "Environment path: $ENV_PATH"
echo "Python version: $PYTHON_VERSION"

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed or not available in PATH"
    echo "Please install uv first: https://docs.astral.sh/uv/"
    exit 1
fi

# Check if Python 3.10 is available
if ! command -v python3.10 &> /dev/null; then
    echo "Error: Python 3.10 is not installed or not available in PATH"
    echo "Please install Python 3.10 first"
    exit 1
fi

# Create the environment directory if it doesn't exist
mkdir -p "$(dirname "$ENV_PATH")"

# Create virtual environment using uv
echo "Creating virtual environment at $ENV_PATH using uv..."
uv venv "$ENV_PATH" --python "$PYTHON_VERSION"

# Activate the environment
echo "Activating virtual environment..."
source "$ENV_PATH/bin/activate"

# Install requirements from requirements.txt (skip nvidia-pyindex due to build issues)
echo "Installing requirements from requirements.txt using uv..."
PYTHON_BIN="$ENV_PATH/bin/python"
if [ -f "requirements.txt" ]; then
    # Create a temporary requirements file without nvidia-pyindex
    grep -v "nvidia-pyindex" requirements.txt > requirements_temp.txt || true
    uv pip install --python "$PYTHON_BIN" -r requirements_temp.txt \
        || uv pip install --python "$PYTHON_BIN" -r requirements.txt --ignore-requires-python
    rm -f requirements_temp.txt
else
    echo "Warning: requirements.txt not found in current directory"
fi

# Install wheel first (required for flash-attn)
echo "Installing wheel..."
uv pip install --python "$PYTHON_BIN" wheel

# Install flash-attn with no build isolation
echo "Installing flash-attn..."
uv pip install --python "$PYTHON_BIN" flash-attn --no-build-isolation

# Install huggingface_hub with CLI
echo "Installing huggingface_hub[cli]..."
uv pip install --python "$PYTHON_BIN" "huggingface_hub[cli]"

# Download Wan-AI/Wan2.1-T2V-1.3B model
echo "Downloading Wan-AI/Wan2.1-T2V-1.3B model..."
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B

# Download gdhe17/Self-Forcing model
echo "Downloading gdhe17/Self-Forcing model..."
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .

# Download gdhe17/Self-Forcing ODE initialization checkpoint
echo "Downloading gdhe17/Self-Forcing ODE initialization checkpoint..."
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .

# Download gdhe17/Self-Forcing text prompts
echo "Downloading gdhe17/Self-Forcing text prompts..."
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts

# Install project in editable mode using uv
echo "Installing project in editable mode with uv..."
uv pip install --python "$PYTHON_BIN" -e .

echo "Virtual environment '$ENV_NAME' created and configured successfully at $ENV_PATH"
echo ""
echo "To activate the environment, run:"
echo "source $ENV_PATH/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "deactivate"
