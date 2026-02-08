#!/bin/bash

# Setup script for latent-forcing environment
# Creates a Python 3.10 virtual environment using venv

set -e  # Exit on any error

ENV_PATH="../infinity_rope_env"
PYTHON_VERSION="3.10"

echo "Setting up the environment..."
echo "Environment path: $ENV_PATH"
echo "Python version: $PYTHON_VERSION"

# Check if Python 3.10 is available
if ! command -v python3.10 &> /dev/null; then
    echo "Error: Python 3.10 is not installed or not available in PATH"
    echo "Please install Python 3.10 first"
    exit 1
fi

# Create the environment directory if it doesn't exist
mkdir -p "$(dirname "$ENV_PATH")"

# Create virtual environment using venv
echo "Creating virtual environment at $ENV_PATH..."
python3.10 -m venv "$ENV_PATH"

# Activate the environment
echo "Activating virtual environment..."
source "$ENV_PATH/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install setuptools and wheel (required for building packages like CLIP)
echo "Installing setuptools and wheel..."
pip install setuptools wheel

# Install requirements from requirements.txt (skip nvidia-pyindex and CLIP due to build issues)
echo "Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    # Create a temporary requirements file without nvidia-pyindex and CLIP
    grep -v "nvidia-pyindex" requirements.txt | grep -v "git+https://github.com/openai/CLIP.git" > requirements_temp.txt || true
    pip install -r requirements_temp.txt || pip install -r requirements.txt --ignore-requires-python
    rm -f requirements_temp.txt
    
    # Install CLIP separately with no build isolation (requires setuptools to be available)
    echo "Installing CLIP from GitHub..."
    pip install git+https://github.com/openai/CLIP.git --no-build-isolation || echo "Warning: CLIP installation failed, you may need to install it manually"
else
    echo "Warning: requirements.txt not found in current directory"
fi

# Install flash-attn with no build isolation
echo "Installing flash-attn..."
pip install flash-attn --no-build-isolation

# Install huggingface_hub with CLI
echo "Installing huggingface_hub with CLI support..."
pip install -U "huggingface_hub[cli]"

# Download Wan-AI/Wan2.1-T2V-1.3B model
echo "Downloading Wan-AI/Wan2.1-T2V-1.3B model..."
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-T2V-1.3B', local_dir='wan_models/Wan2.1-T2V-1.3B', local_dir_use_symlinks=False)"

# Download self-forcing checkpoint with attention sink
echo "Downloading self-forcing checkpoint..."
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SOTAMak1r/Infinite-Forcing', local_dir='checkpoints/')"

# Download causal forcing checkpoint
echo "Downloading causal forcing checkpoint..."
python -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('checkpoints/chunkwise', exist_ok=True); hf_hub_download(repo_id='zhuhz22/Causal-Forcing', filename='chunkwise/causal_forcing.pt', local_dir='checkpoints', local_dir_use_symlinks=False)"

# Run setup.py develop
echo "Running setup.py develop..."
python setup.py develop

echo "Virtual environment created and configured successfully at $ENV_PATH"
echo ""
echo "To activate the environment, run:"
echo "source $ENV_PATH/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "deactivate"