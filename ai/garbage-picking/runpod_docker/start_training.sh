#!/bin/bash
# Script to run inside RunPod container
set -e

echo "=== Starting Pi0.5 Fine-Tuning for Garbage Picker ==="

# Login to HuggingFace (set your token as environment variable)
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
else
    echo "WARNING: HF_TOKEN not set. You may need to login manually."
fi

# Navigate to OpenPI directory
cd /workspace/openpi

# Compute normalization stats (only needed once)
# Following official OpenPI instructions: uv run scripts/compute_norm_stats.py --config-name pi05_garbage_picker
if [ ! -f "/workspace/openpi/assets/Monzzz/garbage-picker-v1-combined/norm_stats.json" ]; then
    echo "Computing normalization statistics..."
    python scripts/compute_norm_stats.py --config-name pi05_garbage_picker
else
    echo "Norm stats already exist at /workspace/openpi/assets/Monzzz/garbage-picker-v1-combined/norm_stats.json"
fi

# Set JAX to use 90% of GPU memory (recommended by OpenPI for maximum GPU utilization)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Start training
# Following official OpenPI instructions: uv run scripts/train.py <config_name> --exp-name=<name>
echo "Starting Pi0.5 fine-tuning with XLA_PYTHON_CLIENT_MEM_FRACTION=0.9..."
python scripts/train.py pi05_garbage_picker \
    --exp-name garbage_picker_runpod \
    --wandb-enabled

echo "=== Training Complete ==="
echo "Checkpoints saved to: /workspace/openpi/checkpoints/pi05_garbage_picker/garbage_picker_runpod/"
