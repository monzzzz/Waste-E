# OpenPI Pi0.5 Fine-tuning Environment for RunPod
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /workspace

# Clean any existing cache
RUN rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

# Install system dependencies including Python 3.11
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    linux-libc-dev \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Initialize git-lfs
RUN git lfs install

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip --no-cache-dir

# Install uv for faster package management
RUN python3.11 -m pip install --no-cache-dir uv

# Clone OpenPI repository (Physical Intelligence)
RUN git clone https://github.com/Physical-Intelligence/openpi.git /workspace/openpi

# Install OpenPI dependencies using uv with venv
WORKDIR /workspace/openpi

# Create venv for OpenPI and ensure pip is installed
RUN uv venv .venv --python python3.11 && \
    /workspace/openpi/.venv/bin/python -m ensurepip --upgrade

# Install OpenPI, LeRobot, and fix numpy using the venv's pip directly
RUN /workspace/openpi/.venv/bin/pip3.11 install --no-cache-dir uv && \
    /workspace/openpi/.venv/bin/python -m uv pip install -e . && \
    /workspace/openpi/.venv/bin/pip3.11 install --no-cache-dir git+https://github.com/huggingface/lerobot.git@main && \
    /workspace/openpi/.venv/bin/pip3.11 install --no-cache-dir --force-reinstall numpy==1.26.4

# Install additional dependencies using the venv's pip directly
RUN /workspace/openpi/.venv/bin/pip3.11 install --no-cache-dir \
    huggingface_hub[cli] \
    datasets \
    transformers \
    accelerate \
    wandb \
    tyro \
    jax[cuda12_pip] \
    flax \
    optax \
    einops \
    opencv-python-headless \
    imageio \
    imageio-ffmpeg \
    av \
    torchvision \
    safetensors \
    pyyaml \
    jsonlines \
    termcolor \
    rerun-sdk \
    deepdiff \
    gymnasium

# Verify LeRobot version in the venv
RUN /workspace/openpi/.venv/bin/python -c 'import lerobot; print(f"LeRobot version: {lerobot.__version__}")'

# Set up HuggingFace cache directory
ENV HF_HOME=/workspace/hf_cache
ENV HF_DATASETS_CACHE=/workspace/hf_cache/datasets
RUN mkdir -p ${HF_HOME} ${HF_DATASETS_CACHE}

# Create directories for checkpoints and logs
RUN mkdir -p /workspace/checkpoints /workspace/logs /workspace/assets

# Expose ports for monitoring
EXPOSE 8888 6006 8000

# Set environment variables for optimal performance
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORMS=cuda
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Keep container running
CMD ["tail", "-f", "/dev/null"]