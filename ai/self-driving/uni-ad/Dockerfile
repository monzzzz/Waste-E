FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# === Environment Setup ===
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# === Install system dependencies ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip \
    git wget curl unzip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    gcc g++ build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Use Python 3.9
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    pip install --upgrade pip

# === Install PyTorch 2.0.1 + CUDA 11.8 ===
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# === Build MMCV from source ===
RUN git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    git checkout v1.6.0 && \
    export MMCV_WITH_OPS=1 && \
    export MMCV_CUDA_ARGS="-std=c++17" && \
    pip install -v -e . && \
    cd .. && rm -rf mmcv

# === Install MMDet, MMSeg, MMDet3D (specific versions) ===
RUN pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc6

# === Clone and install UniAD ===
RUN git clone -b v2.0 https://github.com/OpenDriveLab/UniAD.git && \
    cd UniAD && \
    pip install -r requirements.txt && \
    pip install torchmetrics==0.8.2

COPY ./ckpts /workspace/UniAD/ckpts
COPY ./data /workspace/UniAD/data
COPY ./projects /workspace/UniAD/projects
COPY ./tools /workspace/UniAD/tools

# === Set working directory ===
WORKDIR /workspace/UniAD

CMD ["/bin/bash"]