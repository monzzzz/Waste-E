# OpenPI Pi0.5 Fine-tuning on RunPod

## Setup Instructions

### 1. Build and Push Docker Image

```bash
# Build the image
cd runpod_docker
chmod +x build.sh
./build.sh

# Tag for Docker Hub (replace 'yourusername')
docker tag openpi-finetuning:latest yourusername/openpi-finetuning:latest

# Push to Docker Hub
docker login
docker push yourusername/openpi-finetuning:latest
```

### 2. Deploy on RunPod

1. Go to https://runpod.io
2. Create a new GPU Pod
3. Select GPU: A100 40GB or better (recommended)
4. Container Image: `yourusername/openpi-finetuning:latest`
5. Container Disk: 50GB minimum
6. Volume: 100GB+ (for dataset cache and checkpoints)

### 3. Configure Environment Variables

In RunPod pod settings, add:

```
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_key_here (optional)
```

### 4. Start Training

SSH into your RunPod instance:

```bash
# Upload your training config
scp ../train_pi05/openpi/src/openpi/training/config.py root@your-runpod-ip:/workspace/openpi/src/openpi/training/config.py

# Copy garbage picker policy
scp ../train_pi05/openpi/src/openpi/policies/garbage_picker_policy.py root@your-runpod-ip:/workspace/openpi/src/openpi/policies/

# SSH into pod
ssh root@your-runpod-ip

# Run training script
cd /workspace
bash start_training.sh
```

### 5. Monitor Training

```bash
# View logs
tail -f /workspace/logs/training.log

# Or use Weights & Biases dashboard
# https://wandb.ai/your-username/openpi
```

### 6. Download Checkpoints

After training:

```bash
# From your local machine
scp -r root@your-runpod-ip:/workspace/checkpoints ./local_checkpoints
```

## Training Configuration

Your dataset: `Monzzz/garbage-picker-v1-combined`
- 7 datasets merged (v1-4 through v1-10)
- ~24k total frames
- 6-DoF actions (5 joints + gripper)
- 3 cameras (front, top, wrist)

Recommended training time on A100:
- 20k steps: ~4-6 hours
- 30k steps: ~6-9 hours

## Troubleshooting

### Out of Memory
- Reduce batch_size in config.py
- Use gradient checkpointing
- Use smaller GPU-friendly settings

### Dataset Download Issues
- Ensure HF_TOKEN is set correctly
- Check internet connection
- Pre-download dataset to volume

### Slow Training
- Verify GPU is being used: `nvidia-smi`
- Check data loading isn't bottleneck
- Increase num_workers if CPU has capacity

## Quick Start Commands

```bash
# Build image
./build.sh

# Push to Docker Hub
docker tag openpi-finetuning:latest youruser/openpi-finetuning:latest
docker push youruser/openpi-finetuning:latest

# After deploying on RunPod, upload files
scp ../train_pi05/openpi/src/openpi/training/config.py root@RUNPOD_IP:/workspace/openpi/src/openpi/training/
scp ../train_pi05/openpi/src/openpi/policies/garbage_picker_policy.py root@RUNPOD_IP:/workspace/openpi/src/openpi/policies/
scp start_training.sh root@RUNPOD_IP:/workspace/

# SSH and start
ssh root@RUNPOD_IP
bash /workspace/start_training.sh
```
