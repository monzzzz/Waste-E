# Pi0.5 Fine-Tuning Setup for Garbage Picker

This setup follows the official [OpenPI fine-tuning instructions](https://github.com/Physical-Intelligence/openpi#fine-tuning-base-models-on-your-own-data).

## Training Config Adjustments

Based on the official OpenPI documentation, the following parameters were configured in `pi05_garbage_picker` TrainConfig:

### Learning Rate Schedule
```python
lr_schedule=_optimizer.CosineDecaySchedule(
    warmup_steps=500,        # Warmup for 500 steps
    peak_lr=5e-5,            # Peak learning rate (standard for pi05)
    decay_steps=20_000,      # Decay over full training
    decay_lr=1e-6,           # Final learning rate
)
```
- **Why**: Following pi05_aloha_pen_uncap and other official configs
- **Impact**: Smoother training, prevents overfitting

### Training Steps & Batch Size
```python
num_train_steps=20_000   # Standard for pi05 fine-tuning
batch_size=64            # Fits on single A100 80GB
```
- **Why**: All official pi05 fine-tuning examples use 20k steps
- **Impact**: ~4-6 hours on A100, sufficient for convergence

### Logging & Checkpointing
```python
log_interval=100         # Log every 100 steps
save_interval=2_000      # Save checkpoint every 2k steps
keep_period=5_000        # Keep checkpoints every 5k steps
```
- **Why**: Matches official configs, provides good monitoring without excessive disk usage
- **Impact**: 10 checkpoints total (steps 2k, 4k, 6k, 8k, 10k, 12k, 14k, 16k, 18k, 20k)

## Training Workflow (Official OpenPI Process)

### 1. Compute Normalization Statistics
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_garbage_picker
```
- **Required**: Before first training run
- **Output**: `assets/Monzzz/garbage-picker-v1-combined/norm_stats.json`
- **Purpose**: Normalizes state/action values for stable training

### 2. Run Training
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py pi05_garbage_picker --exp-name=garbage_picker_runpod
```
- **XLA_PYTHON_CLIENT_MEM_FRACTION**: Allows JAX to use 90% of GPU memory (recommended)
- **exp-name**: Experiment name for checkpoint organization
- **Checkpoints**: Saved to `checkpoints/pi05_garbage_picker/garbage_picker_runpod/`

### 3. Monitor Training
- **Console**: Loss, metrics logged every 100 steps
- **Weights & Biases**: Full training curves, system metrics
- **Checkpoints**: Latest model saved every 2k steps

## Dataset Configuration

### Data Transforms Pipeline
```python
repack_transforms:
  observation.images.front → images.front
  observation.images.top → images.top
  observation.images.wrist → images.wrist
  observation.state → state
  action → actions

data_transforms:
  GarbagePickerInputs() → Maps to base_0_rgb, top_0_rgb, wrist_0_rgb
  DeltaActions(mask) → Converts joints to deltas, keeps gripper absolute

model_transforms:
  InjectDefaultPrompt() → "Grab the object in front of you and place it in the bin behind you"
  ResizeImages(224, 224) → Resize to PaliGemma input size
  TokenizePrompt() → Convert text to tokens
  PadStatesAndActions(6) → Pad to model dimensions
```

## GPU Memory Requirements

Based on official OpenPI docs:

| Training Type | GPU Memory | Recommended GPU |
|--------------|------------|-----------------|
| Fine-Tuning (LoRA) | > 22.5 GB | RTX 4090 |
| Fine-Tuning (Full) | > 70 GB | A100 80GB / H100 |

**Our setup**: Full fine-tuning with batch_size=64
- **Required**: A100 80GB or H100
- **With XLA_PYTHON_CLIENT_MEM_FRACTION=0.9**: Uses ~72GB during training

### If Running Out of Memory

1. **Enable FSDP** (Fully-Sharded Data Parallelism):
   ```bash
   uv run scripts/train.py pi05_garbage_picker --exp-name=test --fsdp-devices <num_gpus>
   ```

2. **Reduce batch size** in config:
   ```python
   batch_size=32  # or 16
   ```

3. **Disable EMA** (Exponential Moving Average):
   ```python
   ema_decay=None
   ```

## Expected Training Timeline

- **Dataset size**: 24.1k samples (v1-4 through v1-10 combined)
- **Training steps**: 20,000
- **Batch size**: 64
- **Epochs**: ~53 epochs
- **Time on A100**: ~4-6 hours
- **Checkpoints**: 10 saved models (every 2k steps)

## Troubleshooting (from OpenPI docs)

### Missing norm stats error
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_garbage_picker
```

### Dataset download fails
- Check HuggingFace token: `huggingface-cli login`
- Verify dataset access: `Monzzz/garbage-picker-v1-combined`

### Training runs out of GPU memory
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# or enable FSDP
uv run scripts/train.py pi05_garbage_picker --exp-name=test --fsdp-devices 1
```

### Diverging training loss
- Check `assets/Monzzz/garbage-picker-v1-combined/norm_stats.json`
- Look for extreme q01, q99, std values
- Manually adjust if needed

## Next Steps

1. **Upload config to RunPod**:
   ```bash
   # From your local machine
   scp src/openpi/training/config.py runpod:/workspace/openpi/src/openpi/training/
   scp src/openpi/policies/garbage_picker_policy.py runpod:/workspace/openpi/src/openpi/policies/
   ```

2. **Run training**:
   ```bash
   # Inside RunPod container
   bash /workspace/start_training.sh
   ```

3. **Inference after training**:
   ```bash
   uv run scripts/serve_policy.py policy:checkpoint \
     --policy.config=pi05_garbage_picker \
     --policy.dir=checkpoints/pi05_garbage_picker/garbage_picker_runpod/20000
   ```

## References

- [OpenPI Fine-Tuning Guide](https://github.com/Physical-Intelligence/openpi#fine-tuning-base-models-on-your-own-data)
- [LIBERO Example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)
- [DROID Example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README.md)
- [ALOHA Example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/aloha_real/README.md)
