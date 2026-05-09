# Waste-E Isaac Sim Module

Autonomous robot simulation using NVIDIA Isaac Sim 5.x, the WPedestrian SiT dataset, and the Alpamayo 1.5-10B vision-language model for end-to-end navigation.

---

## Architecture

```
SiT Dataset (real camera images + pedestrian trajectories)
        │
        ▼
Isaac Sim World  ←─  Pedestrian agents animated from SiT CSV trajectories
        │
        ▼
SiTImageLoader  ──►  real RGB frames (left / front / right cameras)
        │
        ▼
Alpamayo 1.5-10B  ──►  predicted waypoints (ego-frame x/y)
        │
        ▼
Pure-pursuit controller  ──►  linear_x / angular_z drive commands
        │
        ▼
Output video (left | front + BEV overlay | right) + metrics.csv
```

---

## Key Files

| File | Purpose |
|------|---------|
| `run_simulation.py` | Entry point — creates `SimulationApp` and runs `WasteESimEnv` |
| `sim_env.py` | Main simulation environment: scene setup, pedestrians, cameras, loop |
| `alpamayo_inference.py` | Alpamayo wrapper — spawns Python 3.12 subprocess, sends frames, receives waypoints |
| `alpamayo_worker.py` | Subprocess worker (Python 3.12) — loads Alpamayo model and runs inference |
| `pedestrian_loader.py` | Parses SiT CSV trajectory files and exposes per-frame position/heading |
| `config.yaml` | All tuneable parameters (paths, physics, camera, logging) |

---

## Setup

### 1. Install missing system libraries

```bash
apt-get install -y libxt6 libxrandr2 xvfb ffmpeg unzip
```

### 2. Start a virtual display (required for Isaac Sim even in headless mode)

```bash
Xvfb :99 -screen 0 1280x1024x24 &
export DISPLAY=:99
```

### 3. Download the SiT dataset

```bash
pip install gdown
cd /workspace/Waste-E
gdown --folder "https://drive.google.com/drive/folders/1t55r1kpxlLQP458fS-DjWvOq8aLPkFWI" -O sit_dataset/

# Extract a sequence
unzip sit_dataset/FullSet_v.1.0.0/Cafeteria/Cafeteria_1.zip \
      -d sit_dataset/FullSet_v.1.0.0/Cafeteria/Cafeteria_1/
```

### 4. Run the simulation

```bash
cd /workspace/Waste-E
DISPLAY=:99 timeout 600 python3.11 isaacsim/run_simulation.py --headless --max_steps 300
```

Outputs are written to `output/`:
- `output/frames/frame_XXXXXX.jpg` — per-step composite images
- `output/simulation.mp4` — stitched video
- `output/metrics.csv` — per-step velocity and confidence
- `output/alpamayo_load.log` — model load diagnostics

---

## Configuration (`config.yaml`)

| Key | Description |
|-----|-------------|
| `simulation.max_steps` | Number of physics steps to run |
| `simulation.no_cameras` | Set `false` to enable cameras; `true` skips camera spawning |
| `alpamayo.model_path` | Path to Alpamayo 1.5-10B weights directory |
| `alpamayo.sit_sequence_dir` | Path to extracted SiT sequence (e.g. `Cafeteria_1/`) |
| `alpamayo.device` | `cuda` or `cpu` for model inference |
| `logging.save_video` | Enable/disable frame and video saving |

---

## GPU Usage

Isaac Sim physics runs on **GPU 0** (`physics_gpu=0`, `cudaDevice=0`).  
The Alpamayo model runs in a separate Python 3.12 subprocess via `/opt/alpamayo-env/bin/python` and uses the same GPU via `device_map=cuda`.

Monitor GPU during a run:
```bash
watch -n 1 nvidia-smi
```

Expected: ~26 GB VRAM used, 85–100% GPU utilisation during inference.

---

## Real-World Camera Images (SiT Dataset)

The SiT dataset provides 5× Basler RGB cameras (1920×1200, 10 Hz) arranged in a 360° ring around a Clearpath Husky robot. Camera mapping used in this simulation:

Alpamayo was trained on **4 specific cameras** from the Physical AI AV dataset:

| Alpamayo index | Camera name | SiT cam used | Notes |
|----------------|-------------|--------------|-------|
| 0 | Front-left 120° (`camera_cross_left_120fov`) | cam 1 | left camera |
| 1 | Front wide 120° (`camera_front_wide_120fov`) | cam 4 | front camera |
| 2 | Front-right 120° (`camera_cross_right_120fov`) | cam 5 | right camera |
| 6 | Front telephoto 30° (`camera_front_tele_30fov`) | cam 4 | SiT has no tele — reuses front |

SiT does not have a telephoto camera, so the front image (cam 3) is reused for the telephoto slot. This is an approximation — Alpamayo still receives a valid 4-camera input matching its training format.

Real images replace Isaac Sim's synthetic renderer. `render=False` is set when SiT images are loaded, so no GPU time is wasted on synthetic rendering.

---

## Output Video Annotations

Each frame in the output video shows:
- **Left / Front / Right** — real SiT camera views side-by-side
- **BEV inset** — bird's-eye view of the predicted trajectory overlaid on the front camera. Robot shown as a green triangle; predicted waypoints as cyan dots connected by a line; grid lines every 2 m
- **HUD bar** — step number, linear velocity (m/s), angular rate (rad/s), and model confidence
