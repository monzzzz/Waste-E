# Waste-E

Autonomous AI + robotics stack for sidewalk garbage collection.

<p align="center">
  <img src="assets/images/waste-e-robot.jpg" alt="Waste-E robot platform" width="860">
</p>

Waste-E combines:
- simulation (CARLA + Isaac Sim),
- perception and planning,
- hardware control on Orange Pi / Raspberry Pi,
- robot-learning training and inference workflows.

This README is the top-level guide for what is in this repository and how the pieces fit.

---

## Table of Contents

- [1) What This Repo Contains](#1-what-this-repo-contains)
- [2) System Architecture](#2-system-architecture)
- [3) Repository Map (Detailed)](#3-repository-map-detailed)
- [4) Quick Start by Workflow](#4-quick-start-by-workflow)
- [5) Hardware Runtime (Laptop + OrangePi + RaspPi)](#5-hardware-runtime-laptop--orangepi--rasppi)
- [6) Training / Inference Workflows](#6-training--inference-workflows)
- [7) Outputs and Artifacts](#7-outputs-and-artifacts)
- [8) Ports and Services](#8-ports-and-services)
- [9) Environment Notes](#9-environment-notes)
- [10) Known Notes](#10-known-notes)
- [11) License](#11-license)

---

## 1) What This Repo Contains

Top-level modules:

- `ai/`
  - Sidewalk coverage path planning (`ai/path-planning`)
  - YOLOv8 garbage detection (`ai/garbage-detection`)
  - Garbage-picking model training/inference tooling (`ai/garbage-picking`)
  - UniAD assets/setup (`ai/self-driving/uni-ad`)
- `carla/`
  - ROS2 + CARLA docker stack
  - CARLA bridge + Alpamayo driving runner
  - Web control panel for live command injection
- `hardware/`
  - Robot-side hardware interfaces (GPS, IMU, motors, encoders)
  - Distributed telemetry senders for OrangePi/RaspPi
  - Central operations dashboard
- `isaacsim/`
  - Isaac Sim environment with WPedestrian SiT trajectories + Alpamayo inference
- `output/`
  - Example simulation artifacts (`metrics.csv`, load logs)

---

## 2) System Architecture

Core end-to-end loop:

1. Plan area coverage path from road network shapefiles.
2. Navigate via waypoints (sim or real hardware).
3. Perceive with multi-camera streams.
4. Detect/act on garbage targets.
5. Log telemetry/video for debugging and iteration.

You have multiple execution modes in this repo:

- **CARLA mode:** `carla/run_carla_sim.py` + optional `carla/web_control.py`
- **Isaac Sim mode:** `isaacsim/run_simulation.py`
- **Hardware mode:** `hardware/central_dashboard.py` + board-side senders
- **Manipulation training mode:** `ai/garbage-picking/*`

---

## 3) Repository Map (Detailed)

### Root

- `README.md`: this guide.
- `assets/images/waste-e-robot.jpg`: README hero image.
- `.gitignore`: excludes large/generated artifacts (`output`, `output_carla`, `sit_dataset`, recordings, etc.).
- `=4.57.1`: pip install log artifact (not source code).

### `ai/`

- `ai/camera_connector.py`
  - Minimal Flask receiver for JPEG uploads and MJPEG stream viewing.
- `ai/latest.jpg`
  - Legacy image snapshot.

#### `ai/path-planning/`

- Purpose: interactive sidewalk coverage planning from shapefiles.
- Key files:
  - `main.py`: interactive flow (select area, start point, coverage path).
  - `path_planner.py`: graph generation + coverage route logic.
  - `navigation_controller.py`: turn/heading command generation.
  - `simulate_navigation.py`: visual path-following simulation.
  - `download_map_tiles.py`, `tile_basemap.py`: map tile utilities.
  - `requirements.txt`: geospatial stack (`geopandas`, `networkx`, `shapely`, etc.).
- Typical outputs:
  - `full_network.png`, `selected_area.png`, `coverage_path.png`, `waypoints.txt`.

#### `ai/garbage-detection/`

- Purpose: YOLOv8-based garbage detection with motor command decision logic.
- Key files:
  - `main.py`: CLI test harness for front/left/right images.
  - `detection.py`: model inference + motor speed decision (`left/right/status`).
  - `yolov8n.pt`: local model weights.
  - `data/`, `result/`: sample inputs and annotated outputs.

#### `ai/garbage-picking/`

- Purpose: manipulation policy tooling (LeRobot/OpenPI-based workflows).
- Submodules:
  - `lerobot_setup/`
    - dataset collection helpers, camera checks, dataset fix/rename scripts.
    - README explains LeRobot dataset workflow.
  - `runpod_docker/`
    - Docker + scripts to run Pi0.5 fine-tuning on RunPod GPUs.
    - includes `start_training.sh`, training setup notes.
  - `train_pi05/openpi/`
    - large OpenPI codebase copy (models, training, examples).
    - includes custom `garbage_picker_policy.py`.
    - includes custom training config `pi05_garbage_picker` in `src/openpi/training/config.py`.
  - `inference_pi05/`
    - local/remote inference scripts and debugging tools.
    - includes Flask inference server (`inference_server.py`) and local arm client (`inference_local.py`).

#### `ai/self-driving/uni-ad/`

- Purpose: UniAD environment scaffolding and downloads.
- Key files:
  - `Dockerfile`: CUDA + UniAD environment build.
  - `data_download.sh`: downloads checkpoints/info files.
  - `requirements.txt`, `v1.0-download.py`.
  - `python-env/`: local Python environment folder (large).

### `carla/`

- Purpose: CARLA simulation stack and live control.
- Key files:
  - `docker-compose.yaml`: CARLA + ROS bridge + spawn + custom ROS2 container + RViz.
  - `start_carla.sh`: headless CARLA server startup helper.
  - `run_carla_sim.py`: main Waste-E CARLA loop + Alpamayo integration.
  - `carla_bridge_worker.py`: Python 3.7 bridge worker for CARLA API.
  - `web_control.py`: web UI to control nav text and restart/snapshot simulation.
  - `domain_adapter.py`: optional SDXL-Turbo sim-to-real image adaptation.
- Supporting folders:
  - `docker/`: ROS container images and helper configs.
  - `scripts/`: compose snippets (`carla.yaml`, ROS demo).
  - `workshop/`: tutorial assets/slides.
  - `docs/`: architecture figure + presentation PDF.

### `hardware/`

- Purpose: on-robot controls + distributed camera/telemetry dashboards.
- Key files:
  - `central_dashboard.py`
    - central web server for device registration, status, camera aggregation, recording, drive relay.
  - `make_composite.py`
    - builds composite video from recordings + telemetry overlays.

#### `hardware/orangepi/`

- Main control path:
  - `main.py`: autonomous waypoint robot runtime.
  - `navigation.py`: waypoint follower.
  - `motor_driver.py`, `motorencoder.py`, `GPS.py`, `IMUsensor.py`.
- Web/telemetry:
  - `dashboard.py`: local OrangePi sensor + camera dashboard.
  - `data_sender.py`: registers to central dashboard and streams state/cameras.
  - `camera.py`: camera streaming helper.

#### `hardware/rasppi/`

- `data_sender.py`
  - camera discovery + streaming, sender registration, optional RunPod arm loop.
- `camera_stream.py`
  - MJPEG fallback camera server.
- `recordings/`
  - local session artifacts (`session.json`, telemetry, tracks) if not ignored.

### `isaacsim/`

- Purpose: Isaac Sim + SiT trajectory + Alpamayo closed-loop simulation.
- Key files:
  - `run_simulation.py`: SimulationApp entrypoint + config overrides.
  - `sim_env.py`: main environment + sensor/agent loop.
  - `pedestrian_loader.py`: WPedestrian SiT CSV loader.
  - `alpamayo_inference.py`: model wrapper in main process.
  - `alpamayo_worker.py`: Python 3.12 worker subprocess for model inference.
  - `visualizer.py`: annotated composite visual output.
  - `config.yaml`: simulation/model/camera/logging parameters.
  - `data/wpedestrian_sit/*.csv`: sample trajectories.
  - `note.txt`: setup notes (Isaac Sim + Alpamayo env).

### `output/`

- Example output files:
  - `metrics.csv`
  - `alpamayo_load.log`

---

## 4) Quick Start by Workflow

### A) Path Planning Only

```bash
cd ai/path-planning
pip install -r requirements.txt
python main.py
```

### B) Garbage Detection Only

```bash
cd ai/garbage-detection
pip install -r requirements.txt
python main.py --front data/front.jpg --left data/left.jpg --right data/right.jpg
```

### C) CARLA ROS2 Docker Stack

```bash
cd carla
docker compose up --build
```

### D) Waste-E CARLA + Alpamayo Runner

Start CARLA server first:

```bash
cd carla
bash start_carla.sh
```

Then run simulation:

```bash
cd carla
python3 run_carla_sim.py --steps 300 --num-scenes 1
```

Optional web control:

```bash
cd carla
python3 web_control.py --port 7860 --steps 300 --num-scenes 1
```

### E) Isaac Sim Flow

```bash
cd isaacsim
python3 run_simulation.py --headless --max_steps 300
```

Or run from Isaac Sim Python launcher as documented in `isaacsim/run_simulation.py`.

---

## 5) Hardware Runtime (Laptop + OrangePi + RaspPi)

### 1) Start central dashboard (laptop/control machine)

```bash
python3 hardware/central_dashboard.py
```

Default: `http://0.0.0.0:9000`

### 2) Start OrangePi sender

```bash
cd hardware/orangepi
python3 data_sender.py --server http://<laptop-ip>:9000 --my-ip <orangepi-ip>
```

If using local OrangePi dashboard as sensor source:

```bash
python3 data_sender.py --server http://<laptop-ip>:9000 --dashboard http://localhost:8888
```

### 3) Start Raspberry Pi sender

```bash
cd hardware/rasppi
python3 data_sender.py --server http://<laptop-ip>:9000 --my-ip <rasppi-ip>
```

Optional RunPod arm inference mode:

```bash
python3 data_sender.py \
  --server http://<laptop-ip>:9000 \
  --my-ip <rasppi-ip> \
  --runpod-url https://<runpod-endpoint>
```

### 4) Optional onboard autonomous nav mode (OrangePi)

`hardware/orangepi/main.py` expects `waypoints.txt` in its working directory.

```bash
cd hardware/orangepi
python3 main.py
```

---

## 6) Training / Inference Workflows

### Garbage Picking (OpenPI / Pi0.5)

Main training subtree:
- `ai/garbage-picking/train_pi05/openpi`

Custom policy transform:
- `ai/garbage-picking/train_pi05/openpi/src/openpi/policies/garbage_picker_policy.py`

Custom config name:
- `pi05_garbage_picker` in `src/openpi/training/config.py`

RunPod-focused scripts:
- `ai/garbage-picking/runpod_docker/build.sh`
- `ai/garbage-picking/runpod_docker/start_training.sh`

Remote inference server:
- `ai/garbage-picking/inference_pi05/inference_server.py` (`/predict`, `/health`)

Local arm client:
- `ai/garbage-picking/inference_pi05/inference_local.py`

Offline replay/repredict:
- `replay_dataset.py`, `repredict_data.py`

### LeRobot Data Collection Helpers

See:
- `ai/garbage-picking/lerobot_setup/README.md`
- scripts in `ai/garbage-picking/lerobot_setup/`

---

## 7) Outputs and Artifacts

Common output locations:

- `output/`
  - IsaacSim-related metrics/logs example
- `output_carla/` (ignored by git)
  - `frames/scene_*/frame_*.jpg`
  - `scene_*.mp4`
  - `metrics.csv`
- `ai/path-planning/`
  - planning visualizations + `waypoints.txt`
- `ai/garbage-detection/result/`
  - detection render outputs
- `hardware/rasppi/recordings/` (ignored by git)
  - session telemetry/video artifacts

---

## 8) Ports and Services

Defaults used across code:

- `9000`: central dashboard (`hardware/central_dashboard.py`)
- `8890`: OrangePi camera/data sender default (`hardware/orangepi/data_sender.py`)
- `8888`: OrangePi local dashboard (`hardware/orangepi/dashboard.py`)
- `5000`: RaspPi MJPEG fallback camera stream
- `8554`: MediaMTX RTSP (board-side WebRTC pipeline)
- `8889`: MediaMTX WebRTC
- `7860`: CARLA web control UI
- `8001`: Pi0.5 inference server (`inference_server.py`)
- `2000-2002`: CARLA server ports

---

## 9) Environment Notes

Different modules use different Python/runtime environments:

- `carla/carla_bridge_worker.py`: Python 3.7 + CARLA egg
- `carla/run_carla_sim.py`: Python 3.11
- `isaacsim/alpamayo_worker.py`: Python 3.12
- `ai/garbage-picking/train_pi05/openpi`: OpenPI/uv-managed environment (JAX/PyTorch workflows)
- `hardware/*`: board-specific Python environment with GPIO/I2C/UART libraries

GPU-heavy components:

- CARLA sim / bridge
- Isaac Sim + Alpamayo
- OpenPI training and remote inference

---

## 10) Known Notes

- Several subprojects are large imported/vendor trees (especially `ai/garbage-picking/train_pi05/openpi` and `ai/self-driving/uni-ad/python-env`).
- Some module READMEs are minimal or empty, so this top-level README is the canonical overview.
- There are multiple parallel experiments in this repo (CARLA, IsaacSim, real hardware, manipulation training), so not every folder is required for every workflow.

---

## 11) License

MIT License © 2025 Elmond Pattanan
