# Waste-E: Autonomous AI Robot for Sidewalk Garbage Collection

Waste-E is an AI-powered robotic system built using CARLA Simulator and ROS2, designed to autonomously drive along sidewalks and collect garbage in defined Areas of Interest (AOIs). It can be monitored and controlled via a connected application.

## 🔧 Technologies Used

* [CARLA Simulator](https://carla.org/) v0.9.13
* [carla-ros-bridge](https://github.com/carla-simulator/ros-bridge)
* ROS 2 Foxy Fitzroy (Ubuntu 20.04 Focal)
* Docker & Docker Compose

---

## 🚀 Getting Started

### 1. Carla Simulator (v0.9.13) (Reinforment Learning and Data Collection)

To run the CARLA simulator with ROS bridge and all dependencies:

```bash
docker compose build
docker compose up
```

This will launch:

* CARLA server
* ROS2 container with carla-ros-bridge
* RViz2 (optional visualization)
* ROS nodes for spawning the AI robot

Make sure your environment supports GPU rendering (or offscreen mode if on a headless server).

### 2. UniAD (https://github.com/OpenDriveLab/UniAD)

To download every required file

```bash
./data_download.sh
```

---

## 📦 Project Features

* ✅ **Pedestrian-level autonomous navigation**
* ✅ **Sidewalk path planning**
* ✅ **Onboard RGB camera and perception**
* ✅ **ROS2 integration for modular control**
* ✅ **Configurable pickup points via app interface (WIP)**

## 🧠 AI Behavior (Coming Soon)

* Garbage detection
* Target zone navigation
* Pick-and-drop mechanism
* Human-safe motion planning

---

## 📱 App Control (WIP)

The system will integrate with a mobile/web app to:

* Select garbage zones
* Monitor real-time location
* Trigger missions

---

## 🧪 Testing Environment

Tested on:

* Ubuntu 20.04 (Focal)
* Docker version 24+
* NVIDIA GPU with CUDA support

---

## 📌 Notes

* Ensure CARLA is run with `--RenderOffscreen` if using headless servers like Lambda Labs.
* Attach cameras to robots using `carla_spawn_objects` or Python API if needed.
* Adjust lane detection to follow `LaneType.Sidewalk`.

---

## 📜 License

MIT License © 2025 Elmond Pattanan
