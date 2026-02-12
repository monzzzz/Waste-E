# LeRobot Data Collection for Waste-E

This directory contains scripts for collecting robot demonstration data using the LeRobot framework.

## Overview

LeRobot is a library for robot learning that provides tools for:
- Collecting and storing robot demonstration data
- Training policies from demonstrations
- Evaluating trained policies

The code you mentioned is indeed from the LeRobot library - it's the core `LeRobotDataset` class that handles data storage and retrieval.

## Files

- `collect_data.py` - Basic data collection script with dummy data (for testing)
- `collect_real_robot_data.py` - Real robot data collection with camera and sensors
- `data_structure.py` - (Your file) Can be used for custom dataset utilities
- `setup.py` - (Your file)Can be used for installation/setup scripts

## Installation

```bash
# Activate your conda environment
conda activate lerobot

# Install LeRobot
pip install lerobot

# Install additional dependencies
pip install opencv-python pillow numpy torch
```

## Quick Start

### 1. Test with Dummy Data

```bash
python collect_data.py \
    --repo-id waste-e/test-dataset \
    --num-episodes 5 \
    --frames-per-episode 100 \
    --fps 30
```

### 2. Collect Real Robot Data

```bash
python collect_real_robot_data.py \
    --repo-id waste-e/garbage-picking \
    --camera-id 0 \
    --num-episodes 50 \
    --episode-duration 15.0 \
    --fps 30
```

## Dataset Structure

LeRobot datasets have this structure:

```
dataset_name/
├── data/                    # Parquet files with observations & actions
│   ├── chunk-000/
│   │   ├── file-000.parquet
│   │   └── file-001.parquet
│   └── chunk-001/
├── meta/                    # Metadata
│   ├── episodes/
│   ├── info.json           # Dataset info (fps, features, etc.)
│   ├── stats.json          # Normalization statistics
│   └── tasks.parquet       # Task descriptions
└── videos/                 # Compressed video files
    └── observation.images.top/
        ├── chunk-000/
        └── chunk-001/
```

## Customizing for Your Robot

Edit `collect_real_robot_data.py` and modify the `RobotInterface` class:

### 1. Connect to Your Hardware

```python
class RobotInterface:
    def __init__(self):
        # Import your hardware modules
        from hardware.motor import MotorController
        from hardware.GPS import GPSReader
        from hardware.IMUsensor import IMUReader
        
        self.motors = MotorController()
        self.gps = GPSReader()
        self.imu = IMUReader()
```

### 2. Update State Reading

```python
def get_robot_state(self):
    return {
        "motor_positions": self.motors.get_positions(),
        "motor_velocities": self.motors.get_velocities(),
        "gps_lat": self.gps.latitude,
        "gps_lon": self.gps.longitude,
        "imu_orientation": self.imu.get_orientation(),
    }
```

### 3. Update Action Execution

```python
def execute_action(self, action):
    self.motors.set_commands(action)
```

## Features Configuration

Define what sensors/actuators your robot has:

```python
features = {
    # Cameras
    "observation.images.top": {
        "dtype": "video",  # or "image"
        "shape": (3, 480, 640),  # (C, H, W)
        "names": ["C", "H", "W"],
    },
    
    # Robot state (proprioception)
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),  # Define size based on your state
        "names": ["joint1", "joint2", ...],  # Name each dimension
    },
    
    # Actions
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["joint1_cmd", "joint2_cmd", ...],
    },
}
```

## Data Collection Workflow

1. **Setup Environment**
   ```bash
   conda activate lerobot
   ```

2. **Test Camera**
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL')"
   ```

3. **Collect Episodes**

    ```
    lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower_v1 --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=so101_leader_v1
    ```
    
    ```
    lerobot-record   --robot.type=so101_follower   --robot.port=/dev/ttyACM0   --robot.id=my_awesome_follower_arm   --robot.cameras='{"front":{"type":"opencv","index_or_path":4,"width":1280,"height":720,"fps":10,"warmup_s":5},"top":{"type":"opencv","index_or_path":6,"width":1280,"height":720,"fps":10,"warmup_s":5},"wrist":{"type":"opencv","index_or_path":2,"width":1280,"height":720,"fps":10,"warmup_s":5}}'   --teleop.type=so101_leader   --teleop.port=/dev/ttyACM1   --teleop.id=garbage-picker-v1   --display_data=true   --dataset.repo_id=Monzzz/garbage-picker-v1-2   --dataset.num_episodes=3   --dataset.single_task="Grab the object in front of you and place it in the bin behind you."
    ```

   - Start the collection script
   - Perform robot demonstrations (teleoperation)
   - Data is automatically saved


4. **Verify Dataset**
   ```python
   from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
   
   dataset = LeRobotDataset("waste-e/garbage-picking", root="./lerobot_datasets")
   print(f"Episodes: {dataset.num_episodes}")
   print(f"Frames: {dataset.num_frames}")
   
   # Load a sample
   sample = dataset[0]
   print(sample.keys())
   ```

5. **Push to Hugging Face Hub** (optional)
   ```python
   dataset.push_to_hub(
       tags=["robotics", "waste-e"],
       license="apache-2.0",
   )
   ```

## Tips

- **Use video format**: Set `use_videos=True` for efficient storage
- **Batch encoding**: Set `batch_encoding_size > 1` to encode videos in batches
- **Image quality**: Videos use compression, images are lossless PNG
- **FPS**: Match your robot's control frequency (e.g., 30 Hz)
- **Episode length**: Keep episodes focused on one task (5-30 seconds)
- **Multiple cameras**: Add more camera features in the features dict

## Troubleshooting

### Camera not opening
```bash
# List available cameras
ls /dev/video*

# Try different camera IDs
python collect_real_robot_data.py --camera-id 1
```

### Memory issues
- Reduce `--fps` to save fewer frames
- Use `batch_encoding_size=10` to encode videos in batches
- Enable `image_writer_processes` for parallel image saving

### Import errors
```bash
pip install --upgrade lerobot
pip install opencv-python pillow numpy torch
```

## Next Steps

After collecting data:

1. **Train a policy** using LeRobot's training scripts
2. **Evaluate** the policy on real robot
3. **Iterate** - collect more data in areas where policy fails

## Resources

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [LeRobot Datasets Hub](https://huggingface.co/lerobot)
- [Example Datasets](https://huggingface.co/datasets?search=lerobot)
