#!/usr/bin/env python3
"""
Visualize predicted vs ground-truth robot movements using URDF model.
Shows both GT and predicted trajectories in 3D animation with accurate robot geometry.
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from huggingface_hub import snapshot_download
import urdfpy

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi.training import config as train_config
from openpi.training import checkpoints as train_checkpoints
from openpi.policies import policy_config


# -------------------
# SETTINGS
# -------------------
DATASET_NAME = "Monzzz/garbage-picker-v2-combined"
EPISODE_INDEX = 0
FRAME_STRIDE = 1  # Use every frame for smooth animation
TASK_DESCRIPTION = "Pick an object in front of you and place in the bin behind you."

# Your trained checkpoint on HF
CHECKPOINT_REPO = "Monzzz/pi05-garbage-picker-v2-5000"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Model config name you used for training
PI_CONFIG_NAME = "pi05_garbage_picker"

# Animation settings
MAX_FRAMES = 200  # Limit frames for faster rendering
SAVE_VIDEO = True
VIDEO_PATH = "prediction_comparison_urdf.mp4"
URDF_PATH = "so101.urdf"  # Downloaded URDF file


def get_joint_positions_from_urdf(urdf_robot, joint_angles):
    """
    Get link positions using URDF forward kinematics.
    
    Args:
        urdf_robot: URDF robot object
        joint_angles: [shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll, gripper]
    
    Returns:
        positions: List of (x, y, z) positions for each link
    """
    # Map joint angles to URDF joint names
    # Adjust these names based on actual URDF joint names
    joint_names = [
        'motor1',  # shoulder_pan
        'motor2',  # shoulder_lift  
        'motor3',  # elbow
        'motor4',  # wrist_flex
        'motor5',  # wrist_roll
        'motor6',  # gripper
    ]
    
    # Create configuration dictionary
    cfg = {}
    for i, name in enumerate(joint_names):
        if i < len(joint_angles):
            cfg[name] = joint_angles[i]
    
    # Get forward kinematics
    fk = urdf_robot.link_fk(cfg=cfg)
    
    # Extract positions from transform matrices
    positions = []
    for link_name in urdf_robot.links:
        if link_name.name in fk:
            transform = fk[link_name.name]
            pos = transform[:3, 3]  # Extract translation
            positions.append(pos)
    
    return positions


def get_first_existing(example: dict, candidates: list[str]):
    for k in candidates:
        if k in example:
            return k
    return None


def main():
    # -------------------
    # Load URDF
    # -------------------
    print(f"Loading URDF from {URDF_PATH}...")
    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF file not found: {URDF_PATH}. Make sure to download it first.")
    
    urdf_robot = urdfpy.URDF.load(URDF_PATH)
    print(f"✅ URDF loaded: {len(urdf_robot.links)} links, {len(urdf_robot.joints)} joints")
    print(f"   Links: {[link.name for link in urdf_robot.links]}")
    print(f"   Joints: {[joint.name for joint in urdf_robot.joints]}")

    # -------------------
    # Load model
    # -------------------
    print("\nLoading Pi0.5 policy...")
    cfg = train_config.get_config(PI_CONFIG_NAME)

    print("Downloading checkpoint...")
    checkpoint_dir = snapshot_download(
        repo_id=CHECKPOINT_REPO,
        repo_type="model",
        token=HF_TOKEN,
    )
    checkpoint_dir = Path(checkpoint_dir)

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    policy = policy_config.create_trained_policy(
        cfg,
        checkpoint_dir,
        repack_transforms=data_config.repack_transforms,
    )
    print("✅ Policy loaded\n")

    # -------------------
    # Load dataset
    # -------------------
    print(f"📊 Loading dataset: {DATASET_NAME}")
    dataset = LeRobotDataset(DATASET_NAME)

    episode_indices = [
        i for i, ep in enumerate(dataset.hf_dataset["episode_index"]) if ep == EPISODE_INDEX
    ]
    n = min(len(episode_indices), MAX_FRAMES)
    print(f"✅ Episode {EPISODE_INDEX}: {len(episode_indices)} frames (using first {n})")

    sample = dataset[episode_indices[0]]
    action_key = get_first_existing(sample, ["action", "actions"])
    state_key = get_first_existing(sample, ["observation.state", "state", "observation/state"])

    if action_key is None or state_key is None:
        raise KeyError("Could not find action or state keys")

    # -------------------
    # Collect predictions and GT
    # -------------------
    print("\n--- Collecting predictions ---")
    gt_positions_list = []
    pred_positions_list = []
    
    cumulative_state = None  # Track cumulative state for predicted trajectory

    for local_idx in range(0, n, FRAME_STRIDE):
        dataset_idx = episode_indices[local_idx]
        item = dataset[dataset_idx]
        
        current_state = np.array(item.get("observation.state", item.get("state")), dtype=np.float32)
        
        # Ground truth action (convert to absolute)
        gt_action_raw = np.array(item[action_key], dtype=np.float32)
        gt_action_abs = gt_action_raw.copy()
        
        # Build input
        pi_input = {
            "observation.state": current_state,
            "observation.images.front": item.get("observation.images.front"),
            "observation.images.top": item.get("observation.images.top"),
            "observation.images.wrist": item.get("observation.images.wrist"),
            "action": item[action_key],
            "prompt": TASK_DESCRIPTION,
        }

        # Predict
        try:
            out = policy.infer(pi_input)
            pred_actions = np.array(out["actions"], dtype=np.float32)
            pred0 = pred_actions[0]  # First action in chunk
            
            # Initialize cumulative state on first frame
            if cumulative_state is None:
                cumulative_state = current_state.copy()
            
            # Apply predicted deltas to get absolute predicted position
            # First 5: add deltas to cumulative state
            cumulative_state[:5] += pred0[:5]
            # Last (gripper): use absolute value
            cumulative_state[5] = pred0[5]
            
            # Compute FK using URDF for both
            gt_positions = get_joint_positions_from_urdf(urdf_robot, gt_action_abs)
            pred_positions = get_joint_positions_from_urdf(urdf_robot, cumulative_state)
            
            gt_positions_list.append(gt_positions)
            pred_positions_list.append(pred_positions)
            
            if local_idx % 10 == 0:
                print(f"  Frame {local_idx}/{n}")
                
        except Exception as e:
            print(f"[frame {local_idx}] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✅ Collected {len(gt_positions_list)} frames")

    # -------------------
    # Create animation
    # -------------------
    print("\n--- Creating animation ---")
    
    fig = plt.figure(figsize=(16, 8))
    
    # GT subplot
    ax_gt = fig.add_subplot(121, projection='3d')
    ax_gt.set_xlim([-0.4, 0.4])
    ax_gt.set_ylim([-0.4, 0.4])
    ax_gt.set_zlim([0, 0.5])
    ax_gt.set_xlabel('X (m)')
    ax_gt.set_ylabel('Y (m)')
    ax_gt.set_zlabel('Z (m)')
    ax_gt.set_title('Ground Truth', fontsize=14, fontweight='bold')
    
    # Predicted subplot
    ax_pred = fig.add_subplot(122, projection='3d')
    ax_pred.set_xlim([-0.4, 0.4])
    ax_pred.set_ylim([-0.4, 0.4])
    ax_pred.set_zlim([0, 0.5])
    ax_pred.set_xlabel('X (m)')
    ax_pred.set_ylabel('Y (m)')
    ax_pred.set_zlabel('Z (m)')
    ax_pred.set_title('Predicted', fontsize=14, fontweight='bold')
    
    # Initialize lines for GT (connect all links)
    gt_line, = ax_gt.plot([], [], [], 'o-', linewidth=3, markersize=8, color='blue')
    gt_trail, = ax_gt.plot([], [], [], '-', linewidth=1, alpha=0.3, color='cyan')
    
    # Initialize lines for Predicted
    pred_line, = ax_pred.plot([], [], [], 'o-', linewidth=3, markersize=8, color='red')
    pred_trail, = ax_pred.plot([], [], [], '-', linewidth=1, alpha=0.3, color='magenta')
    
    # Trail storage
    gt_trail_x, gt_trail_y, gt_trail_z = [], [], []
    pred_trail_x, pred_trail_y, pred_trail_z = [], [], []
    max_trail = 30
    
    def init():
        gt_line.set_data([], [])
        gt_line.set_3d_properties([])
        pred_line.set_data([], [])
        pred_line.set_3d_properties([])
        gt_trail.set_data([], [])
        gt_trail.set_3d_properties([])
        pred_trail.set_data([], [])
        pred_trail.set_3d_properties([])
        return gt_line, pred_line, gt_trail, pred_trail
    
    def update(frame):
        if frame >= len(gt_positions_list):
            return gt_line, pred_line, gt_trail, pred_trail
        
        # Update GT
        gt_pos = gt_positions_list[frame]
        if len(gt_pos) > 0:
            gt_pos_array = np.array(gt_pos)
            gt_line.set_data(gt_pos_array[:, 0], gt_pos_array[:, 1])
            gt_line.set_3d_properties(gt_pos_array[:, 2])
            
            # Trail from last link
            gt_trail_x.append(gt_pos[-1][0])
            gt_trail_y.append(gt_pos[-1][1])
            gt_trail_z.append(gt_pos[-1][2])
            if len(gt_trail_x) > max_trail:
                gt_trail_x.pop(0)
                gt_trail_y.pop(0)
                gt_trail_z.pop(0)
            gt_trail.set_data(gt_trail_x, gt_trail_y)
            gt_trail.set_3d_properties(gt_trail_z)
        
        # Update Predicted
        pred_pos = pred_positions_list[frame]
        if len(pred_pos) > 0:
            pred_pos_array = np.array(pred_pos)
            pred_line.set_data(pred_pos_array[:, 0], pred_pos_array[:, 1])
            pred_line.set_3d_properties(pred_pos_array[:, 2])
            
            # Trail from last link
            pred_trail_x.append(pred_pos[-1][0])
            pred_trail_y.append(pred_pos[-1][1])
            pred_trail_z.append(pred_pos[-1][2])
            if len(pred_trail_x) > max_trail:
                pred_trail_x.pop(0)
                pred_trail_y.pop(0)
                pred_trail_z.pop(0)
            pred_trail.set_data(pred_trail_x, pred_trail_y)
            pred_trail.set_3d_properties(pred_trail_z)
        
        fig.suptitle(f'Frame {frame}/{len(gt_positions_list)}', fontsize=16)
        
        return gt_line, pred_line, gt_trail, pred_trail
    
    anim = FuncAnimation(fig, update, init_func=init, frames=len(gt_positions_list),
                        interval=100, blit=True, repeat=True)
    
    if SAVE_VIDEO:
        print(f"Saving animation to {VIDEO_PATH}...")
        anim.save(VIDEO_PATH, writer='ffmpeg', fps=10, dpi=100)
        print(f"✅ Animation saved to {VIDEO_PATH}")
    
    plt.show()


if __name__ == "__main__":
    main()
