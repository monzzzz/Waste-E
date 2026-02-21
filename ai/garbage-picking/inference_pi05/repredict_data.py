#!/usr/bin/env python3
"""
Offline debug: run your trained Pi0.5 policy on RECORDED dataset frames
and compare predicted actions vs recorded actions.

- Loads a LeRobotDataset from Hugging Face
- Filters a chosen episode_index
- For every Nth frame:
    * builds pi_input from recorded observation (state + images)
    * runs policy.infer(pi_input)
    * prints:
        - recorded action (GT)
        - predicted action[0] (first action in predicted chunk)
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import numpy as np
import cv2
from pathlib import Path
from huggingface_hub import snapshot_download

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi.training import config as train_config
from openpi.training import checkpoints as train_checkpoints
from openpi.policies import policy_config


# -------------------
# SETTINGS
# -------------------
DATASET_NAME = "Monzzz/garbage-picker-v2-combined"
EPISODE_INDEX = 0
FRAME_STRIDE = 10 # print every 10 frames
TASK_DESCRIPTION = "Pick an object in front of you and place in the bin behind you."

# Your trained checkpoint on HF
CHECKPOINT_REPO = "Monzzz/pi05-garbage-picker-v2-5000"
HF_TOKEN = os.environ.get("HF_TOKEN")  # if private

# Model config name you used for training
PI_CONFIG_NAME = "pi05_garbage_picker"


# If you trained absolute actions, your dataset action is likely absolute already.
# We'll just compare raw numbers here.
JOINT_ORDER = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

CAM_ORDER = ["observation.images.front", "observation.images.top", "observation.images.wrist"]


def get_first_existing(example: dict, candidates: list[str]):
    for k in candidates:
        if k in example:
            return k
    return None

def main():
    # -------------------
    # Load model
    # -------------------
    print("Loading Pi0.5 policy...")
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

    # Load and print norm stats
    norm_stats = train_checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    print("📊 Normalization stats:")
    for key, value in norm_stats.items():
        print(f"  {key}:")
        print(f"    mean: {value.mean}")
        print(f"    std: {value.std}")
        if value.q01 is not None:
            print(f"    q01: {value.q01}")
        if value.q99 is not None:
            print(f"    q99: {value.q99}")
    print()

    # -------------------
    # Load dataset
    # -------------------
    print(f"📊 Loading dataset: {DATASET_NAME}")
    dataset = LeRobotDataset(DATASET_NAME)

    episode_indices = [
        i for i, ep in enumerate(dataset.hf_dataset["episode_index"]) if ep == EPISODE_INDEX
    ]
    n = len(episode_indices)
    print(f"✅ Episode {EPISODE_INDEX} frames: {n}")

    # Peek keys once so you know what's inside
    if n == 0:
        raise ValueError(f"No frames found for episode_index={EPISODE_INDEX}")
    sample = dataset[episode_indices[0]]
    print("\nSample keys:", list(sample.keys()))
    
    # Debug: print nested structure
    if "observation.images" in sample:
        print("observation.images keys:", list(sample["observation.images"].keys()))

    # Try to find likely action/state/image keys
    action_key = get_first_existing(sample, ["action", "actions"])
    state_key = get_first_existing(sample, ["observation.state", "state", "observation/state"])
    # images in LeRobot datasets are often stored as:
    # - 'front', 'top', 'wrist' directly
    # - or 'observation.images' with nested camera keys
    # - or 'observation.images.front' style (less common)
    print(f"\nDetected action_key={action_key}, state_key={state_key}")

    if action_key is None:
        raise KeyError("Could not find action key in dataset example. Check sample keys above.")
    if state_key is None:
        raise KeyError("Could not find state key in dataset example. Check sample keys above.")
    
    # Check for image keys in the full dataset (with cameras)
    print("\nChecking for image columns in full dataset...")
    print(f"Full dataset column names: {dataset.hf_dataset.column_names}")

    # -------------------
    # Run inference on recorded frames
    # -------------------
    print("\n--- Running inference on recorded frames ---\n")

    # Track previous state for computing deltas
    prev_state = None

    for local_idx in range(0, n, FRAME_STRIDE):
        dataset_idx = episode_indices[local_idx]
        try:
            item = dataset[dataset_idx]
        except Exception as e:
            print(f"[frame {local_idx}] ⚠️ Error loading item at index {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Debug: print available keys on first frame
        if local_idx == 0:
            print("Item keys from dataset:", list(item.keys()))

        current_state = np.array(item.get("observation.state", item.get("state")), dtype=np.float32)

        # Build input in the same format as the training pipeline expects.
        # We pass raw dataset keys and let the policy's repack + data transforms handle the rest.
        pi_input = {
            "observation.state": current_state,
            "observation.images.front": item.get("observation.images.front"),
            "observation.images.top": item.get("observation.images.top"),
            "observation.images.wrist": item.get("observation.images.wrist"),
            "action": item[action_key],
            "prompt": TASK_DESCRIPTION,
        }

        # Compute ground-truth action: deltas for first 5 joints, absolute for gripper
        gt_action_raw = np.array(item[action_key], dtype=np.float32)
        gt_action = gt_action_raw.copy()
        # First 5 dims: delta = action - current_state (matching DeltaActions transform)
        gt_action[:5] = gt_action_raw[:5] - current_state[:5]
        # Last dim (gripper): keep absolute
        gt_action[5:] = gt_action_raw[5:]

        # Predict
        try:
            out = policy.infer(pi_input)

            pred_actions = np.array(out["actions"], dtype=np.float32)
            
            pred0 = pred_actions[0]  

            # Print in your nice "frame_XX = [...]" format
            print(f"FRAME {local_idx}")
            print("  GT  :", gt_action.tolist())
            print("  PRED:", pred0[: len(gt_action)].tolist())
            print("  (pred chunk len:", len(pred_actions), ")")
            print("-" * 60)
        except Exception as e:
            print(f"[frame {local_idx}] ❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()