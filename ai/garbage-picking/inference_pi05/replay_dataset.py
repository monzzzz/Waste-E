#!/usr/bin/env python3
"""
Action Comparison Tool
Compare recorded actions with what the robot actually does during replay.
Uses LeRobotS01Config for robot control.
"""

import os
# Configure JAX memory allocation before importing JAX-related modules
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import numpy as np
import matplotlib.pyplot as plt
import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def compare_actions(dataset_name, episode_index=0, num_frames=50):
    """
    Compare recorded actions vs actual robot replay.
    
    Args:
        dataset_name: Dataset to load
        episode_index: Episode to analyze
        num_frames: Number of frames to compare
    """
    
    # Load dataset
    print(f"\n📊 Loading dataset: {dataset_name}")
    dataset = LeRobotDataset(dataset_name)
    
    episode_data = dataset.hf_dataset.filter(
        lambda x: x['episode_index'] == episode_index
    )
    
    recorded_actions = np.array([episode_data[i]['action'] for i in range(len(episode_data))])
    recorded_states = np.array([episode_data[i]['observation.state'] for i in range(len(episode_data))])
    
    print(f"✅ Loaded {len(recorded_actions)} frames from dataset")
    
    # Connect to robot
    print(f"\n🤖 Connecting to robot...")
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="my_awesome_follower_arm"
    )
    robot = SO101Follower(robot_config)
    robot.connect()

    print("✅ Robot connected")
    
    # Replay and compare first N frames
    print(f"\n📈 Replaying first {num_frames} frames for comparison...\n")

    
    try:
        for step in range(min(num_frames, len(recorded_actions))):
            t0 = time.perf_counter()
            
            action = recorded_actions[step]
            
            # Send action to robot using LeRobotS01 format
            action_dict = {}
            for i, action_name in enumerate(list(robot.action_features.keys())):
                if i < len(action):
                    action_dict[action_name] = action[i] 
            
            # DEBUG: Print what we're sending
            if  (step%10 == 0):
                print(f"\nDEBUG Frame {step}:")
                print(f"  Raw action array: {action}")
                print(f"  Action dict keys: {list(action_dict.keys())}")
                print(f"  Action dict values: {action_dict}")
            
            robot.send_action(action_dict)

            dt = time.perf_counter() - t0
            time.sleep(max(0, 0.1 - dt)) 

    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")
    
    finally:
        robot.disconnect()


def main():
    
    DATASET_NAME = "Monzzz/garbage-picker-v2-2"
    EPISODE_INDEX = 0
    NUM_FRAMES = 500  # Compare first 50 frames
    
    compare_actions(DATASET_NAME, EPISODE_INDEX, NUM_FRAMES)


if __name__ == "__main__":
    main()
