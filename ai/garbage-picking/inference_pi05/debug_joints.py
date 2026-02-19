#!/usr/bin/env python3
"""
Debug visualizer to understand joint angle behavior
"""
import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def plot_joint_angles_over_time(dataset_name, episode_index=0, max_frame=200):
    """Plot how each joint angle changes over time to understand the motion"""
    print(f"Loading dataset: {dataset_name}")
    dataset = LeRobotDataset(dataset_name)
    
    # Get episode data
    episode_data = dataset.hf_dataset.filter(lambda x: x['episode_index'] == episode_index)
    joint_angles = np.array([episode_data[i]['observation.state'] for i in range(len(episode_data))])
    
    # Limit frames
    max_frame = min(max_frame, len(joint_angles))
    joint_angles = joint_angles[:max_frame]
    
    # Joint names
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    
    # Create plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, name in enumerate(joint_names):
        axes[i].plot(joint_angles[:, i], linewidth=2)
        axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Frame')
        axes[i].set_ylabel('Angle (degrees)')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero')
        
        # Add value annotations at key points
        axes[i].annotate(f'Start: {joint_angles[0, i]:.1f}°', 
                        xy=(0, joint_angles[0, i]), fontsize=8)
        axes[i].annotate(f'End: {joint_angles[-1, i]:.1f}°', 
                        xy=(len(joint_angles)-1, joint_angles[-1, i]), fontsize=8)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('robot_motion_debug.png', dpi=150, bbox_inches='tight')
    print("✅ Saved robot_motion_debug.png")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("JOINT ANGLE STATISTICS")
    print("="*60)
    for i, name in enumerate(joint_names):
        print(f"\n{name}:")
        print(f"  Min: {joint_angles[:, i].min():.2f}°")
        print(f"  Max: {joint_angles[:, i].max():.2f}°")
        print(f"  Range: {joint_angles[:, i].max() - joint_angles[:, i].min():.2f}°")
        print(f"  Start: {joint_angles[0, i]:.2f}°")
        print(f"  End: {joint_angles[-1, i]:.2f}°")

if __name__ == "__main__":
    dataset_name = "Monzzz/garbage-picker-v1-combined-2"
    plot_joint_angles_over_time(dataset_name, episode_index=0, max_frame=200)
