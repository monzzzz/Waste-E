import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load your dataset
dataset = LeRobotDataset("Monzzz/garbage-picker-v1-9")

# Pick an episode
episode_index = 0
episode_data = dataset.hf_dataset.filter(lambda x: x['episode_index'] == episode_index)

# Extract actions and states
actions = np.array([episode_data[i]['action'] for i in range(len(episode_data))])
states = np.array([episode_data[i]['observation.state'] for i in range(len(episode_data))])

# Plot each joint
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
joint_names = ['waist', 'shoulder', 'elbow', 'wrist_angle', 'wrist_rotate', 'gripper']

for i, (ax, name) in enumerate(zip(axes.flat, joint_names)):
    ax.plot(states[:, i], label='state', alpha=0.7)
    ax.plot(actions[:, i], label='action', alpha=0.7)
    ax.set_title(f'{name}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('robot_motion_debug.png')
plt.show()