import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def forward_kinematics(joint_angles):
    """
    Compute 3D positions of robot arm links given joint angles.
    Assumes a simple 6-DOF arm: shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll, gripper
    
    Args:
        joint_angles: [shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll, gripper]
    
    Returns:
        positions: List of (x, y, z) positions for each joint
    """
    scale = 1.5
    # Link lengths (adjust these based on your actual robot dimensions)
    L1 = 0.05 * scale  # Base to shoulder
    L2 = 0.15 * scale  # Shoulder to elbow
    L3 = 0.15 * scale  # Elbow to wrist
    L4 = 0.08 * scale  # Wrist to end-effector
    
    # Convert degrees to radians if needed
    angles = np.deg2rad(joint_angles) if np.max(np.abs(joint_angles)) > 10 else joint_angles
    
    shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll, gripper = angles
    
    # YOUR EXACT POSITION CALCULATIONS
    positions = []

    # motor1 (p0)
    p0 = np.array([0, 0, 0])  
    positions.append(p0)
    
    # joint 1 (p1)
    p1 = np.array([L1 * np.cos(shoulder_pan), L1 * np.sin(shoulder_pan), 0])
    positions.append(p1)

    # motor2 (p2)
    p2 = np.array([L1 * np.cos(shoulder_pan), L1 * np.sin(shoulder_pan), 0.05])
    positions.append(p2)

    # motor3 (p3)
    p3 = p2 + np.array([
        L2 * np.cos(shoulder_pan) * np.sin(shoulder_lift),
        L2 * np.sin(shoulder_pan) * np.sin(shoulder_lift),
        L2 * np.cos(shoulder_lift)
    ])
    positions.append(p3)

    # motor4 (p4) - CORRECTED with +90 offset
    p4 = p3 + np.array([
        L3 * np.cos(shoulder_pan) * np.sin(shoulder_lift + elbow + np.deg2rad(90)),
        L3 * np.sin(shoulder_pan) * np.sin(shoulder_lift + elbow + np.deg2rad(90)),
        L3 * np.cos(shoulder_lift + elbow + np.deg2rad(90))
    ])
    positions.append(p4)

    # motor5 (p5) - CORRECTED with +90 offset
    p5 = p4 + np.array([
        L4 * np.cos(shoulder_pan) * np.sin(shoulder_lift + elbow + np.deg2rad(90) + wrist_flex),
        L4 * np.sin(shoulder_pan) * np.sin(shoulder_lift + elbow + np.deg2rad(90) + wrist_flex),
        L4 * np.cos(shoulder_lift + elbow + np.deg2rad(90) + wrist_flex)
    ])
    positions.append(p5)
    
    return positions


def create_3d_animation(dataset_name, episode_index=0, use_actions=False, save_path='robot_3d_animation.mp4'):
    """
    Create 3D animation of robot arm movements.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        episode_index: Which episode to visualize
        use_actions: If True, visualize actions instead of states
        save_path: Path to save the animation video
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = LeRobotDataset(dataset_name)
    
    # Get episode data
    episode_data = dataset.hf_dataset.filter(lambda x: x['episode_index'] == episode_index)
    
    # Extract joint angles
    if use_actions:
        joint_angles = np.array([episode_data[i]['action'] for i in range(len(episode_data))])
        title_suffix = "(Actions)"
    else:
        joint_angles = np.array([episode_data[i]['observation.state'] for i in range(len(episode_data))])
        title_suffix = "(States)"
    
    print(f"Episode {episode_index}: {len(joint_angles)} frames")
    print(f"Joint angle shape: {joint_angles.shape}")
    
    # Setup the figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 0.5])
    ax.set_xlabel('X (m')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Robot Arm Animation - Episode {episode_index} {title_suffix}')
    
    # Initialize plot elements with different colors for each segment
    line_p0_p1, = ax.plot([], [], [], 'o-', linewidth=3, markersize=8, color='red', label='Joint 1')
    line_p1_p2, = ax.plot([], [], [], 'o-', linewidth=3, markersize=8, color='orange', label='Motor 2')
    line_p2_p3, = ax.plot([], [], [], 'o-', linewidth=3, markersize=8, color='green', label='Motor 3')
    line_p3_p4, = ax.plot([], [], [], 'o-', linewidth=3, markersize=8, color='blue', label='Motor 4')
    line_p4_p5, = ax.plot([], [], [], 'o-', linewidth=3, markersize=8, color='purple', label='Motor 5 (Wrist)')
    trail, = ax.plot([], [], [], '-', linewidth=1, alpha=0.3, color='cyan', label='End-effector trail')
    
    ax.legend()
    
    # Store trail points
    trail_x, trail_y, trail_z = [], [], []
    max_trail_length = 50
    
    def init():
        """Initialize animation"""
        line_p0_p1.set_data([], [])
        line_p0_p1.set_3d_properties([])
        line_p1_p2.set_data([], [])
        line_p1_p2.set_3d_properties([])
        line_p2_p3.set_data([], [])
        line_p2_p3.set_3d_properties([])
        line_p3_p4.set_data([], [])
        line_p3_p4.set_3d_properties([])
        line_p4_p5.set_data([], [])
        line_p4_p5.set_3d_properties([])
        trail.set_data([], [])
        trail.set_3d_properties([])
        return line_p0_p1, line_p1_p2, line_p2_p3, line_p3_p4, line_p4_p5, trail
    
    def update(frame):
        """Update animation for each frame"""
        # Get joint positions for this frame
        positions = forward_kinematics(joint_angles[frame])
        
        # Unpack positions (now 6: p0, p1, p2, p3, p4, p5)
        p0, p1, p2, p3, p4, p5 = positions
        
        # Update each arm segment with different colors
        # Base to joint 1 - Red
        line_p0_p1.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line_p0_p1.set_3d_properties([p0[2], p1[2]])
        
        # Joint 1 to motor 2 - Orange
        line_p1_p2.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        line_p1_p2.set_3d_properties([p1[2], p2[2]])
        
        # Motor 2 to motor 3 - Green
        line_p2_p3.set_data([p2[0], p3[0]], [p2[1], p3[1]])
        line_p2_p3.set_3d_properties([p2[2], p3[2]])
        
        # Motor 3 to motor 4 - Blue
        line_p3_p4.set_data([p3[0], p4[0]], [p3[1], p4[1]])
        line_p3_p4.set_3d_properties([p3[2], p4[2]])
        
        # Motor 4 to motor 5 (wrist) - Purple
        line_p4_p5.set_data([p4[0], p5[0]], [p4[1], p5[1]])
        line_p4_p5.set_3d_properties([p4[2], p5[2]])
        
        # Update trail (end-effector path) - use p5 as end effector
        trail_x.append(p5[0])
        trail_y.append(p5[1])
        trail_z.append(p5[2])
        
        # Keep only recent trail points
        if len(trail_x) > max_trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
            trail_z.pop(0)
        
        trail.set_data(trail_x, trail_y)
        trail.set_3d_properties(trail_z)
        
        # Update title with frame info
        ax.set_title(f'Robot Arm Animation - Episode {episode_index} {title_suffix}\nFrame {frame}/{len(joint_angles)}')
        
        return line_p0_p1, line_p1_p2, line_p2_p3, line_p3_p4, line_p4_p5, trail
    
    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, update, init_func=init, frames=len(joint_angles), 
                        interval=100, blit=True, repeat=True)
    
    # Save animation
    print(f"Saving animation to {save_path}...")
    anim.save(save_path, writer='ffmpeg', fps=10, dpi=100)
    print(f"✅ Animation saved!")
    
    # Also show it
    plt.show()
    
    return anim


def create_static_comparison(dataset_name, episode_index=0, num_frames=15, max_frame=500):
    """
    Create a static figure showing multiple frames of the 3D robot pose.
    Now creates multiple images if there are too many frames to fit on screen.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        episode_index: Which episode to visualize
        num_frames: Number of frames to show
        max_frame: Last frame to show (default 500)
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = LeRobotDataset(dataset_name)
    
    # Get episode data
    episode_data = dataset.hf_dataset.filter(lambda x: x['episode_index'] == episode_index)
    joint_angles = np.array([episode_data[i]['observation.state'] for i in range(len(episode_data))])
    
    # Limit to max_frame
    max_frame = min(max_frame, len(joint_angles) - 1)
    
    # Select evenly spaced frames from 0 to max_frame
    frame_indices = np.linspace(0, max_frame, num_frames, dtype=int)
    
    print(f"Visualizing frames: {frame_indices[0]} to {frame_indices[-1]} (total: {num_frames} frames)")
    
    # Split into multiple images (6 frames per image for better visibility)
    frames_per_image = 6
    num_images = (num_frames + frames_per_image - 1) // frames_per_image
    
    for img_idx in range(num_images):
        start_idx = img_idx * frames_per_image
        end_idx = min(start_idx + frames_per_image, num_frames)
        current_frames = frame_indices[start_idx:end_idx]
        
        # Create subplots - 2 rows x 3 columns for 6 frames
        fig = plt.figure(figsize=(18, 12))
        
        for idx, frame_idx in enumerate(current_frames):
            ax = fig.add_subplot(2, 3, idx+1, projection='3d')
            
            # Get positions (now 6: p0, p1, p2, p3, p4, p5)
            positions = forward_kinematics(joint_angles[frame_idx])
            p0, p1, p2, p3, p4, p5 = positions
            
            # Plot arm with different colors for each segment
            # Base to joint 1 (motor 1) - Red
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 
                   'o-', linewidth=3, markersize=8, color='red')
            
            # Joint 1 to motor 2 - Orange
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   'o-', linewidth=3, markersize=8, color='orange')
            
            # Motor 2 to motor 3 (shoulder) - Green
            ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], 
                   'o-', linewidth=3, markersize=8, color='green')
            
            # Motor 3 to motor 4 (elbow) - Blue
            ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], 
                   'o-', linewidth=3, markersize=8, color='blue')
            
            # Motor 4 to motor 5 (wrist) - Purple
            ax.plot([p4[0], p5[0]], [p4[1], p5[1]], [p4[2], p5[2]], 
                   'o-', linewidth=3, markersize=8, color='purple')
            
            # Set axis properties
            ax.set_xlim([-0.4, 0.4])
            ax.set_ylim([-0.4, 0.4])
            ax.set_zlim([0, 0.5])
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Frame {frame_idx}', fontsize=12, fontweight='bold')
            ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        output_filename = f'robot_3d_frames_part{img_idx+1}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved {output_filename} (frames {current_frames[0]} to {current_frames[-1]})")
        plt.close()
    
    print(f"✅ Created {num_images} images with {frames_per_image} frames each")


if __name__ == "__main__":
    # Example usage
    dataset_name = "Monzzz/garbage-picker-v2-combined"
    
    print("=== Robot Arm 3D Visualizer ===")
    print("1. Creating static comparison...")
    create_static_comparison(dataset_name, episode_index=0, num_frames=15, max_frame=400)  # Changed to frame 400
    
    print("\n2. Creating 3D animation...")
    create_3d_animation(dataset_name, episode_index=0, use_actions=False, 
                       save_path='robot_3d_animation.mp4')
    
    print("\n✅ All visualizations complete!")
