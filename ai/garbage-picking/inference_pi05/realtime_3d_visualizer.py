#!/usr/bin/env python3
"""
Real-time 3D visualization of SO-100 robot arm.
Connects to the robot and displays joint positions in 3D as they move.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import OmegaConf
from pathlib import Path
from lerobot.robots.so_follower.so_follower import SOFollower


def unlock_motors(robot):
    """Disable torque on all motors to free them."""
    print("🔓 Disabling motor torque...")
    try:
        robot.bus.disable_torque()
        print("✅ All motors are FREE! You can move them manually.")
        return True
    except Exception as e:
        print(f"❌ Error disabling torque: {e}")
        return False


class RealTime3DVisualizer:
    """Real-time 3D visualization of robot arm movements."""
    
    def __init__(self, robot, fps=30):
        self.robot = robot
        self.fps = fps
        self.interval = 1000 / fps  # milliseconds
        
        # Setup figure
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set axis limits
        self.ax.set_xlim([-0.4, 0.4])
        self.ax.set_ylim([-0.4, 0.4])
        self.ax.set_zlim([0, 0.5])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Real-Time Robot Arm - Live View')
        
        # Initialize plot elements with different colors
        self.line_p0_p1, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=8, 
                                        color='red', label='Joint 1')
        self.line_p1_p2, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=8, 
                                        color='orange', label='Motor 2')
        self.line_p2_p3, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=8, 
                                        color='green', label='Motor 3')
        self.line_p3_p4, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=8, 
                                        color='blue', label='Motor 4')
        self.line_p4_p5, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=8, 
                                        color='purple', label='Motor 5 (Wrist)')
        self.trail, = self.ax.plot([], [], [], '-', linewidth=1, alpha=0.3, 
                                   color='cyan', label='End-effector trail')
        
        # Joint angle text display
        self.joint_text = self.ax.text2D(0.02, 0.98, '', transform=self.ax.transAxes,
                                        verticalalignment='top', fontfamily='monospace',
                                        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.ax.legend(loc='upper right')
        
        # Store trail points
        self.trail_x = []
        self.trail_y = []
        self.trail_z = []
        self.max_trail_length = 100
        
        # Store current joint angles
        self.current_angles = None
        
    def forward_kinematics(self, joint_angles):
        """
        Compute 3D positions of robot arm links given joint angles.
        
        Args:
            joint_angles: [shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll, gripper]
        
        Returns:
            positions: List of (x, y, z) positions for each joint
        """
        scale = 1.5
        L1 = 0.05 * scale  # Base to shoulder
        L2 = 0.15 * scale  # Shoulder to elbow
        L3 = 0.15 * scale  # Elbow to wrist
        L4 = 0.08 * scale  # Wrist to end-effector
        
        # Convert degrees to radians if needed
        angles = np.deg2rad(joint_angles) if np.max(np.abs(joint_angles)) > 10 else joint_angles
        shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll, gripper = angles
        
        positions = []
        
        # p0: Base
        p0 = np.array([0, 0, 0])
        positions.append(p0)
        
        # p1: Shoulder pan joint
        p1 = np.array([L1 * np.cos(shoulder_pan), L1 * np.sin(shoulder_pan), 0])
        positions.append(p1)
        
        # p2: Shoulder motor housing
        p2 = np.array([L1 * np.cos(shoulder_pan), L1 * np.sin(shoulder_pan), 0.05])
        positions.append(p2)
        
        # p3: Elbow joint
        p3 = p2 + np.array([
            L2 * np.cos(shoulder_pan) * np.sin(shoulder_lift),
            L2 * np.sin(shoulder_pan) * np.sin(shoulder_lift),
            L2 * np.cos(shoulder_lift)
        ])
        positions.append(p3)
        
        # p4: Wrist joint
        p4 = p3 + np.array([
            L3 * np.cos(shoulder_pan) * np.sin(shoulder_lift + elbow + 90),
            L3 * np.sin(shoulder_pan) * np.sin(shoulder_lift + elbow + 90),
            L3 * np.cos(shoulder_lift + elbow + 90)
        ])
        positions.append(p4)
        
        # p5: End-effector (gripper)
        p5 = p4 + np.array([
            L4 * np.cos(shoulder_pan) * np.sin(shoulder_lift + elbow + 90 + wrist_flex),
            L4 * np.sin(shoulder_pan) * np.sin(shoulder_lift + elbow + 90 + wrist_flex),
            L4 * np.cos(shoulder_lift + elbow + 90 + wrist_flex)
        ])
        positions.append(p5)
        
        return positions
    
    def get_robot_state(self):
        """Get current joint angles from robot."""
        try:
            obs = self.robot.get_observation()
            
            # Extract joint angles (filter out cameras)
            joint_angles = []
            joint_names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 
                          'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
            
            for name in joint_names:
                if name in obs:
                    joint_angles.append(obs[name])
                else:
                    joint_angles.append(0.0)  # Default if not found
            
            self.current_angles = np.array(joint_angles)
            return self.current_angles
            
        except Exception as e:
            print(f"Error reading robot state: {e}")
            return np.array([0, 0, 0, 0, 0, 0])
    
    def init_animation(self):
        """Initialize animation."""
        self.line_p0_p1.set_data([], [])
        self.line_p0_p1.set_3d_properties([])
        self.line_p1_p2.set_data([], [])
        self.line_p1_p2.set_3d_properties([])
        self.line_p2_p3.set_data([], [])
        self.line_p2_p3.set_3d_properties([])
        self.line_p3_p4.set_data([], [])
        self.line_p3_p4.set_3d_properties([])
        self.line_p4_p5.set_data([], [])
        self.line_p4_p5.set_3d_properties([])
        self.trail.set_data([], [])
        self.trail.set_3d_properties([])
        return (self.line_p0_p1, self.line_p1_p2, self.line_p2_p3, 
                self.line_p3_p4, self.line_p4_p5, self.trail, self.joint_text)
    
    def update_animation(self, frame):
        """Update animation with current robot state."""
        # Get current joint angles from robot
        joint_angles = self.get_robot_state()
        
        # Compute forward kinematics
        positions = self.forward_kinematics(joint_angles)
        p0, p1, p2, p3, p4, p5 = positions
        
        # Update arm segments
        self.line_p0_p1.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        self.line_p0_p1.set_3d_properties([p0[2], p1[2]])
        
        self.line_p1_p2.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        self.line_p1_p2.set_3d_properties([p1[2], p2[2]])
        
        self.line_p2_p3.set_data([p2[0], p3[0]], [p2[1], p3[1]])
        self.line_p2_p3.set_3d_properties([p2[2], p3[2]])
        
        self.line_p3_p4.set_data([p3[0], p4[0]], [p3[1], p4[1]])
        self.line_p3_p4.set_3d_properties([p3[2], p4[2]])
        
        self.line_p4_p5.set_data([p4[0], p5[0]], [p4[1], p5[1]])
        self.line_p4_p5.set_3d_properties([p4[2], p5[2]])
        
        # Update trail (end-effector path)
        self.trail_x.append(p5[0])
        self.trail_y.append(p5[1])
        self.trail_z.append(p5[2])
        
        # Keep only recent trail points
        if len(self.trail_x) > self.max_trail_length:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
            self.trail_z.pop(0)
        
        self.trail.set_data(self.trail_x, self.trail_y)
        self.trail.set_3d_properties(self.trail_z)
        
        # Update joint angle text
        joint_names = ['Pan', 'Lift', 'Elbow', 'WristFlex', 'WristRoll', 'Gripper']
        text = 'Joint Angles:\n'
        for i, (name, angle) in enumerate(zip(joint_names, joint_angles)):
            text += f'{name:10s}: {angle:6.1f}°\n'
        self.joint_text.set_text(text)
        
        # Update title with frame rate
        self.ax.set_title(f'Real-Time Robot Arm - Live View (Frame {frame})')
        
        return (self.line_p0_p1, self.line_p1_p2, self.line_p2_p3, 
                self.line_p3_p4, self.line_p4_p5, self.trail, self.joint_text)
    
    def start(self):
        """Start the real-time visualization."""
        print("Starting real-time 3D visualization...")
        print("Close the window to stop.")
        
        # Create animation
        self.anim = FuncAnimation(
            self.fig, 
            self.update_animation,
            init_func=self.init_animation,
            interval=self.interval,
            blit=True,
            cache_frame_data=False
        )
        
        plt.show()


def main():
    """Main function to run real-time visualization."""
    print("=== Real-Time 3D Robot Visualizer ===")
    print("Connecting to robot...")
    
    # Configure robot
    cfg = OmegaConf.create({
        "port": "/dev/ttyACM0",
        "id": 0,
        "baudrate": 115200,
        "calibration_dir": Path("/home/monzzz/.cache/huggingface/lerobot/calibration/robots/so_follower"),
        "use_degrees": True,
        "cameras": {},
        "max_relative_target": None,
        "disable_torque_on_disconnect": True
    })
    
    # Create and connect robot
    robot = SOFollower(cfg)
    robot.connect()
    
    print("✅ Robot connected!")
    
    # Ask if user wants to unlock motors first
    response = input("\n🔓 Unlock motors before starting? (y/n): ").lower()
    if response == 'y':
        unlock_motors(robot)
        input("Press ENTER when ready to start visualization...")
    
    print("Starting visualization in 2 seconds...")
    time.sleep(2)
    
    try:
        # Create visualizer
        visualizer = RealTime3DVisualizer(robot, fps=30)
        
        # Start visualization (blocks until window is closed)
        visualizer.start()
        
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup - automatically unlocks on disconnect
        print("Disconnecting robot...")
        unlock_motors(robot)  # Free motors before disconnect
        robot.disconnect()
        print("✅ Done!")


if __name__ == "__main__":
    main()
