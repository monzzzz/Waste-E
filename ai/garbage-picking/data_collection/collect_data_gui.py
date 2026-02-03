#!/usr/bin/env python

"""
GUI-based LeRobot Data Collection Interface

Interactive data collection with visual controls:
- Start/Stop recording episodes
- Next Episode button
- Done button to finalize
- Live camera preview
- Episode counter and status

Usage:
    python collect_data_gui.py --camera-ids 0 1 2
"""

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
# import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# try:
#     from lerobot.src.lerobot.datasets.lerobot_dataset import LeRobotDataset
# except ImportError:whic
#     print("Error: LeRobot library not found. Install it with: pip install lerobot")
#     exit(1)


class RobotInterface:
    """Interface to connect with robot hardware."""
    
    def __init__(self, camera_ids={'top': 0, 'front': 1, 'wrist': 2}):
        self.cameras = {}
        for cam_name, cam_id in camera_ids.items():
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                print(f"Warning: Failed to open camera '{cam_name}' (ID {cam_id})")
                continue
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.cameras[cam_name] = cap
            print(f"Camera '{cam_name}' (ID {cam_id}) initialized")
        
        if not self.cameras:
            raise RuntimeError("No cameras were successfully initialized")
    
    def get_camera_observations(self):
        """Capture images from all cameras."""
        images = {}
        for cam_name, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images[cam_name] = Image.fromarray(frame_rgb)
        
        return images
    
    def get_robot_state(self):
        """Get current robot state."""
        # TODO: Replace with actual sensor readings
        state = {
            "motor_positions": np.random.randn(6).astype(np.float32),
            "motor_velocities": np.random.randn(6).astype(np.float32),
        }
        return state
    
    def get_action(self):
        """Get robot action from teleoperation."""
        # TODO: Implement actual teleoperation interface
        action = np.random.randn(6).astype(np.float32)
        return action
    
    def execute_action(self, action):
        """Execute action on robot."""
        # TODO: Send commands to motors
        pass
    
    def close(self):
        """Release resources."""
        for cam_name, cap in self.cameras.items():
            cap.release()


class DataCollectionGUI:
    def __init__(self, root, robot, dataset, fps=30, task="pick_garbage"):
        self.root = root
        self.robot = robot
        self.dataset = dataset
        self.fps = fps
        self.task = task
        
        self.is_recording = False
        self.is_running = True
        self.current_episode = 0
        self.frame_count = 0
        self.episode_start_time = None
        
        self.frame_queue = Queue(maxsize=10)
        
        self.setup_ui()
        
        # Start camera preview thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.recording_loop, daemon=True)
        self.recording_thread.start()
    
    def setup_ui(self):
        """Setup the GUI interface."""
        self.root.title("LeRobot Data Collection")
        self.root.geometry("1400x800")
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Camera previews
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Previews", padding="10")
        camera_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Camera labels
        self.camera_labels = {}
        cam_names = ['top', 'front', 'wrist']
        for idx, cam_name in enumerate(cam_names):
            label = ttk.Label(camera_frame, text=f"{cam_name.title()} Camera")
            label.grid(row=idx*2, column=0, pady=5)
            
            canvas = tk.Label(camera_frame, bg='black', width=320, height=240)
            canvas.grid(row=idx*2+1, column=0, pady=5)
            self.camera_labels[cam_name] = canvas
        
        # Right panel - Controls and status
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Status display
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding="10")
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(status_frame, text="Dataset:").grid(row=0, column=0, sticky=tk.W)
        self.dataset_label = ttk.Label(status_frame, text=self.dataset.repo_id, font=('Arial', 10, 'bold'))
        self.dataset_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(status_frame, text="Episode:").grid(row=1, column=0, sticky=tk.W)
        self.episode_label = ttk.Label(status_frame, text="0", font=('Arial', 16, 'bold'))
        self.episode_label.grid(row=1, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(status_frame, text="Frames:").grid(row=2, column=0, sticky=tk.W)
        self.frame_label = ttk.Label(status_frame, text="0", font=('Arial', 14))
        self.frame_label.grid(row=2, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(status_frame, text="Duration:").grid(row=3, column=0, sticky=tk.W)
        self.duration_label = ttk.Label(status_frame, text="0.0s", font=('Arial', 14))
        self.duration_label.grid(row=3, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(status_frame, text="Recording:").grid(row=4, column=0, sticky=tk.W)
        self.recording_label = ttk.Label(status_frame, text="●", font=('Arial', 20), foreground='red')
        self.recording_label.grid(row=4, column=1, sticky=tk.W, padx=10)
        
        # Task input
        task_frame = ttk.LabelFrame(control_frame, text="Task Description", padding="10")
        task_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.task_entry = ttk.Entry(task_frame, width=40)
        self.task_entry.insert(0, self.task)
        self.task_entry.grid(row=0, column=0, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame, padding="10")
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.start_button = ttk.Button(
            button_frame, 
            text="Start Recording", 
            command=self.start_recording,
            style='Start.TButton'
        )
        self.start_button.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.stop_button = ttk.Button(
            button_frame, 
            text="Stop Recording", 
            command=self.stop_recording,
            state=tk.DISABLED,
            style='Stop.TButton'
        )
        self.stop_button.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.next_button = ttk.Button(
            button_frame, 
            text="Next Episode →", 
            command=self.next_episode,
            state=tk.DISABLED,
            style='Next.TButton'
        )
        self.next_button.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.done_button = ttk.Button(
            button_frame, 
            text="✓ Done & Finalize", 
            command=self.finalize_dataset,
            style='Done.TButton'
        )
        self.done_button.grid(row=3, column=0, pady=20, sticky=(tk.W, tk.E))
        
        # Log/Info area
        log_frame = ttk.LabelFrame(control_frame, text="Log", padding="10")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.log_text = tk.Text(log_frame, height=15, width=50, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Configure button styles
        style = ttk.Style()
        style.configure('Start.TButton', font=('Arial', 12, 'bold'), foreground='green')
        style.configure('Stop.TButton', font=('Arial', 12, 'bold'), foreground='red')
        style.configure('Next.TButton', font=('Arial', 12, 'bold'), foreground='blue')
        style.configure('Done.TButton', font=('Arial', 14, 'bold'), foreground='darkgreen')
        
        self.log("Data collection interface ready")
        self.log(f"Dataset: {self.dataset.repo_id}")
        self.log(f"Recording at {self.fps} FPS")
        self.log("Remember: Click 'Done & Finalize' before closing to save properly")
    
    def log(self, message):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        print(f"[{timestamp}] {message}")
    
    def camera_loop(self):
        """Update camera previews."""
        while self.is_running:
            try:
                images = self.robot.get_camera_observations()
                
                for cam_name, img in images.items():
                    if cam_name in self.camera_labels:
                        # Resize for preview
                        img_resized = img.resize((320, 240), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img_resized)
                        
                        # Update in main thread
                        self.root.after(0, self.update_camera_label, cam_name, photo)
                
                time.sleep(1/30)  # 30 FPS preview
            except Exception as e:
                self.log(f"Camera error: {e}")
                time.sleep(0.1)
    
    def update_camera_label(self, cam_name, photo):
        """Update camera label (must run in main thread)."""
        label = self.camera_labels.get(cam_name)
        if label:
            label.config(image=photo)
            label.image = photo  # Keep reference
    
    def recording_loop(self):
        """Handle frame recording in background thread."""
        while self.is_running:
            if self.is_recording:
                try:
                    # Get observations
                    camera_images = self.robot.get_camera_observations()
                    robot_state = self.robot.get_robot_state()
                    action = self.robot.get_action()
                    
                    # Execute action
                    self.robot.execute_action(action)
                    
                    # Create observation state
                    obs_state = np.concatenate([
                        robot_state["motor_positions"],
                        robot_state["motor_velocities"],
                    ]).astype(np.float32)
                    
                    # Create frame
                    frame = {
                        "observation.images.top": camera_images.get('top'),
                        "observation.images.front": camera_images.get('front'),
                        "observation.images.wrist": camera_images.get('wrist'),
                        "observation.state": obs_state,
                        "action": action,
                        "task": self.task_entry.get(),
                    }
                    
                    # Add to dataset
                    self.dataset.add_frame(frame)
                    self.frame_count += 1
                    
                    # Update UI
                    elapsed = time.time() - self.episode_start_time
                    self.root.after(0, self.update_recording_status, self.frame_count, elapsed)
                    
                    # Maintain framerate
                    time.sleep(1/self.fps)
                    
                except Exception as e:
                    self.log(f"Recording error: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    def update_recording_status(self, frames, duration):
        """Update recording status in UI."""
        self.frame_label.config(text=str(frames))
        self.duration_label.config(text=f"{duration:.1f}s")
    
    def start_recording(self):
        """Start recording episode."""
        self.is_recording = True
        self.frame_count = 0
        self.episode_start_time = time.time()
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.DISABLED)
        self.recording_label.config(foreground='green', text='● REC')
        
        self.log(f"Started recording Episode {self.current_episode}")
    
    def stop_recording(self):
        """Stop recording episode."""
        self.is_recording = False
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.NORMAL)
        self.recording_label.config(foreground='orange', text='● PAUSED')
        
        elapsed = time.time() - self.episode_start_time
        self.log(f"Stopped recording. {self.frame_count} frames in {elapsed:.1f}s")
    
    def next_episode(self):
        """Save current episode and prepare for next."""
        if self.frame_count == 0:
            messagebox.showwarning("No Data", "No frames recorded in this episode.")
            return
        
        try:
            self.log(f"Saving Episode {self.current_episode}...")
            self.dataset.save_episode()
            
            self.current_episode += 1
            self.frame_count = 0
            
            self.episode_label.config(text=str(self.current_episode))
            self.frame_label.config(text="0")
            self.duration_label.config(text="0.0s")
            self.recording_label.config(foreground='red', text='●')
            
            self.next_button.config(state=tk.DISABLED)
            
            self.log(f"✓ Episode saved. Total episodes: {self.dataset.meta.total_episodes}")
            self.log(f"Ready for Episode {self.current_episode}")
            
        except Exception as e:
            self.log(f"Error saving episode: {e}")
            messagebox.showerror("Save Error", f"Failed to save episode: {e}")
    
    def finalize_dataset(self):
        """Finalize dataset and close."""
        if self.is_recording:
            messagebox.showwarning("Still Recording", "Please stop recording before finalizing.")
            return
        
        response = messagebox.askyesno(
            "Finalize Dataset",
            f"Finalize dataset with {self.dataset.meta.total_episodes} episodes?\n\n"
            "This will close the application."
        )
        
        if response:
            try:
                # Save current episode if there are frames
                if self.frame_count > 0:
                    self.log("Saving current episode...")
                    self.dataset.save_episode()
                
                self.log("Finalizing dataset...")
                self.dataset.finalize()
                
                self.log("✓ Dataset finalized successfully!")
                self.log(f"Total episodes: {self.dataset.meta.total_episodes}")
                self.log(f"Total frames: {self.dataset.meta.total_frames}")
                self.log(f"Location: {self.dataset.root}")
                
                messagebox.showinfo(
                    "Success",
                    f"Dataset finalized!\n\n"
                    f"Episodes: {self.dataset.meta.total_episodes}\n"
                    f"Frames: {self.dataset.meta.total_frames}\n"
                    f"Location: {self.dataset.root}"
                )
                
                self.cleanup()
                
            except Exception as e:
                self.log(f"Error finalizing: {e}")
                messagebox.showerror("Finalization Error", f"Failed to finalize: {e}")
    
    def cleanup(self):
        """Cleanup and close."""
        self.is_running = False
        self.is_recording = False
        
        # Wait for threads
        time.sleep(0.5)
        
        # Close robot
        self.robot.close()
        
        # Close window
        self.root.quit()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="GUI-based data collection for LeRobot")
    parser.add_argument("--repo-id", type=str, default="waste-e/garbage-picking-gui",
                        help="Dataset repository ID")
    parser.add_argument("--root", type=str, default="./lerobot_datasets",
                        help="Root directory for dataset storage")
    parser.add_argument("--camera-ids", type=int, nargs=3, default=[0, 1, 2],
                        help="Camera device IDs for top, front, wrist")
    parser.add_argument("--fps", type=int, default=30,
                        help="Recording framerate")
    parser.add_argument("--robot-type", type=str, default="waste-e",
                        help="Robot type")
    parser.add_argument("--task", type=str, default="pick_garbage",
                        help="Default task description")
    
    args = parser.parse_args()
    
    # Define features
    features = {
        "observation.images.top": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["C", "H", "W"],
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["C", "H", "W"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["C", "H", "W"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": [
                "motor_pos_0", "motor_pos_1", "motor_pos_2",
                "motor_pos_3", "motor_pos_4", "motor_pos_5",
                "motor_vel_0", "motor_vel_1", "motor_vel_2",
                "motor_vel_3", "motor_vel_4", "motor_vel_5",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["motor_cmd_0", "motor_cmd_1", "motor_cmd_2",
                     "motor_cmd_3", "motor_cmd_4", "motor_cmd_5"],
        },
    }
    
    # Initialize robot
    print("Initializing robot interface...")
    camera_ids = {
        'top': args.camera_ids[0],
        'front': args.camera_ids[1],
        'wrist': args.camera_ids[2]
    }
    robot = RobotInterface(camera_ids=camera_ids)
    
    # Create dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_id = f"{args.repo_id}_{timestamp}"
    
    print(f"Creating dataset: {repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        features=features,
        root=Path(args.root),
        robot_type=args.robot_type,
        use_videos=True,
        image_writer_threads=4,
    )
    
    # Create GUI
    root = tk.Tk()
    app = DataCollectionGUI(root, robot, dataset, fps=args.fps, task=args.task)
    
    # Handle window close
    def on_closing():
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?\n\nDataset will be finalized."):
            app.cleanup()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main()
