#!/usr/bin/env python3
"""
Evaluate Pi0.5 policy running on remote RunPod server
"""
import time
import numpy as np
import requests
import base64
import io
from PIL import Image

# run this ssh -L 8001:localhost:8001 vglalala@207.6.198.210 -p 2222

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

# -------------------
# USER SETTINGS
# -------------------
FPS = 10  # Match your recording FPS
TASK_DESCRIPTION = "Pick an object in front of you and place in the bin behind you."
ACTIONS_TO_EXECUTE = 20

INFERENCE_SERVER_URL = "http://localhost:8001"  # Get from RunPod exposed port

# Robot hardware settings (match your recording config)
ROBOT_PORT = "/dev/ttyACM0"  # Your follower arm port
ROBOT_ID = "my_awesome_follower_arm"  # Your follower arm ID (for calibration and connection)

# Camera settings (match your recording config)
CAMERA_CONFIG = {
    "front": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=10, warmup_s=5),
    "top": OpenCVCameraConfig(index_or_path=2, width=1280, height=720, fps=10, warmup_s=5),
    "wrist": OpenCVCameraConfig(index_or_path=6, width=1280, height=720, fps=10, warmup_s=5),
}

def encode_image(image_array):
    """Encode numpy image to base64 string"""
    img = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_remote_prediction(obs, prompt):
    """Call RunPod server for inference"""
    # Debug: Print observation keys to understand structure
    print(f"Debug - Observation keys: {list(obs.keys())}")
    
    # Encode images - FIX: keys are just 'front', 'top', 'wrist', not 'observation.images.X'
    images_b64 = {}
    for cam in ["front", "top", "wrist"]:
        if cam in obs:
            images_b64[cam] = encode_image(obs[cam])
            print(f"Debug - Encoded {cam} image")
    
    # Get joint states - look for keys ending in .pos
    state = []
    motor_keys = sorted([k for k in obs.keys() if k.endswith(".pos")])
    
    for key in motor_keys:
        state.append(float(obs[key]))
    
    if len(state) == 0:
        raise RuntimeError(f"Could not extract state. Keys: {list(obs.keys())}")
    
    print(f"Debug - State: length={len(state)}, values={state}")
    print(f"Debug - Images: {list(images_b64.keys())}")
    
    # Build request payload
    payload = {
        "prompt": prompt,
        "state": state,
        "images": images_b64
    }
    
    # Debug: Print payload structure (without full image data)
    debug_payload = {
        "prompt": prompt,
        "state": state,
        "images": {k: f"<base64_image_{len(v)}_bytes>" for k, v in images_b64.items()}
    }
    print(f"Debug - Sending payload structure: {debug_payload}")
    
    # Call server
    try:
        response = requests.post(
            f"{INFERENCE_SERVER_URL}/predict",
            json=payload,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Debug - Response status: {response.status_code}")
        print(f"Debug - Response text: {response.text[:500]}...")
        
        if response.status_code != 200:
            raise RuntimeError(f"Server error (status {response.status_code}): {response.text}")
        
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"Prediction failed: {result.get('error', 'Unknown error')}")
        
        # Get actions from server
        actions = np.array(result["actions"])
        
        
        return actions
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")

def main():
    # Check server health
    try:
        print(f"Connecting to RunPod server at {INFERENCE_SERVER_URL}...")
        health = requests.get(f"{INFERENCE_SERVER_URL}/health", timeout=5).json()
        print(f"✅ Server status: {health}")
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to server: {e}")
        print("Make sure RunPod server is running and URL is correct")
        return
    
    # Connect to robot hardware (motors + cameras)
    print(f"\nConnecting to robot on {ROBOT_PORT}...")
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=CAMERA_CONFIG
    )
    robot = SO101Follower(robot_config)
    robot.connect()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print(f"✅ Robot connected! Starting control loop...")
    print(f"   Cameras: {list(CAMERA_CONFIG.keys())}")
    print(f"   FPS: {FPS}")
    
    step = 0
    last_actions = None
    action_index = 0
    pred_times = []

    try:
        while True:
            t0 = time.perf_counter()

            # Need new predictions?
            if last_actions is None or action_index >= min(ACTIONS_TO_EXECUTE, len(last_actions)):
                obs = robot.get_observation()
                
                # Get predictions from RunPod server
                t_pred_start = time.perf_counter()
                last_actions = get_remote_prediction(obs, TASK_DESCRIPTION)
                t_pred = time.perf_counter() - t_pred_start
                
                pred_times.append(t_pred)
                pred_times = pred_times[-10:]
                
                action_index = 0
                
                print(
                    f"Step {step}: received {len(last_actions)} actions from server | "
                    f"pred {t_pred:.3f}s (avg {np.mean(pred_times):.3f}s)"
                )
            
            # Execute action
            action = last_actions[action_index]
            action_dict = {}
            
            # Map action array to motor commands
            motor_names = [name for name in robot.action_features.keys()]
            for i, name in enumerate(motor_names):
                if i < len(action):
                    action_dict[name] = float(action[i])
            
            robot.send_action(action_dict)
            action_index += 1
            step += 1
            
            # Maintain FPS
            dt = time.perf_counter() - t0
            time.sleep(max(0, 1.0 / FPS - dt))
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        print("Disconnecting robot...")
        robot.disconnect()
        print("✅ Done!")

if __name__ == "__main__":
    main()