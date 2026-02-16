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

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import busy_wait

# -------------------
# USER SETTINGS
# -------------------
FPS = 30
TASK_DESCRIPTION = "Pick an object in front of you and place in the bin behind you."
ACTIONS_TO_EXECUTE = 20

RUNPOD_SERVER_URL = "https://your-runpod-url.runpod.io"  # Get from RunPod exposed port
ROBOT_IP = "YOUR_ROBOT_IP"
ROBOT_ID = "waste-e-pi05-finetuned-v1"

def encode_image(image_array):
    """Encode numpy image to base64 string"""
    img = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_remote_prediction(obs, prompt):
    """Call RunPod server for inference"""
    # Encode images
    images_b64 = {}
    for cam in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        if f"observation.images.{cam}" in obs:
            images_b64[cam] = encode_image(obs[f"observation.images.{cam}"])
    
    # Build request
    payload = {
        "prompt": prompt,
        "state": obs["observation.state"].tolist(),
        "images": images_b64
    }
    
    # Call server
    response = requests.post(
        f"{RUNPOD_SERVER_URL}/predict",
        json=payload,
        timeout=30
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"Server error: {response.text}")
    
    result = response.json()
    if not result["success"]:
        raise RuntimeError(f"Prediction failed: {result['error']}")
    
    return np.array(result["actions"])

def main():
    # Test server connection
    print(f"Testing connection to {RUNPOD_SERVER_URL}...")
    health = requests.get(f"{RUNPOD_SERVER_URL}/health").json()
    print(f"Server status: {health}")
    
    # Connect to robot
    robot_config = LeKiwiClientConfig(remote_ip=ROBOT_IP, id=ROBOT_ID)
    robot = LeKiwiClient(robot_config)
    robot.connect()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print(f"✅ Robot connected! Starting control loop...")
    
    step = 0
    last_actions = None
    action_index = 0
    pred_times = []

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
        for i, name in enumerate(list(robot.action_features.keys())):
            if i < len(action):
                action_dict[name] = float(action[i])
        
        robot.send_action(action_dict)
        action_index += 1
        step += 1
        
        # Maintain FPS
        busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

if __name__ == "__main__":
    main()