#!/usr/bin/env python3
"""
Evaluate Pi0.5 policy running on remote RunPod server (via SSH port-forward)
Fixes:
- Uses a single explicit joint order for BOTH state extraction and action sending (NO sorting)
- Validates observation keys, image keys, and action shape
- Keeps FPS stable
"""

import time
import numpy as np
import requests
import base64
import io
from PIL import Image

# Run this on your local machine (if server is not publicly exposed):
# ssh -L 8001:localhost:8001 vglalala@207.6.198.210 -p 2222

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

# -------------------
# USER SETTINGS
# -------------------
FPS = 10
TASK_DESCRIPTION = "Pick an object in front of you and place in the bin behind you."
ACTIONS_TO_EXECUTE = 20

INFERENCE_SERVER_URL = "http://localhost:8001"

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_awesome_follower_arm"

CAMERA_CONFIG = {
    "front": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=10, warmup_s=5),
    "top": OpenCVCameraConfig(index_or_path=2, width=1280, height=720, fps=10, warmup_s=5),
    "wrist": OpenCVCameraConfig(index_or_path=6, width=1280, height=720, fps=10, warmup_s=5),
}

# -------------------
# IMPORTANT: FIXED JOINT ORDER
# Use the same order for:
#   (1) state vector you send to the model
#   (2) action vector you send to the robot
# -------------------
JOINT_ORDER = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

CAM_ORDER = ["front", "top", "wrist"]


def encode_image(image_array: np.ndarray) -> str:
    """Encode numpy image to base64 string (JPEG)."""
    img = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_state(obs: dict) -> list[float]:
    """Extract state in JOINT_ORDER from observation."""
    missing = [k for k in JOINT_ORDER if k not in obs]
    if missing:
        raise RuntimeError(
            f"Observation missing joint keys: {missing}\n"
            f"Available keys: {list(obs.keys())}"
        )
    return [float(obs[k]) for k in JOINT_ORDER]


def build_images(obs: dict) -> dict[str, str]:
    """Extract and encode images in CAM_ORDER from observation."""
    images_b64: dict[str, str] = {}
    missing = []
    for cam in CAM_ORDER:
        if cam in obs:
            images_b64[cam] = encode_image(obs[cam])
        else:
            missing.append(cam)

    if missing:
        # Not always fatal, but usually you want all cams
        print(f"⚠️ Warning: Missing camera(s) in observation: {missing}. Available: {list(obs.keys())}")

    if not images_b64:
        raise RuntimeError(f"No camera images found in observation. Available keys: {list(obs.keys())}")

    return images_b64


def get_remote_prediction(obs: dict, prompt: str) -> np.ndarray:
    """Call inference server for action sequence."""
    state = build_state(obs)
    images_b64 = build_images(obs)

    payload = {
        "prompt": prompt,
        "state": state,
        "images": images_b64,
    }

    debug_payload = {
        "prompt": prompt,
        "state": state,
        "images": {k: f"<base64_image_{len(v)}_bytes>" for k, v in images_b64.items()},
    }
    print(f"Debug - Sending payload structure: {debug_payload}")

    try:
        response = requests.post(
            f"{INFERENCE_SERVER_URL}/predict",
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"},
        )

        print(f"Debug - Response status: {response.status_code}")
        print(f"Debug - Response text: {response.text[:500]}...")

        if response.status_code != 200:
            raise RuntimeError(f"Server error (status {response.status_code}): {response.text}")

        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"Prediction failed: {result.get('error', 'Unknown error')}")

        actions = np.array(result["actions"], dtype=np.float32)

        # Validate shape: should be (T, 6)
        if actions.ndim != 2:
            raise RuntimeError(f"Expected actions to be 2D (T, D). Got shape {actions.shape}")
        if actions.shape[1] < len(JOINT_ORDER):
            raise RuntimeError(
                f"Expected at least {len(JOINT_ORDER)} action dims, got {actions.shape[1]}.\n"
                f"Actions shape: {actions.shape}"
            )

        return actions

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}") from e


def main():
    # Health check
    try:
        print(f"Connecting to inference server at {INFERENCE_SERVER_URL}...")
        health = requests.get(f"{INFERENCE_SERVER_URL}/health", timeout=5).json()
        print(f"✅ Server status: {health}")
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to server: {e}")
        print("Make sure SSH port-forward is running and server is up.")
        return

    # Connect robot
    print(f"\nConnecting to robot on {ROBOT_PORT}...")
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=CAMERA_CONFIG,
    )
    robot = SO101Follower(robot_config)
    robot.connect()

    if not robot.is_connected:
        raise RuntimeError("Robot is not connected!")

    # Optional: print robot action keys so you can compare
    print("Robot action_features order:", list(robot.action_features.keys()))
    print("Using JOINT_ORDER:", JOINT_ORDER)

    print("\n✅ Robot connected! Starting control loop...")
    print(f"   Cameras: {list(CAMERA_CONFIG.keys())}")
    print(f"   FPS: {FPS}")

    step = 0
    last_actions: np.ndarray | None = None
    action_index = 0
    pred_times: list[float] = []

    try:
        while True:
            t0 = time.perf_counter()

            need_new = (
                last_actions is None
                or action_index >= min(ACTIONS_TO_EXECUTE, len(last_actions))
            )

            if need_new:
                obs = robot.get_observation()
                
                # Extract current state for delta->absolute conversion
                current_state = np.array(build_state(obs), dtype=np.float32)

                t_pred_start = time.perf_counter()
                predicted_deltas = get_remote_prediction(obs, TASK_DESCRIPTION)
                t_pred = time.perf_counter() - t_pred_start

                # Convert deltas to absolute actions
                # First 5 joints: absolute = delta + current_state
                # Last joint (gripper): already absolute
                last_actions = predicted_deltas.copy()
                last_actions[:, :5] = predicted_deltas[:, :5] + current_state[:5]
                # Gripper (index 5) stays as-is (already absolute)

                pred_times.append(t_pred)
                pred_times = pred_times[-10:]
                action_index = 0

                print(
                    f"Step {step}: received {len(last_actions)} actions | "
                    f"pred {t_pred:.3f}s (avg {np.mean(pred_times):.3f}s)"
                )
                # Debug a sample
                print("Debug - ACTION[0]:", last_actions[0, :len(JOINT_ORDER)].tolist())

            # Execute one action
            action = last_actions[action_index]

            # IMPORTANT: Map by JOINT_ORDER (same order as state)
            action_dict = {
                JOINT_ORDER[i]: float(action[i]) for i in range(len(JOINT_ORDER))
            }

            robot.send_action(action_dict)

            action_index += 1
            step += 1

            # Maintain FPS
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, (1.0 / FPS) - dt))

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print("Disconnecting robot...")
        robot.disconnect()
        print("✅ Done!")


if __name__ == "__main__":
    main()