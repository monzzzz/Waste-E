from __future__ import annotations

import base64
import io
import os
import threading
from typing import Any

try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    LEROBOT_OK = True
except ImportError:
    LEROBOT_OK = False
    print("[RasPi arm] lerobot not installed — arm endpoints will return 503")

ARM_PORT = os.getenv("ARM_PORT", "/dev/ttyACM0")
ARM_ID   = os.getenv("ARM_ID", "waste_e_arm")

JOINT_ORDER = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class ArmController:
    def __init__(self, port: str = ARM_PORT, arm_id: str = ARM_ID) -> None:
        self.port   = port
        self.arm_id = arm_id
        self._lock  = threading.Lock()
        self._robot = None
        self._error: str | None = None

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._robot is not None and self._robot.is_connected

    def connect_async(self, port: str | None = None) -> None:
        threading.Thread(target=self._connect, args=(port or self.port,), daemon=True).start()

    def _connect(self, port: str) -> None:
        try:
            cfg   = SO101FollowerConfig(port=port, id=self.arm_id, cameras={})
            robot = SO101Follower(cfg)
            robot.connect()
            with self._lock:
                self._robot = robot
                self._error = None
            print(f"[RasPi arm] connected on {port}")
        except Exception as exc:
            with self._lock:
                self._robot = None
                self._error = str(exc)
            print(f"[RasPi arm] connect failed: {exc}")

    def disconnect(self) -> None:
        with self._lock:
            robot       = self._robot
            self._robot = None
            self._error = None
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:
                print(f"[RasPi arm] disconnect error: {exc}")

    def send_action(self, action: dict | list) -> dict:
        with self._lock:
            robot = self._robot
        if robot is None or not robot.is_connected:
            raise RuntimeError("arm not connected")

        if isinstance(action, list):
            if len(action) != len(JOINT_ORDER):
                raise ValueError(f"expected {len(JOINT_ORDER)} values, got {len(action)}")
            action_dict = {JOINT_ORDER[i]: float(action[i]) for i in range(len(JOINT_ORDER))}
        elif isinstance(action, dict):
            missing = [k for k in JOINT_ORDER if k not in action]
            if missing:
                raise ValueError(f"missing joints: {missing}")
            action_dict = {k: float(action[k]) for k in JOINT_ORDER}
        else:
            raise TypeError("action must be a list or dict")

        robot.send_action(action_dict)
        return action_dict

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            robot = self._robot
        if robot is None or not robot.is_connected:
            raise RuntimeError("arm not connected")

        obs    = robot.get_observation()
        state  = {k: float(obs[k]) for k in JOINT_ORDER if k in obs}
        images = _encode_images(obs)
        return {"state": state, "images": images, "joint_order": JOINT_ORDER}

    def status(self) -> dict:
        with self._lock:
            connected = self._robot is not None and self._robot.is_connected
            error     = self._error
        state = None
        if connected:
            try:
                obs   = self._robot.get_observation()
                state = {k: float(obs[k]) for k in JOINT_ORDER if k in obs}
            except Exception:
                pass
        return {
            "connected": connected,
            "port":      self.port,
            "joints":    JOINT_ORDER,
            "state":     state,
            "error":     error,
        }


def _encode_images(obs: dict) -> dict[str, str]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return {}

    images: dict[str, str] = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray) and val.ndim == 3:
            try:
                buf = io.BytesIO()
                Image.fromarray(val.astype(np.uint8)).save(buf, format="JPEG")
                images[key] = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                pass
    return images
