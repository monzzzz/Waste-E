"""
NVIDIA Alpamayo 1.5 inference wrapper for Waste-E Isaac Sim.

Alpamayo runs in a Python 3.12 subprocess (alpamayo_worker.py) because the
alpamayo1_5 package requires Python 3.12 while Isaac Sim uses Python 3.11.

Communication: newline-delimited JSON on stdin/stdout.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np


# Camera index constants matching Alpamayo's training camera set:
#   0 = camera_cross_left_120fov   (Front left)
#   1 = camera_front_wide_120fov   (Front wide 120°)
#   2 = camera_cross_right_120fov  (Front right)
#   6 = camera_front_tele_30fov    (Front telephoto)
# SiT only has 5 × 360° cameras, so we map best-effort:
#   SiT cam2 → idx 0 (front-left), SiT cam3 → idx 1 (front wide),
#   SiT cam4 → idx 2 (front-right), SiT cam3 reused → idx 6 (telephoto)
_CAM_IDX_FRONT_LEFT  = 0
_CAM_IDX_FRONT_WIDE  = 1
_CAM_IDX_FRONT_RIGHT = 2
_CAM_IDX_FRONT_TELE  = 6

# Worker Python (3.12 venv with alpamayo1_5 installed)
_WORKER_PYTHON = os.environ.get(
    "ALPAMAYO_PYTHON",
    "/opt/alpamayo-env/bin/python",
)
_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "alpamayo_worker.py")

# History window: 16 steps @ 10 Hz = 1.6 s
_HISTORY_LEN = 16


@dataclass
class DriveCommand:
    linear_x: float
    angular_z: float
    confidence: float = 1.0
    waypoints: Optional[np.ndarray] = None


def _waypoints_to_command(
    waypoints: np.ndarray,
    lookahead_steps: int = 3,
    max_linear: float = 1.2,
    max_angular: float = 1.0,
    min_linear: float = 0.3,   # floor speed so small waypoints still drive forward
) -> DriveCommand:
    """Pure-pursuit: ego-frame (x=forward, y=left) waypoints → drive command."""
    n = min(lookahead_steps, len(waypoints))
    if n == 0:
        return DriveCommand(linear_x=0.0, angular_z=0.0, confidence=0.0)

    x, y = float(waypoints[n - 1, 0]), float(waypoints[n - 1, 1])
    dist = math.hypot(x, y)
    if dist < 1e-4:
        # Model predicts staying still — use minimum forward speed
        return DriveCommand(linear_x=min_linear, angular_z=0.0, waypoints=waypoints)

    curvature = 2.0 * y / (dist ** 2)
    speed = min(dist / (lookahead_steps * 0.1), max_linear)
    speed = max(speed, min_linear)  # enforce floor
    angular = float(np.clip(speed * curvature, -max_angular, max_angular))
    linear  = float(np.clip(speed, min_linear, max_linear))
    return DriveCommand(linear_x=linear, angular_z=angular, waypoints=waypoints)


class _EgoHistory:
    """Maintains a rolling window of ego-frame position/rotation history."""

    def __init__(self, length: int = _HISTORY_LEN):
        self.length = length
        self._xyz: Deque[np.ndarray] = deque(maxlen=length)
        self._rot: Deque[np.ndarray] = deque(maxlen=length)

    def update(self, pos: np.ndarray, rot_mat: np.ndarray):
        self._xyz.append(pos.astype(np.float32))
        self._rot.append(rot_mat.astype(np.float32))

    def as_tensors(self) -> Tuple[List, List]:
        """Return (ego_history_xyz, ego_history_rot) as nested Python lists."""
        # Pad with the oldest entry (or zeros) until we have self.length frames
        xyz_list = list(self._xyz)
        rot_list = list(self._rot)

        pad_xyz = np.zeros(3, dtype=np.float32) if not xyz_list else xyz_list[0]
        pad_rot = np.eye(3, dtype=np.float32)   if not rot_list else rot_list[0]

        while len(xyz_list) < self.length:
            xyz_list.insert(0, pad_xyz)
            rot_list.insert(0, pad_rot)

        # Shape: (1, 1, T, 3) and (1, 1, T, 3, 3)
        xyz_arr = np.stack(xyz_list)[None, None]   # (1,1,T,3)
        rot_arr = np.stack(rot_list)[None, None]   # (1,1,T,3,3)
        return xyz_arr.tolist(), rot_arr.tolist()


class AlpamayoModel:
    def __init__(
        self,
        model_path: str,
        input_width: int = 640,
        input_height: int = 384,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self.confidence_threshold = confidence_threshold
        self.device = device

        self._proc: Optional[subprocess.Popen] = None
        self._backend: Optional[str] = None
        self._ego = _EgoHistory()
        self._tmp_dir: Optional[str] = None
        self._last_cmd: DriveCommand = DriveCommand(linear_x=0.3, angular_z=0.0)
        self._infer_every: int = 6   # call model every N steps, reuse between
        self._infer_counter: int = 0

        if model_path:
            self._start_worker(model_path)

    # ------------------------------------------------------------------
    def _start_worker(self, model_path: str):
        if not os.path.exists(_WORKER_PYTHON):
            print(f"[Alpamayo] Worker Python not found: {_WORKER_PYTHON}")
            print("[Alpamayo] Running with dummy 0.3 m/s forward command.")
            return

        print(f"[Alpamayo] Starting worker ({_WORKER_PYTHON}) …")
        self._proc = subprocess.Popen(
            [_WORKER_PYTHON, _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._tmp_dir = tempfile.mkdtemp(prefix="alpamayo_imgs_")

        # Send init
        init = {"model_path": model_path, "device": self.device}
        self._proc.stdin.write(json.dumps(init) + "\n")
        self._proc.stdin.flush()

        # Wait for "loading" then "ready"
        for _ in range(2):
            raw = self._proc.stdout.readline()
            if not raw:
                break
            resp = json.loads(raw)
            print(f"[Alpamayo] Worker: {resp.get('status', '?')}")
            if resp.get("status") == "ready":
                self._backend = "alpamayo1_5"
                print("[Alpamayo] Model loaded and ready.")
                return

        print("[Alpamayo] Worker failed to reach ready state — using dummy command.")
        self._cleanup_proc()

    def _cleanup_proc(self):
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
        import shutil
        if self._tmp_dir and os.path.isdir(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    # ------------------------------------------------------------------
    def _save_image(self, img: Optional[np.ndarray], name: str) -> Optional[str]:
        """Write img (H×W×3 uint8) to a temp PNG; return path or None."""
        if img is None or self._tmp_dir is None:
            return None
        import cv2
        path = os.path.join(self._tmp_dir, f"{name}.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return path

    def update_ego(self, pos: np.ndarray, rot_mat: np.ndarray):
        """Call each simulation step with the robot's world pose."""
        self._ego.update(pos, rot_mat)

    # ------------------------------------------------------------------
    def infer(
        self,
        front_img:  Optional[np.ndarray] = None,
        left_img:   Optional[np.ndarray] = None,
        right_img:  Optional[np.ndarray] = None,
        tele_img:   Optional[np.ndarray] = None,
        nav_text:   Optional[str] = None,  # e.g. "Turn left", "Go straight", "Stop"
    ) -> DriveCommand:
        if self._backend is None:
            return DriveCommand(linear_x=0.3, angular_z=0.0, confidence=0.0)

        # Only call the model every _infer_every steps — reuse last result between calls
        self._infer_counter += 1
        if self._infer_counter % self._infer_every != 0:
            return self._last_cmd

        if tele_img is None:
            tele_img = front_img  # SiT has no telephoto — reuse front

        try:
            front_path = self._save_image(front_img, "front")
            left_path  = self._save_image(left_img,  "left")
            right_path = self._save_image(right_img, "right")
            tele_path  = self._save_image(tele_img,  "tele")

            ego_xyz, ego_rot = self._ego.as_tensors()

            # Send 4 cameras matching Alpamayo's training set
            request = {
                "image_paths":    [left_path, front_path, right_path, tele_path],
                "camera_indices": [_CAM_IDX_FRONT_LEFT, _CAM_IDX_FRONT_WIDE,
                                   _CAM_IDX_FRONT_RIGHT, _CAM_IDX_FRONT_TELE],
                "ego_history_xyz": ego_xyz,
                "ego_history_rot": ego_rot,
                "nav_text": nav_text,
            }
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()

            raw = self._proc.stdout.readline()
            if not raw:
                print("[Alpamayo] Worker closed unexpectedly.")
                self._backend = None
                return DriveCommand(linear_x=0.3, angular_z=0.0, confidence=0.0)

            resp = json.loads(raw)
            if resp.get("status") == "error":
                print(f"[Alpamayo] Worker error: {resp.get('msg', '')[:200]}")
                return DriveCommand(linear_x=0.3, angular_z=0.0, confidence=0.0)

            waypoints = np.array(resp["waypoints"])  # (T, 2)
            if len(waypoints) == 0:
                return DriveCommand(linear_x=0.3, angular_z=0.0, confidence=0.5)

            self._last_cmd = _waypoints_to_command(waypoints)
            return self._last_cmd

        except Exception as e:
            print(f"[Alpamayo] Inference error: {e}")
            return DriveCommand(linear_x=0.3, angular_z=0.0, confidence=0.0)

    def __del__(self):
        self._cleanup_proc()
