"""
NVIDIA Alpamayo 1.5 inference wrapper for Waste-E Isaac Sim.

Alpamayo outputs 64 future waypoints at 10 Hz (6.4 s horizon).
We take the first few waypoints and convert them to (linear_x, angular_z)
commands for the differential-drive Waste-E robot.

Requires:
  pip install git+https://github.com/NVlabs/alpamayo1.5.git
  pip install transformers>=4.57.1 accelerate torch>=2.8
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class DriveCommand:
    linear_x: float    # forward velocity (m/s)
    angular_z: float   # yaw rate         (rad/s)
    confidence: float = 1.0
    waypoints: Optional[np.ndarray] = None   # (N, 2) xy waypoints if available


def _waypoints_to_command(
    waypoints: np.ndarray,
    lookahead_steps: int = 3,
    max_linear: float = 1.2,
    max_angular: float = 1.0,
) -> DriveCommand:
    """
    Pure-pursuit conversion from Alpamayo waypoints → drive command.
    waypoints: (N, 2+) array of (x, y, ...) in ego frame (x=forward, y=left).
    """
    n = min(lookahead_steps, len(waypoints))
    if n == 0:
        return DriveCommand(linear_x=0.0, angular_z=0.0, confidence=0.0)

    target = waypoints[n - 1, :2]   # (x, y) of lookahead point
    x, y = float(target[0]), float(target[1])
    dist = math.hypot(x, y)

    if dist < 1e-4:
        return DriveCommand(linear_x=0.0, angular_z=0.0, waypoints=waypoints)

    # Pure pursuit curvature: κ = 2y / dist²
    curvature = 2.0 * y / (dist ** 2)
    speed = min(dist / (lookahead_steps * 0.1), max_linear)   # dt=0.1 s per step
    angular = float(np.clip(speed * curvature, -max_angular, max_angular))
    linear  = float(np.clip(speed, 0.0, max_linear))

    return DriveCommand(linear_x=linear, angular_z=angular, waypoints=waypoints)


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
        self._model = None
        self._processor = None
        self._backend: Optional[str] = None

        if model_path:
            self._load(model_path)

    # ------------------------------------------------------------------
    def _load(self, path: str):
        import os
        if not os.path.isdir(path) and not os.path.isfile(path):
            raise FileNotFoundError(f"Alpamayo model not found: {path}")

        self._load_alpamayo_package(path)

    def _load_alpamayo_package(self, path: str):
        """Load using the official alpamayo1_5 package from NVlabs/alpamayo1.5."""
        try:
            import torch
            # Try the official alpamayo1_5 package first
            from alpamayo1_5 import AlpamayoForCausalLM, AlpamayoProcessor

            print(f"[Alpamayo] Loading model from {path} ...")
            self._processor = AlpamayoProcessor.from_pretrained(path)
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                self._model = AlpamayoForCausalLM.from_pretrained(
                    path, quantization_config=bnb_cfg, device_map="auto"
                )
                print("[Alpamayo] Loaded in 4-bit quantization (low VRAM mode).")
            except Exception:
                self._model = AlpamayoForCausalLM.from_pretrained(
                    path, torch_dtype=torch.bfloat16, device_map="auto"
                )
            self._model.eval()
            self._backend = "alpamayo1_5"
            print("[Alpamayo] Model loaded successfully via alpamayo1_5 package.")
            return
        except ImportError:
            print("[Alpamayo] alpamayo1_5 package not found.")
            print("[Alpamayo] Install it with:")
            print("  <isaac_sim>/python.sh -m pip install git+https://github.com/NVlabs/alpamayo1.5.git")
        except Exception as e:
            import traceback
            print(f"[Alpamayo] alpamayo1_5 load error: {e}")
            traceback.print_exc()

        # Fallback: try generic HuggingFace AutoModel with trust_remote_code
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            print(f"[Alpamayo] Trying AutoModelForCausalLM with trust_remote_code=True ...")
            self._processor = AutoProcessor.from_pretrained(
                path, trust_remote_code=True
            )
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                self._model = AutoModelForCausalLM.from_pretrained(
                    path, trust_remote_code=True, quantization_config=bnb_cfg, device_map="auto"
                )
                print("[Alpamayo] Loaded in 4-bit quantization (low VRAM mode).")
            except Exception:
                self._model = AutoModelForCausalLM.from_pretrained(
                    path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
                )
            self._model.eval()
            self._backend = "transformers"
            print("[Alpamayo] Model loaded via AutoModelForCausalLM.")
            return
        except Exception as e:
            print(f"[Alpamayo] AutoModel fallback failed: {e}")

        print("[Alpamayo] All load attempts failed — using dummy 0.3 m/s forward command.")

    # ------------------------------------------------------------------
    def _prepare_images(self, *images: Optional[np.ndarray]) -> List[np.ndarray]:
        import cv2
        out = []
        # Alpamayo expects 1080x1920; processor will downsample to 320x576
        target_h, target_w = 1080, 1920
        for img in images:
            if img is None:
                img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (target_w, target_h))
            out.append(img)
        # Pad to 4 cameras if needed (Alpamayo default = 4)
        while len(out) < 4:
            out.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
        return out

    def _run_alpamayo1_5(
        self, front, left, right
    ) -> DriveCommand:
        import torch
        images = self._prepare_images(front, left, right)

        # Build a minimal prompt — can be expanded with navigation guidance
        prompt = "Drive forward safely."

        inputs = self._processor(
            images=images,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Decode trajectory waypoints from model output
        waypoints = self._extract_waypoints(outputs)
        if waypoints is not None and len(waypoints) > 0:
            return _waypoints_to_command(waypoints)
        return DriveCommand(linear_x=0.3, angular_z=0.0, confidence=0.5)

    def _extract_waypoints(self, outputs) -> Optional[np.ndarray]:
        """Extract (N, 2) xy waypoints from model outputs if available."""
        try:
            if hasattr(outputs, "waypoints"):
                return np.array(outputs.waypoints)
            if hasattr(outputs, "trajectory"):
                traj = outputs.trajectory
                if hasattr(traj, "cpu"):
                    traj = traj.cpu().numpy()
                return np.array(traj)[..., :2]
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def infer(
        self,
        front_img: Optional[np.ndarray] = None,
        left_img:  Optional[np.ndarray] = None,
        right_img: Optional[np.ndarray] = None,
    ) -> DriveCommand:
        if self._backend is None:
            # Dummy mode
            return DriveCommand(linear_x=0.3, angular_z=0.0, confidence=0.0)

        try:
            return self._run_alpamayo1_5(front_img, left_img, right_img)
        except Exception as e:
            print(f"[Alpamayo] Inference error: {e}")
            return DriveCommand(linear_x=0.3, angular_z=0.0, confidence=0.0)
