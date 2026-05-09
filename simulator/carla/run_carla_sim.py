"""
Waste-E CARLA simulation — Alpamayo 1.5 navigation in a realistic urban world.

Architecture:
  run_carla_sim.py (Python 3.11)
      └── carla_bridge_worker.py (Python 3.7, CARLA 0.9.15 API)
              └── CARLA server 0.9.15
      └── alpamayo_worker.py (Python 3.12, Alpamayo 1.5-10B)

Communication: newline-delimited JSON via subprocess stdin/stdout.

Usage:
    python3 run_carla_sim.py [--map MAP] [--steps N] [--output DIR]
                             [--nav-text TEXT] [--no-peds]

Prerequisites:
    1. CARLA server running:
       nohup env DISPLAY=:99 XDG_RUNTIME_DIR=/tmp/runtime-dir LD_PRELOAD=/tmp/fakeroot.so \\
           /workspace/carla_server/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping \\
           CarlaUE4 -RenderOffScreen -nosound -carla-server -fps=20 &
    2. pip install carla==0.9.16  (Python 3.11 side — used only for types, NOT API)
    3. python3.7 installed with CARLA 0.9.15 egg (bridge worker)
    4. Alpamayo env: /opt/alpamayo-env/bin/python (Python 3.12)
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── project paths ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "isaacsim"))

from alpamayo_inference import AlpamayoModel, DriveCommand   # noqa: E402

try:
    from visualizer import build_debug_frame
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from domain_adapter import DomainAdapter
    _HAS_ADAPTER = True
except ImportError:
    _HAS_ADAPTER = False

# ── bridge worker config ──────────────────────────────────────────────────────
_WORKER_PYTHON = "/usr/bin/python3.7"
_WORKER_SCRIPT = str(Path(__file__).parent / "carla_bridge_worker.py")

_IMG_W, _IMG_H = 640, 384


# ── CARLA bridge (subprocess) ─────────────────────────────────────────────────
class CarlaBridge:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        map_name: str = "Town03",
        num_peds: int = 30,
    ):
        env = {**os.environ,
               "PYTHONPATH": (
                   "/workspace/carla_server/PythonAPI/carla/dist/"
                   "carla-0.9.15-py3.7-linux-x86_64.egg"
               )}
        self._proc = subprocess.Popen(
            [_WORKER_PYTHON, _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=env,
        )

        init = {
            "host": host, "port": port,
            "map": map_name, "num_peds": num_peds,
        }
        self._proc.stdin.write(json.dumps(init) + "\n")
        self._proc.stdin.flush()
        print("[Bridge] Waiting for CARLA world to load…")

        resp = json.loads(self._proc.stdout.readline())
        if resp.get("status") != "ready":
            raise RuntimeError(f"Bridge failed to start: {resp}")
        print(f"[Bridge] Ready. Map: {resp.get('map')}")

    def step(self, cmd: DriveCommand, max_speed_ms: float = 3.0,
             min_speed_ms: float = 1.5) -> Optional[Dict]:
        """Send drive command, receive images + pose. Returns None on failure."""
        effective_linear = max(cmd.linear_x, min_speed_ms)
        msg = {"linear_x": effective_linear, "angular_z": cmd.angular_z,
               "max_speed_ms": max_speed_ms}
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        if not raw:
            return None
        return json.loads(raw)

    def close(self):
        try:
            self._proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
            self._proc.stdin.flush()
            self._proc.stdout.readline()  # "bye"
        except Exception:
            pass
        self._proc.terminate()


# ── image decoding ────────────────────────────────────────────────────────────
def _b64_to_rgb(b64: Optional[str]) -> Optional[np.ndarray]:
    if b64 is None:
        return None
    jpg = base64.b64decode(b64)
    arr = np.frombuffer(jpg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── fallback frame builder ────────────────────────────────────────────────────
def _simple_frame(
    imgs: Dict[str, Optional[np.ndarray]],
    cmd: DriveCommand,
    step: int,
    scene: int = 0,
) -> np.ndarray:
    panels = []
    for k in ("left", "front", "right"):
        img = imgs.get(k)
        if img is None:
            img = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)
        panels.append(img)
    row = np.hstack(panels)
    hud = np.zeros((40, row.shape[1], 3), dtype=np.uint8)
    txt = (f"scene={scene:03d}  step={step:05d}  lin={cmd.linear_x:.2f} m/s  "
           f"ang={cmd.angular_z:.2f} rad/s  conf={cmd.confidence:.2f}")
    cv2.putText(hud, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 255, 200), 1, cv2.LINE_AA)
    return np.vstack([row, hud])


# ── real-time nav command helpers ────────────────────────────────────────────
_NAV_FILE = "/tmp/waste_e_nav.txt"


def _read_nav(default: str) -> str:
    """Read current nav command from file; return default if missing/empty."""
    try:
        txt = Path(_NAV_FILE).read_text().strip()
        return txt if txt else default
    except OSError:
        return default


def _nav_angular_bias(nav_text: str) -> float:
    """
    Keyword-based angular bias injected on top of the model's output.
    Positive = counter-clockwise (left turn) in right-handed frame.
    """
    t = nav_text.lower()
    if "sharp left" in t:
        return 0.8
    if "sharp right" in t:
        return -0.8
    if "left" in t:
        return 0.4
    if "right" in t:
        return -0.4
    return 0.0


def _nav_speed_scale(nav_text: str) -> float:
    t = nav_text.lower()
    if "stop" in t:
        return 0.0
    if "slow" in t:
        return 0.5
    return 1.0


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Waste-E CARLA simulation")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", default=str(REPO_ROOT / "output_carla"))
    parser.add_argument("--model-path", default=str(REPO_ROOT / "alpamayo-1.5-10b"))
    parser.add_argument("--nav-text", default="Go straight")
    parser.add_argument("--nav-file", default=_NAV_FILE,
                        help="File polled every step for real-time nav commands")
    parser.add_argument("--no-peds", action="store_true")
    parser.add_argument("--num-scenes", type=int, default=1,
                        help="Number of scenes (respawn locations) to run")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--max-speed", type=float, default=1.5,
                        help="Max walker speed m/s (Husky: ~1.2 m/s)")
    parser.add_argument("--min-speed", type=float, default=0.8,
                        help="Min cruise speed m/s when Alpamayo predicts near-zero")
    parser.add_argument("--waypoint-scale", type=float, default=10.0,
                        help="Scale Alpamayo waypoints up (>1) to amplify small predictions")
    parser.add_argument("--debug-waypoints", action="store_true",
                        help="Print raw Alpamayo waypoints every step")
    parser.add_argument("--use-adapter", action="store_true",
                        help="Apply SDXL-Turbo sim-to-real domain adaptation")
    parser.add_argument("--adapter-strength", type=float, default=0.35,
                        help="img2img strength 0=no change, 1=full redraw (default 0.35)")
    parser.add_argument("--adapter-prompt", type=str,
                        default="photorealistic urban sidewalk, city street, "
                                "real photograph, DSLR camera, sharp focus, natural lighting",
                        help="Prompt describing the target real-world domain")
    args = parser.parse_args()

    output_dir = Path(args.output)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── CARLA bridge ──────────────────────────────────────────────────────────
    bridge = CarlaBridge(
        host=args.host,
        port=args.port,
        map_name=args.map,
        num_peds=0 if args.no_peds else 30,
    )

    # ── Alpamayo ──────────────────────────────────────────────────────────────
    model = AlpamayoModel(
        model_path=args.model_path,
        input_width=_IMG_W,
        input_height=_IMG_H,
        confidence_threshold=0.5,
        device="cuda",
    )

    # ── domain adapter (optional) ─────────────────────────────────────────────
    adapter: Optional[DomainAdapter] = None
    if args.use_adapter:
        if not _HAS_ADAPTER:
            print("[Waste-E] domain_adapter.py not found — skipping adapter.")
        else:
            adapter = DomainAdapter(
                strength=args.adapter_strength,
                prompt=args.adapter_prompt,
                device="cuda",
            )

    # ── write initial nav command to file ────────────────────────────────────
    Path(args.nav_file).write_text(args.nav_text)
    print(f"[Waste-E] Nav command file: {args.nav_file}")
    print(f"[Waste-E] Change nav command with:  echo 'Turn left' > {args.nav_file}")

    # ── metrics ───────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)  # ensure dir exists regardless of CWD
    metrics_path = output_dir / "metrics.csv"
    mf = open(metrics_path, "w", newline="")
    writer = csv.writer(mf)
    writer.writerow(["scene", "step", "pos_x", "pos_y", "pos_z", "yaw_deg",
                     "linear_x", "angular_z", "confidence", "nav_text"])

    frame_idx = 0

    print(f"[Waste-E] Running {args.num_scenes} scene(s) × {args.steps} steps. "
          f"nav_text='{args.nav_text}'")

    try:
      for scene in range(args.num_scenes):
        if scene > 0:
            # Respawn at a new random navmesh location
            bridge._proc.stdin.write(json.dumps({"respawn": True}) + "\n")
            bridge._proc.stdin.flush()
            resp = json.loads(bridge._proc.stdout.readline())
            print(f"[Waste-E] Scene {scene}: respawned → {resp.get('pos')}")

        robot_history: List[Tuple[float, float]] = []
        cmd = DriveCommand(linear_x=0.3, angular_z=0.0)
        scene_frames_dir = frames_dir / f"scene_{scene:03d}"
        scene_frames_dir.mkdir(parents=True, exist_ok=True)
        scene_frame_idx = 0

        for step in range(args.steps):
            # ── send current command, receive new state ────────────────────
            state = bridge.step(cmd, max_speed_ms=args.max_speed,
                                min_speed_ms=args.min_speed)
            if state is None or state.get("status") != "ok":
                print(f"[Waste-E] Bridge error at scene {scene} step {step}: {state}")
                break

            # ── decode images ──────────────────────────────────────────────
            imgs = {
                "left":         _b64_to_rgb(state.get("left_img")),
                "front":        _b64_to_rgb(state.get("front_img")),
                "right":        _b64_to_rgb(state.get("right_img")),
                "tele":         _b64_to_rgb(state.get("tele_img")),
                "third_person": _b64_to_rgb(state.get("third_person_img")),
            }

            # ── robot pose ─────────────────────────────────────────────────
            # CARLA is left-handed (x=fwd, y=right, z=up).
            # Alpamayo expects right-handed (x=fwd, y=left, z=up).
            # Fix: negate y so the lateral axis is correct.
            raw = state["pos"]
            pos = np.array([raw[0], -raw[1], raw[2]], dtype=np.float32)

            # CARLA yaw: positive = clockwise (right turn).
            # Right-handed convention: positive yaw = counter-clockwise.
            # Fix: negate yaw before building the rotation matrix.
            yaw_rad = -math.radians(state["yaw_deg"])
            cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
            rot_mat = np.array([
                [cos_y, -sin_y, 0],
                [sin_y,  cos_y, 0],
                [0,      0,     1],
            ], dtype=np.float32)
            model.update_ego(pos, rot_mat)
            robot_history.append((float(pos[0]), float(pos[1])))

            # ── poll real-time nav command + adapter strength ──────────────
            current_nav = _read_nav(args.nav_text)
            if adapter is not None:
                try:
                    s = float(Path("/tmp/waste_e_adapter_strength").read_text())
                    adapter.set_strength(s)
                except (OSError, ValueError):
                    pass
                try:
                    p = Path("/tmp/waste_e_adapter_prompt").read_text().strip()
                    if p:
                        adapter.prompt = p
                except OSError:
                    pass

            # ── sim-to-real domain adaptation ──────────────────────────────
            if adapter is not None:
                adapted = adapter.adapt([
                    imgs.get("left"), imgs.get("front"),
                    imgs.get("right"), imgs.get("tele"),
                ])
                feed_imgs = {
                    "left": adapted[0], "front": adapted[1],
                    "right": adapted[2], "tele":  adapted[3],
                }
            else:
                feed_imgs = imgs

            # ── Alpamayo inference ─────────────────────────────────────────
            cmd = model.infer(
                front_img=feed_imgs.get("front"),
                left_img=feed_imgs.get("left"),
                right_img=feed_imgs.get("right"),
                tele_img=feed_imgs.get("tele"),
                nav_text=current_nav,
            )

            # Scale waypoints to amplify small predictions caused by domain gap.
            if cmd.waypoints is not None and args.waypoint_scale != 1.0:
                cmd = type(cmd)(
                    linear_x=cmd.linear_x,
                    angular_z=cmd.angular_z,
                    confidence=cmd.confidence,
                    waypoints=cmd.waypoints * args.waypoint_scale,
                )

            # Keyword-based angular bias on top of model output.
            # This makes "Turn left/right" commands actually steer the robot
            # even when the model's domain-gap predictions are near zero.
            ang_bias = _nav_angular_bias(current_nav)
            spd_scale = _nav_speed_scale(current_nav)
            cmd = DriveCommand(
                linear_x=cmd.linear_x * spd_scale,
                angular_z=cmd.angular_z + ang_bias,
                confidence=cmd.confidence,
                waypoints=cmd.waypoints,
            )

            # ── logging ────────────────────────────────────────────────────
            writer.writerow([
                scene, step,
                f"{pos[0]:.3f}", f"{pos[1]:.3f}", f"{pos[2]:.3f}",
                f"{state['yaw_deg']:.2f}",
                f"{cmd.linear_x:.3f}", f"{cmd.angular_z:.3f}",
                f"{cmd.confidence:.3f}",
                current_nav,
            ])

            if step % 20 == 0 or args.debug_waypoints:
                wp = cmd.waypoints
                wp_str = ""
                if wp is not None and len(wp):
                    pts = [f"({wp[i,0]:.2f},{wp[i,1]:.2f})" for i in range(min(3, len(wp)))]
                    wp_str = f"  wp={' '.join(pts)}"
                print(f"  scene {scene:03d} step {step:05d} | nav='{current_nav}' "
                      f"| pos=({pos[0]:.1f},{pos[1]:.1f}) "
                      f"| lin={cmd.linear_x:.2f} ang={cmd.angular_z:.3f} "
                      f"conf={cmd.confidence:.2f}{wp_str}")

            # ── save frame ─────────────────────────────────────────────────
            # Top section: raw CARLA cameras (what the sim looks like)
            if _HAS_VIZ:
                frame = build_debug_frame(
                    cam_imgs=[imgs.get("left"), imgs.get("front"), imgs.get("right")],
                    waypoints=cmd.waypoints,
                    trajectories=[],
                    sim_frame=step,
                    robot_pos=pos,
                    robot_yaw=math.radians(state["yaw_deg"]),
                    robot_history=robot_history,
                    cmd_linear=cmd.linear_x,
                    cmd_angular=cmd.angular_z,
                    confidence=cmd.confidence,
                )
            else:
                frame = _simple_frame(imgs, cmd, step, scene=scene)

            # If adapter is active, add a second row showing adapted images
            if adapter is not None:
                adapted_panels = []
                for k in ("left", "front", "right"):
                    img = feed_imgs.get(k)
                    if img is None:
                        img = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)
                    adapted_panels.append(img)
                adapted_row = np.hstack(adapted_panels)
                adapted_row = cv2.resize(adapted_row, (frame.shape[1], _IMG_H))
                label = (f"Adapted (strength={adapter.strength:.2f})  "
                         f"→ Alpamayo sees this")
                cv2.putText(adapted_row, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 180), 2, cv2.LINE_AA)
                frame = np.vstack([frame, adapted_row])

            # Third-person row always at the bottom
            tp = imgs.get("third_person")
            if tp is not None:
                tp_wide = cv2.resize(tp, (frame.shape[1], _IMG_H))
                cv2.putText(tp_wide,
                            f"3rd person | nav: {current_nav} | scene={scene:03d} step={step:05d}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 220, 80), 2, cv2.LINE_AA)
                frame = np.vstack([frame, tp_wide])

            frame_path = scene_frames_dir / f"frame_{scene_frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            scene_frame_idx += 1
            frame_idx += 1

        print(f"[Waste-E] Scene {scene} done. {scene_frame_idx} frames → {scene_frames_dir}")

    finally:
        mf.close()
        bridge.close()
        print(f"[Waste-E] Done. {frame_idx} total frames.")

    # ── stitch one video per scene ────────────────────────────────────────────
    if args.save_video and frame_idx > 0:
        for scene in range(args.num_scenes):
            sd = frames_dir / f"scene_{scene:03d}"
            if not any(sd.glob("frame_*.jpg")):
                continue
            video_path = output_dir / f"scene_{scene:03d}.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "20",
                "-i", str(sd / "frame_%06d.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(video_path),
            ], check=False)
            print(f"[Waste-E] Video: {video_path}")


if __name__ == "__main__":
    main()
