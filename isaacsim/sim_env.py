"""
Isaac Sim environment for Waste-E + WPedestrian SIT dataset.
Updated for Isaac Sim 5.x API.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np
import yaml

# Isaac Sim 5.x API (falls back to omni.isaac for older installs)
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCapsule, GroundPlane
    from isaacsim.core.prims import SingleRigidPrim as RigidPrim
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.sensors.camera import Camera
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCapsule, GroundPlane
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.sensor import Camera

from pxr import Gf, UsdGeom

from pedestrian_loader import PedestrianTrajectory, WPedestrianSITDataset
from alpamayo_inference import AlpamayoModel, DriveCommand


class SiTImageLoader:
    """
    Loads real camera images from the SiT dataset instead of Isaac Sim renders.

    SiT camera layout (Clearpath Husky, 360° coverage):
      cam 1 = rear-left  cam 2 = front-left  cam 3 = front
      cam 4 = front-right  cam 5 = rear-right

    We feed cams 2 (left), 3 (front), 4 (right) to Alpamayo.
    """

    # SiT cam IDs → Alpamayo slot names
    # cam1=left, cam4=front, cam5=right
    # No telephoto in SiT — reuse cam4 (front) for the tele slot
    CAM_MAP = {"left": "1", "front": "4", "right": "5", "tele": "4"}

    def __init__(self, sequence_dir: str, target_w: int = 640, target_h: int = 384):
        self.sequence_dir = sequence_dir
        self.target_w = target_w
        self.target_h = target_h
        self._frame_ids: list[int] = []
        self._cache: dict = {}

        # Discover available frame IDs from the front camera
        front_dir = os.path.join(sequence_dir, "cam_img", "3", "data_undist")
        if os.path.isdir(front_dir):
            self._frame_ids = sorted(
                int(f.replace(".png", ""))
                for f in os.listdir(front_dir)
                if f.endswith(".png")
            )
        print(f"[SiTImageLoader] Loaded sequence: {sequence_dir} ({len(self._frame_ids)} frames)", flush=True)

    def __len__(self):
        return len(self._frame_ids)

    def get(self, step: int) -> dict[str, Optional[np.ndarray]]:
        """Return {'left', 'front', 'right'} RGB arrays for simulation step."""
        if not self._frame_ids:
            return {"left": None, "front": None, "right": None}

        frame_id = self._frame_ids[step % len(self._frame_ids)]
        result = {}
        for name, cam_id in self.CAM_MAP.items():
            path = os.path.join(
                self.sequence_dir, "cam_img", cam_id, "data_undist", f"{frame_id}.png"
            )
            if not os.path.exists(path):
                result[name] = None
                continue
            try:
                from PIL import Image as _Image
                img = _Image.open(path).convert("RGB").resize(
                    (self.target_w, self.target_h)
                )
                result[name] = np.array(img, dtype=np.uint8)
            except Exception as e:
                print(f"  [SiTImageLoader] Failed to load {path}: {e}", flush=True)
                result[name] = None
        return result


class PedestrianAgent:
    def __init__(
        self,
        prim_path: str,
        trajectory: PedestrianTrajectory,
        loop: bool = True,
        speed: float = 1.0,
    ):
        self.prim_path = prim_path
        self.trajectory = trajectory
        self.loop = loop
        self.speed = speed
        self._xform = UsdGeom.Xformable(
            World.instance().stage.GetPrimAtPath(prim_path)
        )

    def update(self, sim_frame: int):
        total_frames = self.trajectory.end_frame - self.trajectory.start_frame
        if total_frames <= 0:
            return

        scaled = int(sim_frame * self.speed)
        local_frame = (
            self.trajectory.start_frame + (scaled % total_frames)
            if self.loop
            else min(self.trajectory.start_frame + scaled, self.trajectory.end_frame)
        )

        pos = self.trajectory.position_at(local_frame)
        if pos is None:
            return
        heading = self.trajectory.heading_at(local_frame)

        xform_ops = self._xform.GetOrderedXformOps()
        if not xform_ops:
            self._xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
            self._xform.AddRotateZOp().Set(math.degrees(heading))
        else:
            for op in xform_ops:
                if "translate" in op.GetOpName():
                    op.Set(Gf.Vec3d(*pos))
                elif "rotateZ" in op.GetOpName():
                    op.Set(math.degrees(heading))


class WasteESimEnv:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.world: Optional[World] = None
        self.robot = None
        self.cameras: Dict[str, Camera] = {}
        self.pedestrian_agents: List[PedestrianAgent] = []
        self.alpamayo: Optional[AlpamayoModel] = None
        self.sit_images: Optional[SiTImageLoader] = None
        self._sit_frame_cache: dict = {}
        self._sim_frame = 0
        self._robot_history: List[tuple] = []   # (wx, wy) trail for BEV
        self._all_trajectories = []             # filled after pedestrian load

    # ------------------------------------------------------------------
    def setup(self):
        sim_cfg = self.cfg["simulation"]
        self.world = World(
            physics_dt=eval(str(sim_cfg["physics_dt"])),
            rendering_dt=eval(str(sim_cfg["rendering_dt"])),
            stage_units_in_meters=self.cfg["scene"]["stage_units_in_meters"],
        )

        self._setup_scene()
        self._spawn_robot()
        if not self.cfg.get("simulation", {}).get("no_cameras", False):
            self._spawn_cameras()
        self._spawn_pedestrians()
        self._load_alpamayo()

        self.world.reset()
        self.world.play()

        # Initialize cameras and register RGB annotator after world.reset()
        for name, cam in self.cameras.items():
            try:
                cam.initialize()
                cam.add_distance_to_image_plane_to_frame()  # warms up annotator pipeline
            except Exception as e:
                print(f"[WasteESimEnv] WARNING: Could not initialize camera '{name}': {e}")

        # Warm-up render passes so first get_rgba() isn't empty
        for _ in range(5):
            self.world.step(render=True)

        print("[WasteESimEnv] Setup complete — starting simulation loop.")

    def _setup_scene(self):
        scene_cfg = self.cfg["scene"]
        world_usd = scene_cfg.get("world_usd", "")
        if world_usd and os.path.isfile(world_usd):
            add_reference_to_stage(usd_path=world_usd, prim_path="/World/Environment")
        else:
            GroundPlane(
                prim_path="/World/GroundPlane",
                size=scene_cfg["ground_plane_size"],
                color=np.array([0.4, 0.4, 0.4]),
            )

        # Add lighting so cameras don't render black
        from pxr import UsdLux
        stage = self.world.stage
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(1000.0)
        dome.CreateColorAttr((1.0, 1.0, 1.0))
        dist = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        dist.CreateIntensityAttr(2000.0)
        dist.CreateAngleAttr(0.53)

    def _spawn_robot(self):
        robot_cfg = self.cfg["robot"]
        usd_path = robot_cfg.get("usd_path", "")
        pos = robot_cfg["start_position"]
        ori = robot_cfg["start_orientation"]

        if usd_path and os.path.isfile(usd_path):
            add_reference_to_stage(usd_path=usd_path, prim_path="/World/WasteE")
            self.robot = Robot(prim_path="/World/WasteE", name="waste_e")
            self.world.scene.add(self.robot)
        else:
            create_prim(
                prim_path="/World/WasteE",
                prim_type="Cube",
                position=np.array(pos),
                orientation=np.array(ori),
                scale=np.array([0.4, 0.3, 0.2]),
            )
            self.robot = RigidPrim(prim_path="/World/WasteE", name="waste_e")
            self.world.scene.add(self.robot)

        print(f"[WasteESimEnv] Robot spawned at {pos}")

    def _spawn_cameras(self):
        cam_cfg = self.cfg["cameras"]
        # Ensure the robot prim is an Xform so cameras can be children
        from pxr import UsdGeom
        stage = self.world.stage
        robot_prim = stage.GetPrimAtPath("/World/WasteE")
        if robot_prim and not robot_prim.IsA(UsdGeom.Xform):
            UsdGeom.Xform.Define(stage, "/World/CameraRig")
            cam_parent = "/World/CameraRig"
        else:
            cam_parent = "/World/WasteE"

        for name, ccfg in cam_cfg.items():
            try:
                cam = Camera(
                    prim_path=f"{cam_parent}/{name}_camera",
                    name=f"{name}_camera",
                    position=np.array(ccfg["position"], dtype=float),
                    resolution=(ccfg["resolution"][0], ccfg["resolution"][1]),
                    frequency=30,
                )
                cam.set_visibility(False)  # hide frustum in scene
                self.world.scene.add(cam)
                self.cameras[name] = cam
            except Exception as e:
                print(f"[WasteESimEnv] WARNING: Could not spawn camera '{name}': {e}")
        print(f"[WasteESimEnv] Cameras spawned: {list(self.cameras.keys())}")

    def _spawn_pedestrians(self):
        ped_cfg = self.cfg["pedestrians"]
        dataset_path = ped_cfg["dataset_path"]
        max_peds = ped_cfg["num_pedestrians"]
        loop = ped_cfg["loop_trajectories"]
        speed = ped_cfg["playback_speed"]
        ped_usd = ped_cfg.get("usd_path", "")

        # Resolve path relative to this file's directory
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)

        if not os.path.isdir(dataset_path):
            print(f"[WasteESimEnv] WARNING: Dataset not found at {dataset_path}")
            print("[WasteESimEnv] Run:  python isaacsim/generate_dummy_dataset.py")
            print("[WasteESimEnv] Continuing without pedestrians.")
            return

        try:
            dataset = WPedestrianSITDataset(dataset_path)
            trajectories = dataset.all_trajectories()[:max_peds]
        except Exception as e:
            print(f"[WasteESimEnv] WARNING: Could not load dataset: {e}")
            return

        for i, traj in enumerate(trajectories):
            prim_path = f"/World/Pedestrians/ped_{i:04d}"
            try:
                if ped_usd and os.path.isfile(ped_usd):
                    add_reference_to_stage(usd_path=ped_usd, prim_path=prim_path)
                else:
                    start_pos = traj.position_at(traj.start_frame) or (0.0, float(i), 0.0)
                    DynamicCapsule(
                        prim_path=prim_path,
                        position=np.array(start_pos) + np.array([0, 0, 0.9]),
                        radius=0.2,
                        height=1.6,
                        color=np.array([0.2, 0.6, 0.9]),
                    )
                agent = PedestrianAgent(
                    prim_path=prim_path,
                    trajectory=traj,
                    loop=loop,
                    speed=speed,
                )
                self.pedestrian_agents.append(agent)
            except Exception as e:
                print(f"[WasteESimEnv] WARNING: Could not spawn pedestrian {i}: {e}")

        print(f"[WasteESimEnv] Spawned {len(self.pedestrian_agents)} pedestrian(s)")
        self._all_trajectories = [a.trajectory for a in self.pedestrian_agents]

    def _load_alpamayo(self):
        import sys
        log_dir = self.cfg.get("logging", {}).get("output_dir", "output")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "alpamayo_load.log")

        alp_cfg = self.cfg["alpamayo"]
        model_path = alp_cfg.get("model_path", "")
        with open(log_path, "w") as lf:
            lf.write(f"raw model_path from config: {model_path!r}\n")

        # Resolve path relative to project root
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(
                os.path.dirname(__file__), "..", model_path.lstrip("./")
            )

        with open(log_path, "a") as lf:
            lf.write(f"resolved model_path: {model_path!r}\n")
            lf.write(f"exists: {os.path.exists(model_path)}\n")
            lf.write(f"isdir:  {os.path.isdir(model_path)}\n")
            import glob as _glob
            shards = _glob.glob(os.path.join(model_path, "*.safetensors"))
            lf.write(f"safetensors found: {shards}\n")

        if model_path and not os.path.exists(model_path):
            print(f"[WasteESimEnv] WARNING: Alpamayo model not found at {model_path}")
            print("[WasteESimEnv] Running with dummy forward command (0.3 m/s).")
            model_path = ""

        self.alpamayo = AlpamayoModel(
            model_path=model_path,
            input_width=alp_cfg["input_image_width"],
            input_height=alp_cfg["input_image_height"],
            confidence_threshold=alp_cfg["confidence_threshold"],
            device=alp_cfg["device"],
        )

        # Load real SiT images if a sequence directory is configured
        sit_seq = alp_cfg.get("sit_sequence_dir", "")
        if sit_seq:
            if not os.path.isabs(sit_seq):
                sit_seq = os.path.join(os.path.dirname(__file__), sit_seq)
            sit_seq = os.path.normpath(sit_seq)
            print(f"[WasteESimEnv] SiT sequence path: {sit_seq}", flush=True)
            print(f"[WasteESimEnv] SiT sequence exists: {os.path.isdir(sit_seq)}", flush=True)
            if os.path.isdir(sit_seq):
                self.sit_images = SiTImageLoader(
                    sequence_dir=sit_seq,
                    target_w=alp_cfg["input_image_width"],
                    target_h=alp_cfg["input_image_height"],
                )
            else:
                print(f"[WasteESimEnv] WARNING: sit_sequence_dir not found: {sit_seq}", flush=True)

    # ------------------------------------------------------------------
    def _get_sit_images(self) -> dict:
        """Return real SiT images for current step, or empty dict."""
        if self.sit_images is not None:
            imgs = self.sit_images.get(self._sim_frame)
            if self._sim_frame <= 12 and self._sim_frame % 6 == 0:
                for k, v in imgs.items():
                    if v is not None:
                        print(f"  [SiT/{k}] shape={v.shape} max={v.max()} mean={v.mean():.1f}", flush=True)
            return imgs
        return {}

    def _get_camera_image(self, name: str) -> Optional[np.ndarray]:
        """Get image: real SiT image if available, else Isaac Sim camera."""
        sit = self._sit_frame_cache.get(name)
        if sit is not None:
            return sit
        cam = self.cameras.get(name)
        if cam is None:
            return None
        try:
            rgba = cam.get_rgba()
            return rgba[:, :, :3] if rgba is not None else None
        except Exception:
            return None

    def _apply_drive_command(self, cmd: DriveCommand):
        if self.robot is None:
            return
        try:
            wheel_base = 0.3
            left_vel  = cmd.linear_x - cmd.angular_z * wheel_base / 2
            right_vel = cmd.linear_x + cmd.angular_z * wheel_base / 2

            if hasattr(self.robot, "get_articulation_controller"):
                ctrl = self.robot.get_articulation_controller()
                ctrl.apply_action(joint_velocities=np.array([left_vel, right_vel]))
            else:
                dt = eval(str(self.cfg["simulation"]["physics_dt"]))
                if hasattr(self.robot, "get_world_pose"):
                    pos, ori = self.robot.get_world_pose()
                    yaw = 2 * math.atan2(float(ori[3]), float(ori[0]))
                    pos[0] += cmd.linear_x * math.cos(yaw) * dt
                    pos[1] += cmd.linear_x * math.sin(yaw) * dt
                    self.robot.set_world_pose(position=pos)
        except Exception as e:
            print(f"[WasteESimEnv] Drive command error: {e}")

    def _get_robot_pose(self):
        """Return (pos np.ndarray(3,), rot_mat np.ndarray(3,3)) in world frame."""
        try:
            if hasattr(self.robot, "get_world_pose"):
                pos, ori = self.robot.get_world_pose()
                pos = np.array(pos, dtype=np.float32)
                # quaternion xyzw → rotation matrix
                qx, qy, qz, qw = (float(ori[i]) for i in range(4))
                rot = np.array([
                    [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                    [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                    [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
                ], dtype=np.float32)
                return pos, rot
        except Exception:
            pass
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)

    def step(self):
        self.world.step(render=len(self.cameras) > 0 and self.sit_images is None)

        for agent in self.pedestrian_agents:
            agent.update(self._sim_frame)

        # Update ego history every 6 frames (≈10 Hz at 60 Hz sim)
        if self._sim_frame % 6 == 0:
            pos, rot = self._get_robot_pose()
            self.alpamayo.update_ego(pos, rot)
            self._robot_history.append((float(pos[0]), float(pos[1])))
            if len(self._robot_history) > 120:   # keep ~12 s of history
                self._robot_history.pop(0)

        # Fetch real SiT images once per step; fallback to Isaac Sim cameras
        self._sit_frame_cache = self._get_sit_images()
        if self._sim_frame == 1:
            for k, v in self._sit_frame_cache.items():
                print(f"  [step1/SiT/{k}] {'loaded' if v is not None else 'NONE'}", flush=True)

        front = self._get_camera_image("front")
        left  = self._get_camera_image("left")
        right = self._get_camera_image("right")
        tele  = self._get_camera_image("tele")   # reuses front cam; no tele in SiT

        cmd = self.alpamayo.infer(front, left, right, tele)
        self._apply_drive_command(cmd)

        self._sim_frame += 1
        return cmd

    def run(self):
        import time
        import subprocess as _sp

        def _gpu_mb() -> str:
            try:
                out = _sp.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    timeout=2, text=True,
                ).strip().splitlines()[0]
                mem, util = out.split(",")
                return f"GPU {util.strip()}% {mem.strip()}MiB"
            except Exception:
                return "GPU n/a"

        max_steps = self.cfg["simulation"]["max_steps"]
        print(f"[WasteESimEnv] Running for up to {max_steps} steps …", flush=True)

        log_cfg = self.cfg.get("logging", {})
        out_dir = log_cfg.get("output_dir", "output")
        save_video = log_cfg.get("save_video", False)
        frames_dir = os.path.join(out_dir, "frames")
        if save_video:
            os.makedirs(frames_dir, exist_ok=True)
            print(f"[WasteESimEnv] Saving frames to {frames_dir}/", flush=True)

        metrics = []
        _frame_idx = 0   # consecutive index for ffmpeg
        _t0 = time.perf_counter()
        while self.world.is_playing() and self._sim_frame < max_steps:
            _step_t0 = time.perf_counter()
            cmd = self.step()
            step_ms = (time.perf_counter() - _step_t0) * 1000

            # Save debug frame every step
            if save_video and self._sim_frame % 2 == 0:
                try:
                    import cv2
                    from visualizer import build_debug_frame
                    H, W = 384, 640
                    cam_imgs = []
                    for cam_name in ("left", "front", "right"):
                        img = self._sit_frame_cache.get(cam_name)
                        if img is None:
                            img = self._get_camera_image(cam_name)
                        cam_imgs.append(img.copy() if img is not None else np.zeros((H, W, 3), dtype=np.uint8))

                    pos, _ = self._get_robot_pose() if self.robot else (np.zeros(3), None)
                    yaw = float(self._robot_history[-1][0]) if self._robot_history else 0.0
                    # compute yaw from last two history points
                    if len(self._robot_history) >= 2:
                        dx = self._robot_history[-1][0] - self._robot_history[-2][0]
                        dy = self._robot_history[-1][1] - self._robot_history[-2][1]
                        yaw = math.atan2(dy, dx) if (abs(dx) + abs(dy)) > 0.01 else 0.0

                    frame_img = build_debug_frame(
                        cam_imgs=cam_imgs,
                        waypoints=cmd.waypoints,
                        trajectories=self._all_trajectories,
                        sim_frame=self._sim_frame,
                        robot_pos=pos,
                        robot_yaw=yaw,
                        robot_history=self._robot_history,
                        cmd_linear=cmd.linear_x,
                        cmd_angular=cmd.angular_z,
                        confidence=cmd.confidence,
                    )
                    cv2.imwrite(
                        os.path.join(frames_dir, f"frame_{_frame_idx:06d}.jpg"),
                        cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR),
                    )
                    _frame_idx += 1
                except Exception as _fe:
                    import traceback
                    if self._sim_frame <= 4:
                        traceback.print_exc()

            metrics.append({
                "step": self._sim_frame,
                "linear_x": cmd.linear_x,
                "angular_z": cmd.angular_z,
                "confidence": cmd.confidence,
            })

            if self._sim_frame % 10 == 0:
                elapsed = time.perf_counter() - _t0
                sps = self._sim_frame / elapsed if elapsed > 0 else 0
                print(
                    f"  step {self._sim_frame:>6}  "
                    f"v={cmd.linear_x:.2f}m/s  "
                    f"ω={cmd.angular_z:.3f}rad/s  "
                    f"conf={cmd.confidence:.2f}  "
                    f"{step_ms:.0f}ms/step  "
                    f"{sps:.1f}sps  "
                    f"{_gpu_mb()}",
                    flush=True,
                )

        self._save_metrics(metrics)

        if save_video and os.path.isdir(frames_dir):
            video_path = os.path.join(out_dir, "simulation.mp4")
            fps = log_cfg.get("video_fps", 10)
            # Use subprocess so the shell never touches the path
            import subprocess as _sp
            ffmpeg_args = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(frames_dir, "frame_%06d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                video_path,
            ]
            print(f"[WasteESimEnv] Stitching video → {video_path}", flush=True)
            ret = _sp.run(ffmpeg_args, capture_output=True)
            if ret.returncode == 0:
                print(f"[WasteESimEnv] Video saved: {video_path}")
            else:
                print(f"[WasteESimEnv] ffmpeg failed — frames kept in {frames_dir}/")
                print(ret.stderr.decode()[-500:])

        print("[WasteESimEnv] Simulation finished.")

    def _save_metrics(self, metrics: list):
        import csv
        log_cfg = self.cfg.get("logging", {})
        out_dir = log_cfg.get("output_dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "metrics.csv")
        if not metrics:
            return
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)
        print(f"[WasteESimEnv] Metrics saved to {out_path}")
