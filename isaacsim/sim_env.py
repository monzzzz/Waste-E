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
        self._sim_frame = 0

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
        self._spawn_cameras()
        self._spawn_pedestrians()
        self._load_alpamayo()

        self.world.reset()
        self.world.play()
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
        for name, ccfg in cam_cfg.items():
            try:
                cam = Camera(
                    prim_path=f"/World/WasteE/{name}_camera",
                    name=f"{name}_camera",
                    position=np.array(ccfg["position"], dtype=float),
                    resolution=(ccfg["resolution"][0], ccfg["resolution"][1]),
                    frequency=30,
                )
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

    # ------------------------------------------------------------------
    def _get_camera_image(self, name: str) -> Optional[np.ndarray]:
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

    def step(self):
        self.world.step(render=True)

        for agent in self.pedestrian_agents:
            agent.update(self._sim_frame)

        front = self._get_camera_image("front")
        left  = self._get_camera_image("left")
        right = self._get_camera_image("right")

        cmd = self.alpamayo.infer(front, left, right)
        self._apply_drive_command(cmd)

        self._sim_frame += 1
        return cmd

    def run(self):
        max_steps = self.cfg["simulation"]["max_steps"]
        print(f"[WasteESimEnv] Running for up to {max_steps} steps …")

        metrics = []
        while self.world.is_playing() and self._sim_frame < max_steps:
            cmd = self.step()
            metrics.append({
                "step": self._sim_frame,
                "linear_x": cmd.linear_x,
                "angular_z": cmd.angular_z,
                "confidence": cmd.confidence,
            })

            if self._sim_frame % 300 == 0:
                print(
                    f"  step {self._sim_frame:>6}  "
                    f"v={cmd.linear_x:.2f} m/s  "
                    f"ω={cmd.angular_z:.3f} rad/s  "
                    f"conf={cmd.confidence:.2f}"
                )

        self._save_metrics(metrics)
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
