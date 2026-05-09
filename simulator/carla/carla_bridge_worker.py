"""
CARLA bridge worker — runs under Python 3.7 (with CARLA 0.9.15 egg).

Protocol: newline-delimited JSON on stdin/stdout.

Init:
  → {"host": "localhost", "port": 2000, "map": "Town03",
     "num_peds": 30, "img_w": 640, "img_h": 384}
  ← {"status": "ready", "map": "..."}

Per step:
  → {"linear_x": 0.5, "angular_z": 0.1}
  ← {"status": "ok",
     "pos": [x, y, z], "yaw_deg": 0.0,
     "left_img": "<base64-jpg>",
     "front_img": "<base64-jpg>",
     "right_img": "<base64-jpg>",
     "tele_img": "<base64-jpg>"}

Shutdown:
  → {"shutdown": true}
  ← {"status": "bye"}
"""

from __future__ import annotations

import base64
import json
import math
import sys
import time
from typing import Dict, List, Optional

import numpy as np

CARLA_EGG = (
    "/workspace/carla_server/PythonAPI/carla/dist/"
    "carla-0.9.15-py3.7-linux-x86_64.egg"
)
sys.path.insert(0, CARLA_EGG)

import carla


IMG_W = 640
IMG_H = 384

# Camera rig: matching Alpamayo training set
# left (+45° yaw), front (0°), right (-45° yaw), tele (0°, narrow FOV)
# CARLA uses a left-handed coordinate system: x=forward, y=RIGHT, z=up.
# Positive yaw rotates clockwise (to the right) when viewed from above.
# So left camera must be at y=-0.5 (left of centre) with yaw=-45° (pointing left-forward).
_CAM_CONFIGS = {
    "left":         {"x":  0.5, "y": -0.5, "z": 0.5, "pitch":   0.0, "yaw": -60.0, "fov": 110.0},
    "front":        {"x":  0.5, "y":  0.0, "z": 0.5, "pitch":   0.0, "yaw":   0.0, "fov": 110.0},
    "right":        {"x":  0.5, "y":  0.5, "z": 0.5, "pitch":   0.0, "yaw":  60.0, "fov": 110.0},
    "tele":         {"x":  0.5, "y":  0.0, "z": 0.5, "pitch":   0.0, "yaw":   0.0, "fov":  30.0},
    "third_person": {"x": -5.0, "y":  0.0, "z": 4.0, "pitch": -20.0, "yaw":   0.0, "fov":  90.0},
}


class ImageBuffer:
    def __init__(self):
        self._data: Dict[str, Optional[bytes]] = {k: None for k in _CAM_CONFIGS}

    def set(self, name: str, image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((IMG_H, IMG_W, 4))[:, :, :3]  # BGRA → BGR (numpy)
        import cv2
        jpg = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        self._data[name] = jpg

    def get_b64(self, name: str) -> Optional[str]:
        d = self._data.get(name)
        if d is None:
            return None
        return base64.b64encode(d).decode("ascii")


def attach_cameras(world, vehicle, buf: ImageBuffer):
    bp_lib = world.get_blueprint_library()
    sensors = []
    for name, cfg in _CAM_CONFIGS.items():
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(IMG_W))
        bp.set_attribute("image_size_y", str(IMG_H))
        bp.set_attribute("fov", str(cfg["fov"]))
        tf = carla.Transform(
            carla.Location(x=cfg["x"], y=cfg["y"], z=cfg["z"]),
            carla.Rotation(pitch=cfg.get("pitch", 0.0), yaw=cfg["yaw"]),
        )
        sensor = world.spawn_actor(bp, tf, attach_to=vehicle)

        def _cb(img, n=name):
            buf.set(n, img)

        sensor.listen(_cb)
        sensors.append(sensor)
    return sensors


def spawn_pedestrians(world, n: int) -> list:
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    ctrl_bp = bp_lib.find("controller.ai.walker")
    pairs = []
    for _ in range(n * 3):
        if len(pairs) >= n:
            break
        try:
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = walker_bps[np.random.randint(len(walker_bps))]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = world.try_spawn_actor(bp, carla.Transform(loc))
            if walker is None:
                continue
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
            world.tick()
            ctrl.start()
            ctrl.go_to_location(world.get_random_location_from_navigation())
            ctrl.set_max_speed(1.0 + np.random.uniform(-0.2, 0.2))
            pairs.append((walker, ctrl))
        except Exception:
            pass
    return pairs


def main():
    # ── init ─────────────────────────────────────────────────────────────────
    raw = sys.stdin.readline()
    cfg = json.loads(raw)

    host = cfg.get("host", "localhost")
    port = cfg.get("port", 2000)
    map_name = cfg.get("map", "Town03")
    num_peds = cfg.get("num_peds", 30)

    client = carla.Client(host, port)
    client.set_timeout(30.0)

    world = client.load_world(map_name)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 20.0
    world.apply_settings(settings)

    weather = carla.WeatherParameters(
        cloudiness=10.0, precipitation=0.0, sun_altitude_angle=60.0
    )
    world.set_weather(weather)

    # ── Spawn ego walker (robot proxy) ────────────────────────────────────────
    # A walker actor responds directly to WalkerControl(direction, speed) —
    # no static-friction issue, no PID needed, exact speed tracking.
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    ego_bp = walker_bps[0]  # any pedestrian mesh works as a robot proxy
    if ego_bp.has_attribute("is_invincible"):
        ego_bp.set_attribute("is_invincible", "true")

    # Spawn on pedestrian navmesh (sidewalk), not on the road.
    # get_random_location_from_navigation() returns points on the pedestrian navmesh.
    import math as _math
    ego = None
    ego_tf = None
    for _attempt in range(20):
        nav_loc = world.get_random_location_from_navigation()
        if nav_loc is None:
            continue
        candidate_tf = carla.Transform(
            carla.Location(x=nav_loc.x, y=nav_loc.y, z=nav_loc.z + 0.5),
            carla.Rotation(yaw=float(cfg.get("spawn_yaw", 0.0))),
        )
        actor = world.try_spawn_actor(ego_bp, candidate_tf)
        if actor is not None:
            ego = actor
            ego_tf = candidate_tf
            break

    if ego is None:
        # Last-resort fallback: road spawn offset slightly to avoid vehicles
        sp = world.get_map().get_spawn_points()[cfg.get("spawn_idx", 5) % len(world.get_map().get_spawn_points())]
        ego_tf = carla.Transform(
            carla.Location(x=sp.location.x, y=sp.location.y, z=sp.location.z + 0.5),
            sp.rotation,
        )
        ego = world.spawn_actor(ego_bp, ego_tf)
        print("[bridge] WARNING: navmesh spawn failed, fell back to road spawn", file=sys.stderr)

    print(f"Spawned ego walker at {ego_tf.location}", file=sys.stderr)

    # Track heading manually (WalkerControl takes a direction vector, not yaw)
    ego_yaw_deg = ego_tf.rotation.yaw  # initial heading

    # Cameras (attached to the walker)
    buf = ImageBuffer()
    sensors = attach_cameras(world, ego, buf)

    # Pedestrians
    pairs = spawn_pedestrians(world, num_peds)

    # Warm up — let actors settle
    for _ in range(10):
        world.tick()

    print(json.dumps({"status": "ready", "map": map_name}), flush=True)

    # ── step loop ─────────────────────────────────────────────────────────────
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)
        if data.get("shutdown"):
            break

        # ── respawn at a new random navmesh location ───────────────────────
        if data.get("respawn"):
            for sensor in sensors:
                sensor.stop()
                sensor.destroy()
            sensors = []
            for _attempt in range(20):
                nav_loc = world.get_random_location_from_navigation()
                if nav_loc is None:
                    continue
                new_tf = carla.Transform(
                    carla.Location(x=nav_loc.x, y=nav_loc.y, z=nav_loc.z + 0.5),
                    carla.Rotation(yaw=0.0),
                )
                ego.set_transform(new_tf)
                world.tick()
                ego_tf = new_tf
                ego_yaw_deg = 0.0
                break
            buf = ImageBuffer()
            sensors = attach_cameras(world, ego, buf)
            for _ in range(5):
                world.tick()
            pos = [ego_tf.location.x, ego_tf.location.y, ego_tf.location.z]
            print(json.dumps({"status": "respawned", "pos": pos, "yaw_deg": ego_yaw_deg}),
                  flush=True)
            continue

        linear_x  = float(data.get("linear_x", 0.0))
        angular_z = float(data.get("angular_z", 0.0))
        max_speed_ms = float(data.get("max_speed_ms", 1.5))

        speed = float(np.clip(linear_x, 0.0, max_speed_ms))

        # Update heading: angular_z (rad/s, right-handed) at 20 Hz
        # CARLA yaw: positive = clockwise (right), so subtract angular_z
        dt = 1.0 / 20.0
        ego_yaw_deg -= _math.degrees(angular_z) * dt
        yaw_rad = _math.radians(ego_yaw_deg)

        # Forward direction vector in CARLA frame (x=east, y=south)
        direction = carla.Vector3D(
            x=_math.cos(yaw_rad),
            y=_math.sin(yaw_rad),
            z=0.0,
        )
        ego.apply_control(carla.WalkerControl(
            direction=direction,
            speed=speed,
            jump=False,
        ))

        world.tick()

        # Pose after tick
        tf = ego.get_transform()
        pos = [tf.location.x, tf.location.y, tf.location.z]
        yaw_deg = tf.rotation.yaw  # report actual CARLA yaw
        vel = ego.get_velocity()
        spd = _math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        print(f"[bridge] spd={spd:.2f} tgt={speed:.2f} "
              f"ang={angular_z:.3f} yaw={ego_yaw_deg:.1f} "
              f"pos=({pos[0]:.1f},{pos[1]:.1f})", file=sys.stderr, flush=True)

        result = {
            "status": "ok",
            "pos": pos,
            "yaw_deg": ego_yaw_deg,
            "left_img":         buf.get_b64("left"),
            "front_img":        buf.get_b64("front"),
            "right_img":        buf.get_b64("right"),
            "tele_img":         buf.get_b64("tele"),
            "third_person_img": buf.get_b64("third_person"),
        }
        print(json.dumps(result), flush=True)

    # ── cleanup ───────────────────────────────────────────────────────────────
    for sensor in sensors:
        sensor.stop()
        sensor.destroy()
    for walker, ctrl in pairs:
        ctrl.stop()
        ctrl.destroy()
        walker.destroy()
    ego.destroy()

    settings.synchronous_mode = False
    world.apply_settings(settings)

    try:
        print(json.dumps({"status": "bye"}), flush=True)
    except BrokenPipeError:
        pass  # main process already exited


if __name__ == "__main__":
    main()
