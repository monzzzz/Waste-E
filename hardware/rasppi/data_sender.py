#!/usr/bin/env python3
"""
Raspberry Pi data sender for Waste-E.

WebRTC mode (when mediamtx binary is present):
  Starts MediaMTX + ffmpeg per camera → RTSP → browser via WHEP.

MJPEG fallback (no mediamtx binary):
  Starts camera_stream.py and serves MJPEG over HTTP.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import http.client
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

CAM_PORT = int(os.getenv("CAM_PORT", "5000"))
SEND_HZ = 2.0
RETRY_SECONDS = 5.0
MAX_CAMERAS = 3
_CAM_CACHE_TTL = 10.0

MEDIAMTX_BIN = os.getenv("MEDIAMTX_BIN", str(Path(__file__).parent / "mediamtx"))
MEDIAMTX_RTSP_PORT = int(os.getenv("MEDIAMTX_RTSP_PORT", "8554"))
MEDIAMTX_WEBRTC_PORT = int(os.getenv("MEDIAMTX_WEBRTC_PORT", "8889"))
H264_ENCODER = os.getenv("H264_ENCODER", "h264_v4l2m2m")
_WEBRTC_MODE = os.path.isfile(MEDIAMTX_BIN) and os.access(MEDIAMTX_BIN, os.X_OK)

_CAM_PROCS: dict[str, subprocess.Popen] = {}
_CAM_LOCK = threading.Lock()


_VIDIOC_QUERYCAP = 0x80685600  # aarch64 _IOR('V', 0, 104)
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_USB_IFACE_RE = re.compile(r"^\d+-[\d.]+:\d+\.\d+$")  # e.g. "1-1.1:1.0"


def _usb_parent_key(video_name: str) -> str | None:
    """Return the USB device sysfs path (without interface suffix) for deduplication."""
    try:
        real = Path(f"/sys/class/video4linux/{video_name}").resolve()
        for i, part in enumerate(real.parts):
            if _USB_IFACE_RE.match(part):  # USB interface, not PCI addr like "0000:01:00.0"
                return str(Path(*real.parts[:i]))
        return None
    except Exception:
        return None


def _has_video_capture_cap(video_name: str) -> bool:
    """Check VIDEO_CAPTURE capability via ioctl — works even when device is in use."""
    try:
        fd = os.open(f"/dev/{video_name}", os.O_RDONLY | os.O_NONBLOCK)
        try:
            buf = bytearray(104)
            fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
            device_caps = struct.unpack_from("<I", buf, 88)[0]
            return bool(device_caps & _V4L2_CAP_VIDEO_CAPTURE)
        finally:
            os.close(fd)
    except Exception:
        return False


def discover_camera_names(max_cameras: int = MAX_CAMERAS) -> list[str]:
    devices = sorted(
        (Path(p) for p in glob.glob("/dev/video*")),
        key=lambda p: int(p.name[5:]) if p.name.startswith("video") and p.name[5:].isdigit() else 9999,
    )
    seen_usb: set[str] = set()
    result: list[str] = []
    for dev in devices:
        if not dev.exists():
            continue
        key = _usb_parent_key(dev.name)
        if key is None:  # not a USB device (e.g. bcm2835 encoder)
            continue
        if not _has_video_capture_cap(dev.name):
            continue
        if key in seen_usb:
            continue
        seen_usb.add(key)
        result.append(dev.name)
        if len(result) >= max_cameras:
            break
    return result


def _start_mediamtx() -> Optional[subprocess.Popen]:
    if not _WEBRTC_MODE:
        return None
    cfg = Path(MEDIAMTX_BIN).parent / "mediamtx.yml"
    cmd = [MEDIAMTX_BIN] + ([str(cfg)] if cfg.exists() else [])
    print(f"[RasPi] Starting MediaMTX ({MEDIAMTX_BIN})")
    return subprocess.Popen(
        cmd, cwd=str(Path(MEDIAMTX_BIN).parent),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _start_ffmpeg(dev_name: str) -> subprocess.Popen:
    dev = f"/dev/{dev_name}"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", "v4l2", "-input_format", "mjpeg",
        "-framerate", "15",
        "-video_size", "1280x720",
        "-i", dev,
    ]
    if H264_ENCODER == "libx264":
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency"]
    else:
        cmd += ["-c:v", H264_ENCODER, "-vf", "format=yuv420p"]
    cmd += [
        "-b:v", "800k", "-g", "5",
        "-f", "rtsp", "-rtsp_transport", "tcp",
        f"rtsp://127.0.0.1:{MEDIAMTX_RTSP_PORT}/{dev_name}",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0)


def _get_ffmpeg_proc(dev_name: str) -> subprocess.Popen:
    with _CAM_LOCK:
        p = _CAM_PROCS.get(dev_name)
        if p is None or p.poll() is not None:
            print(f"[RasPi] Starting ffmpeg → RTSP for /dev/{dev_name}")
            _CAM_PROCS[dev_name] = _start_ffmpeg(dev_name)
        return _CAM_PROCS[dev_name]


def _ffmpeg_watchdog(camera_names: list[str]) -> None:
    """Restart dead ffmpeg processes in WebRTC mode."""
    while True:
        time.sleep(5)
        for name in camera_names:
            if os.path.exists(f"/dev/{name}"):
                _get_ffmpeg_proc(name)


def start_camera_server(script_path: Path, cam_port: int) -> subprocess.Popen:
    if not script_path.exists():
        raise FileNotFoundError(f"Camera stream script not found: {script_path}")
    env = os.environ.copy()
    env["CAM_PORT"] = str(cam_port)
    return subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        stdout=None, stderr=None,
        env=env,
    )


class _PersistentPoster:
    """Reuses a single TCP connection for repeated POSTs to the same host."""

    def __init__(self, server_url: str, timeout: float = 5.0):
        parsed = urllib.parse.urlsplit(server_url)
        self._host = parsed.netloc
        self._https = parsed.scheme == "https"
        self._timeout = timeout
        self._conn: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        if self._https:
            return http.client.HTTPSConnection(self._host, timeout=self._timeout)
        return http.client.HTTPConnection(self._host, timeout=self._timeout)

    def post(self, path: str, payload: dict) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        for attempt in range(2):
            try:
                if self._conn is None:
                    self._conn = self._connect()
                self._conn.request(
                    "POST", path, body=data,
                    headers={"Content-Type": "application/json", "Connection": "keep-alive"},
                )
                self._conn.getresponse().read()
                return
            except Exception:
                self._conn = None
                if attempt == 0:
                    continue
                raise


_cam_cache: list[str] = []
_cam_cache_ts: float = 0.0


def _fetch_local_camera_names(cam_port: int, max_cameras: int = MAX_CAMERAS) -> list[str]:
    """In WebRTC mode skip the HTTP round-trip and just probe /dev/video*."""
    if _WEBRTC_MODE:
        return discover_camera_names(max_cameras=max_cameras)

    global _cam_cache, _cam_cache_ts
    now = time.monotonic()
    if now - _cam_cache_ts < _CAM_CACHE_TTL and _cam_cache:
        return _cam_cache
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{cam_port}/api/cameras", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        names: list[str] = []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                if not bool(item.get("open", True)):
                    continue
                device = item.get("device")
                if not device:
                    continue
                names.append(Path(str(device)).name)
        if names:
            _cam_cache = names[:max_cameras]
            _cam_cache_ts = now
            return _cam_cache
    except Exception:
        pass
    result = discover_camera_names(max_cameras=max_cameras)
    if result:
        _cam_cache = result
        _cam_cache_ts = now
    return result


def register_and_send_loop(
    server_url: str,
    my_ip: str,
    cam_port: int,
    max_cameras: int = MAX_CAMERAS,
) -> None:
    poster = _PersistentPoster(server_url)

    while True:
        camera_names = _fetch_local_camera_names(cam_port, max_cameras=max_cameras)
        registration = {
            "device": "rasppi",
            "ip": my_ip,
            "cam_port": cam_port,
            "cameras": camera_names,
            "webrtc_port": MEDIAMTX_WEBRTC_PORT if _WEBRTC_MODE else None,
        }
        try:
            poster.post("/api/register", registration)
            print(f"[RasPi sender] Registered: {registration}")
        except Exception as exc:
            print(f"[RasPi sender] Registration failed: {exc} — retrying in {RETRY_SECONDS}s")
            time.sleep(RETRY_SECONDS)
            continue

        while True:
            try:
                camera_names = _fetch_local_camera_names(cam_port, max_cameras=max_cameras)
                payload = {
                    "device": "rasppi",
                    "ts": time.time(),
                    "cam_port": cam_port,
                    "cameras": camera_names,
                    "webrtc_port": MEDIAMTX_WEBRTC_PORT if _WEBRTC_MODE else None,
                }
                poster.post("/api/data", payload)
            except Exception as exc:
                print(f"[RasPi sender] Data POST failed: {exc} — will re-register")
                break
            time.sleep(1.0 / SEND_HZ)


# ── Arm control (Pi0.5 inference via RunPod) ────────────────────────────────

_ARM_JOINT_ORDER = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
_ARM_CAM_LABELS = ["front", "top", "wrist"]
_ARM_ACTIONS_PER_PRED = 20


def _encode_frame_jpeg(frame) -> str:
    """Encode a BGR numpy frame to a base64 JPEG string."""
    import base64
    import io
    from PIL import Image
    import numpy as np
    rgb = frame[..., ::-1].astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _open_arm_camera(dev_name: str, cam_port: int):
    """Return a cv2.VideoCapture reading from the already-running stream.

    WebRTC mode  → RTSP from MediaMTX (e.g. rtsp://127.0.0.1:8554/video0)
    MJPEG mode   → HTTP from camera_stream.py (e.g. http://127.0.0.1:5000/video_feed/0)
    The device is already held by ffmpeg/camera_stream.py so we never open
    /dev/videoN directly here.
    """
    import cv2
    if _WEBRTC_MODE:
        url = f"rtsp://127.0.0.1:{MEDIAMTX_RTSP_PORT}/{dev_name}"
    else:
        idx = int(dev_name.replace("video", "") or 0)
        url = f"http://127.0.0.1:{cam_port}/video_feed/{idx}"
    cap = cv2.VideoCapture(url)
    return cap


def arm_control_loop(
    runpod_url: str,
    arm_port: str,
    task: str,
    arm_fps: int,
    cam_port: int,
    arm_cam_devices: list[str],   # device names in front/top/wrist order, e.g. ["video0","video2","video6"]
) -> None:
    """Inference loop: grab frames → send to RunPod → execute actions on arm.

    Runs as a daemon thread alongside the dashboard registration loop.
    Imports all heavy dependencies lazily so the normal data-sender path
    still works on machines that don't have lerobot/cv2 installed.
    """
    import numpy as np
    import requests

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[Arm] opencv-python not installed — arm control disabled")
        return

    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        print("[Arm] lerobot not installed — arm control disabled")
        return

    # ── Health check ──────────────────────────────────────────────────────
    try:
        health = requests.get(f"{runpod_url}/health", timeout=10).json()
        print(f"[Arm] RunPod server OK: {health}")
    except Exception as exc:
        print(f"[Arm] Cannot reach RunPod at {runpod_url}: {exc}")
        return

    # ── Connect arm (no lerobot cameras — we handle those below) ──────────
    print(f"[Arm] Connecting SO101 arm on {arm_port} …")
    robot = SO101Follower(SO101FollowerConfig(port=arm_port, id="waste_e_arm", cameras={}))
    robot.connect()
    if not robot.is_connected:
        print("[Arm] Arm failed to connect — aborting arm control")
        return
    print("[Arm] Arm connected")

    # ── Open camera streams (wait for MediaMTX / camera_stream.py to be up) ──
    time.sleep(3.0)
    caps: dict[str, object] = {}
    for i, dev in enumerate(arm_cam_devices[:3]):
        label = _ARM_CAM_LABELS[i]
        cap = _open_arm_camera(dev, cam_port)
        if cap.isOpened():
            caps[label] = cap
            print(f"[Arm] Camera '{label}' ({dev}) ready")
        else:
            print(f"[Arm] Camera '{label}' ({dev}) could not be opened — continuing without it")

    if not caps:
        print("[Arm] No arm cameras available — aborting arm control")
        robot.disconnect()
        return

    last_actions: Optional[object] = None
    action_index = 0
    step = 0

    try:
        while True:
            t0 = time.perf_counter()
            need_pred = last_actions is None or action_index >= _ARM_ACTIONS_PER_PRED

            if need_pred:
                # Joint state
                obs = robot.get_observation()
                missing = [k for k in _ARM_JOINT_ORDER if k not in obs]
                if missing:
                    print(f"[Arm] Missing joints {missing} — skipping prediction")
                    time.sleep(1.0 / arm_fps)
                    continue
                state = [float(obs[k]) for k in _ARM_JOINT_ORDER]

                # Camera frames
                images_b64: dict[str, str] = {}
                for label, cap in caps.items():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        images_b64[label] = _encode_frame_jpeg(frame)

                if not images_b64:
                    print("[Arm] All camera reads failed — retrying")
                    time.sleep(1.0 / arm_fps)
                    continue

                # Inference request to RunPod
                try:
                    resp = requests.post(
                        f"{runpod_url}/predict",
                        json={"prompt": task, "state": state, "images": images_b64},
                        timeout=30,
                        headers={"Content-Type": "application/json"},
                    )
                    result = resp.json()
                    if not result.get("success"):
                        print(f"[Arm] Prediction failed: {result.get('error')}")
                        time.sleep(1.0 / arm_fps)
                        continue
                    last_actions = np.array(result["actions"], dtype=np.float32)
                    action_index = 0
                    print(f"[Arm] Step {step}: received {len(last_actions)} actions "
                          f"({len(images_b64)} cams, state={[round(v,2) for v in state]})")
                except Exception as exc:
                    print(f"[Arm] RunPod request failed: {exc} — retrying")
                    time.sleep(1.0 / arm_fps)
                    continue

            # Execute next action
            action = last_actions[action_index]
            robot.send_action({_ARM_JOINT_ORDER[i]: float(action[i])
                               for i in range(len(_ARM_JOINT_ORDER))})
            action_index += 1
            step += 1

            dt = time.perf_counter() - t0
            time.sleep(max(0.0, (1.0 / arm_fps) - dt))

    except Exception as exc:
        print(f"[Arm] Loop crashed: {exc}")
    finally:
        print("[Arm] Disconnecting arm …")
        robot.disconnect()
        for cap in caps.values():
            cap.release()
        print("[Arm] Arm shut down")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi data sender")
    parser.add_argument("--server", default="http://192.168.112.254:9000",
                        help="Central dashboard server URL")
    parser.add_argument("--my-ip", default="192.168.1.11",
                        help="This device's IP address")
    parser.add_argument("--cam-port", type=int, default=CAM_PORT,
                        help=f"This device's local camera server port (default {CAM_PORT})")
    parser.add_argument("--camera-script", default="camera_stream.py",
                        help="Path to the MJPEG fallback camera stream script")
    parser.add_argument("--max-cameras", type=int, default=MAX_CAMERAS,
                        help=f"Maximum number of camera feeds to register (default {MAX_CAMERAS})")
    # ── Arm / inference args (all optional) ──────────────────────────────
    parser.add_argument("--runpod-url", default=None,
                        help="RunPod inference server URL, e.g. https://xxxxx-8001.proxy.runpod.net")
    parser.add_argument("--arm-port", default="/dev/ttyACM0",
                        help="Serial port for SO101 arm (default /dev/ttyACM0)")
    parser.add_argument("--arm-task",
                        default="Pick an object in front of you and place it in the bin behind you.",
                        help="Task description sent to the model")
    parser.add_argument("--arm-fps", type=int, default=10,
                        help="Arm control loop FPS (default 10)")
    parser.add_argument("--arm-cameras", default=None,
                        help="Camera device names for the arm in front,top,wrist order, "
                             "comma-separated (e.g. video0,video2,video6). "
                             "Defaults to the first 3 discovered cameras.")
    args = parser.parse_args()

    camera_names = discover_camera_names(max_cameras=args.max_cameras)
    if not camera_names:
        print("WARNING: no /dev/video* devices found.")

    camera_proc = None

    if _WEBRTC_MODE:
        print(f"[RasPi] WebRTC mode — MediaMTX RTSP:{MEDIAMTX_RTSP_PORT} WebRTC:{MEDIAMTX_WEBRTC_PORT}")
        mediamtx_proc = _start_mediamtx()
        time.sleep(1.5)
        for name in camera_names:
            _get_ffmpeg_proc(name)
        threading.Thread(target=_ffmpeg_watchdog, args=(camera_names,), daemon=True).start()
    else:
        print(f"[RasPi] MJPEG mode — starting camera_stream.py")
        script_path = Path(__file__).parent / args.camera_script
        camera_proc = start_camera_server(script_path, args.cam_port)
        time.sleep(1.5)

    print(f"[RasPi] Cameras: {camera_names}")
    print(f"[RasPi] Central server: {args.server}")

    # ── Start arm inference thread if RunPod URL is given ─────────────────
    if args.runpod_url:
        if args.arm_cameras:
            arm_cam_devices = [s.strip() for s in args.arm_cameras.split(",")]
        else:
            arm_cam_devices = camera_names[:3]

        print(f"[RasPi] Starting arm control → {args.runpod_url}")
        print(f"[RasPi] Arm cameras: {arm_cam_devices}  port: {args.arm_port}  fps: {args.arm_fps}")
        threading.Thread(
            target=arm_control_loop,
            args=(
                args.runpod_url,
                args.arm_port,
                args.arm_task,
                args.arm_fps,
                args.cam_port,
                arm_cam_devices,
            ),
            daemon=True,
        ).start()

    try:
        thread = threading.Thread(
            target=register_and_send_loop,
            args=(args.server, args.my_ip, args.cam_port, args.max_cameras),
            daemon=True,
        )
        thread.start()
        thread.join()
    except KeyboardInterrupt:
        print("[RasPi] Interrupted, shutting down")
    finally:
        if camera_proc and camera_proc.poll() is None:
            camera_proc.terminate()
            camera_proc.wait(timeout=5)
