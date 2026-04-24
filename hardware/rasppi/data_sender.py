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
import glob
import http.client
import json
import os
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


def discover_camera_names(max_cameras: int = MAX_CAMERAS) -> list[str]:
    devices = sorted(
        (Path(p) for p in glob.glob("/dev/video*")),
        key=lambda p: int(p.name[5:]) if p.name.startswith("video") and p.name[5:].isdigit() else 9999,
    )
    return [p.name for p in devices[:max_cameras] if p.exists()]


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
        "-f", "v4l2", "-input_format", "mjpeg",
        "-framerate", "15",
        "-video_size", "1280x720",
        "-i", dev,
    ]
    if H264_ENCODER == "libx264":
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency"]
    else:
        cmd += ["-c:v", H264_ENCODER]
    cmd += [
        "-b:v", "800k", "-g", "30",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi data sender")
    parser.add_argument("--server", default="http://192.168.1.100:9000",
                        help="Central dashboard server URL")
    parser.add_argument("--my-ip", default="192.168.1.11",
                        help="This device's IP address")
    parser.add_argument("--cam-port", type=int, default=CAM_PORT,
                        help=f"This device's local camera server port (default {CAM_PORT})")
    parser.add_argument("--camera-script", default="camera_stream.py",
                        help="Path to the MJPEG fallback camera stream script")
    parser.add_argument("--max-cameras", type=int, default=MAX_CAMERAS,
                        help=f"Maximum number of camera feeds to register (default {MAX_CAMERAS})")
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
