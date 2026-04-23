#!/usr/bin/env python3
"""
Raspberry Pi data sender for Waste-E.

Starts the local camera server on the RasPi and registers the device with the
central dashboard server.
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

CAM_PORT = int(os.getenv("CAM_PORT", "5000"))
SEND_HZ = 2.0
RETRY_SECONDS = 5.0
MAX_CAMERAS = 3
_CAM_CACHE_TTL = 10.0


def discover_camera_names(max_cameras: int = MAX_CAMERAS) -> list[str]:
    devices = sorted(
        (Path(p) for p in glob.glob("/dev/video*")),
        key=lambda p: int(p.name[5:]) if p.name.startswith("video") and p.name[5:].isdigit() else 9999,
    )
    return [p.name for p in devices[:max_cameras] if p.exists()]


def start_camera_server(script_path: Path, cam_port: int) -> subprocess.Popen:
    if not script_path.exists():
        raise FileNotFoundError(f"Camera stream script not found: {script_path}")

    env = os.environ.copy()
    env["CAM_PORT"] = str(cam_port)

    return subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        stdout=None,
        stderr=None,
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
        }
        try:
            poster.post("/api/register", registration)
            print(f"[RasPi sender] Registered with dashboard server: {registration}")
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
                        help="Path to the local camera stream script")
    parser.add_argument("--max-cameras", type=int, default=MAX_CAMERAS,
                        help=f"Maximum number of camera feeds to register (default {MAX_CAMERAS})")
    args = parser.parse_args()

    script_path = Path(__file__).parent / args.camera_script
    camera_names = discover_camera_names(max_cameras=args.max_cameras)

    if not camera_names:
        print("WARNING: no /dev/video* devices found. The camera server may still start.")

    print(f"[RasPi] Starting camera server from {script_path}")
    camera_proc = start_camera_server(script_path, args.cam_port)
    print(f"[RasPi] Camera server should be available on port {args.cam_port}")
    time.sleep(1.5)
    camera_names = _fetch_local_camera_names(args.cam_port, max_cameras=args.max_cameras)
    print(f"[RasPi] Registered camera list: {camera_names}")

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

