#!/usr/bin/env python3
"""
Raspberry Pi data sender for Waste-E.

Starts the local camera server on the RasPi and registers the device with the
central dashboard server.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

CAM_PORT = int(os.getenv("CAM_PORT", "5000"))
SEND_HZ = 2.0
RETRY_SECONDS = 5.0


def discover_camera_names(max_cameras: int = 3) -> list[str]:
    devices = sorted(
        (Path(p) for p in glob.glob("/dev/video*")),
        key=lambda p: int(p.name.replace("video", "")) if p.name.startswith("video") else 9999,
    )
    return [p.name for p in devices[:max_cameras] if p.exists()]


def start_camera_server(script_path: Path) -> subprocess.Popen:
    if not script_path.exists():
        raise FileNotFoundError(f"Camera stream script not found: {script_path}")

    return subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        stdout=None,
        stderr=None,
        env=os.environ.copy(),
    )


def post_json(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()


def register_and_send_loop(server_url: str, my_ip: str, cam_port: int, camera_names: list[str]) -> None:
    registration = {
        "device": "rasppi",
        "ip": my_ip,
        "cam_port": cam_port,
        "cameras": camera_names,
    }

    while True:
        try:
            post_json(f"{server_url.rstrip('/')}/api/register", registration)
            print(f"[RasPi sender] Registered with dashboard server: {registration}")
        except Exception as exc:
            print(f"[RasPi sender] Registration failed: {exc} — retrying in {RETRY_SECONDS}s")
            time.sleep(RETRY_SECONDS)
            continue

        while True:
            try:
                payload = {
                    "device": "rasppi",
                    "ts": time.time(),
                    "sensors": {
                        "camera_count": len(camera_names),
                        "camera_names": camera_names,
                    },
                }
                post_json(f"{server_url.rstrip('/')}/api/data", payload)
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
    args = parser.parse_args()

    script_path = Path(__file__).parent / args.camera_script
    camera_names = discover_camera_names()

    if not camera_names:
        print("WARNING: no /dev/video* devices found. The camera server may still start.")

    print(f"[RasPi] Starting camera server from {script_path}")
    camera_proc = start_camera_server(script_path)
    print(f"[RasPi] Camera server should be available on port {args.cam_port}")

    try:
        thread = threading.Thread(
            target=register_and_send_loop,
            args=(args.server, args.my_ip, args.cam_port, camera_names),
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
