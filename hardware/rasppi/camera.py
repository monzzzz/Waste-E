from __future__ import annotations

import fcntl
import glob
import json
import os
import re
import struct
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

MEDIAMTX_BIN         = os.getenv("MEDIAMTX_BIN", str(Path(__file__).parent / "mediamtx"))
MEDIAMTX_RTSP_PORT   = int(os.getenv("MEDIAMTX_RTSP_PORT", "8554"))
MEDIAMTX_WEBRTC_PORT = int(os.getenv("MEDIAMTX_WEBRTC_PORT", "8889"))
H264_ENCODER         = os.getenv("H264_ENCODER", "libx264")
WEBRTC_MODE          = os.path.isfile(MEDIAMTX_BIN) and os.access(MEDIAMTX_BIN, os.X_OK)

_VIDIOC_QUERYCAP        = 0x80685600
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_USB_IFACE_RE           = re.compile(r"^\d+-[\d.]+:\d+\.\d+$")
_CAM_CACHE_TTL          = 10.0


class CameraManager:
    def __init__(self, max_cameras: int, cam_port: int) -> None:
        self.max_cameras  = max_cameras
        self.cam_port     = cam_port
        self.webrtc_mode  = WEBRTC_MODE
        self.rtsp_port    = MEDIAMTX_RTSP_PORT
        self.webrtc_port  = MEDIAMTX_WEBRTC_PORT
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock        = threading.Lock()
        self._cache: list[str] = []
        self._cache_ts: float  = 0.0

    def discover(self) -> list[str]:
        return _discover_camera_names(self.max_cameras)

    def start_mediamtx(self) -> Optional[subprocess.Popen]:
        if not self.webrtc_mode:
            return None
        cfg = Path(MEDIAMTX_BIN).parent / "mediamtx.yml"
        cmd = [MEDIAMTX_BIN] + ([str(cfg)] if cfg.exists() else [])
        print("[RasPi] Starting MediaMTX")
        return subprocess.Popen(
            cmd, cwd=str(Path(MEDIAMTX_BIN).parent),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def get_or_start_ffmpeg(self, dev_name: str) -> subprocess.Popen:
        with self._lock:
            p = self._procs.get(dev_name)
            if p is None or p.poll() is not None:
                print(f"[RasPi] Starting ffmpeg → RTSP for /dev/{dev_name}")
                self._procs[dev_name] = self._start_ffmpeg(dev_name)
            return self._procs[dev_name]

    def _start_ffmpeg(self, dev_name: str) -> subprocess.Popen:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-f", "v4l2", "-input_format", "mjpeg",
            "-framerate", "15", "-video_size", "640x480",
            "-i", f"/dev/{dev_name}",
        ]
        if H264_ENCODER == "libx264":
            cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency"]
        else:
            cmd += ["-c:v", H264_ENCODER, "-vf", "format=yuv420p"]
        cmd += [
            "-b:v", "800k", "-g", "5",
            "-f", "rtsp", "-rtsp_transport", "tcp",
            f"rtsp://127.0.0.1:{self.rtsp_port}/{dev_name}",
        ]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0)

    def start_watchdog(self, names: list[str]) -> None:
        def _loop():
            while True:
                time.sleep(5)
                for name in names:
                    if os.path.exists(f"/dev/{name}"):
                        self.get_or_start_ffmpeg(name)
        threading.Thread(target=_loop, daemon=True).start()

    def fetch_local_names(self) -> list[str]:
        if self.webrtc_mode:
            return self.discover()

        now = time.monotonic()
        if now - self._cache_ts < _CAM_CACHE_TTL and self._cache:
            return self._cache

        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.cam_port}/api/cameras", timeout=2
            ) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            names: list[str] = []
            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    if not bool(item.get("open", True)):
                        continue
                    device = item.get("device")
                    if device:
                        names.append(Path(str(device)).name)
            if names:
                self._cache    = names[:self.max_cameras]
                self._cache_ts = now
                return self._cache
        except Exception:
            pass

        result = self.discover()
        if result:
            self._cache    = result
            self._cache_ts = now
        return result


def _discover_camera_names(max_cameras: int) -> list[str]:
    devices = sorted(
        (Path(p) for p in glob.glob("/dev/video*")),
        key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else 9999,
    )
    seen_usb: set[str] = set()
    result: list[str]  = []
    for dev in devices:
        if not dev.exists():
            continue
        key = _usb_parent_key(dev.name)
        if key is None or not _has_video_capture_cap(dev.name):
            continue
        if key in seen_usb:
            continue
        seen_usb.add(key)
        result.append(dev.name)
        if len(result) >= max_cameras:
            break
    return result


def _usb_parent_key(video_name: str) -> str | None:
    try:
        real = Path(f"/sys/class/video4linux/{video_name}").resolve()
        for i, part in enumerate(real.parts):
            if _USB_IFACE_RE.match(part):
                return str(Path(*real.parts[:i]))
        return None
    except Exception:
        return None


def _has_video_capture_cap(video_name: str) -> bool:
    try:
        fd = os.open(f"/dev/{video_name}", os.O_RDONLY | os.O_NONBLOCK)
        try:
            buf = bytearray(104)
            fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
            return bool(struct.unpack_from("<I", buf, 88)[0] & _V4L2_CAP_VIDEO_CAPTURE)
        finally:
            os.close(fd)
    except Exception:
        return False
