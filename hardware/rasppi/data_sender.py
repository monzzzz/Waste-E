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
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request

CAM_PORT = int(os.getenv("CAM_PORT", "5000"))
SEND_HZ = 2.0
RETRY_SECONDS = 5.0
MAX_CAMERAS = 3
_CAM_CACHE_TTL = 10.0

MEDIAMTX_BIN = os.getenv("MEDIAMTX_BIN", str(Path(__file__).parent / "mediamtx"))
MEDIAMTX_RTSP_PORT = int(os.getenv("MEDIAMTX_RTSP_PORT", "8554"))
MEDIAMTX_WEBRTC_PORT = int(os.getenv("MEDIAMTX_WEBRTC_PORT", "8889"))
H264_ENCODER = os.getenv("H264_ENCODER", "libx264")
_WEBRTC_MODE = os.path.isfile(MEDIAMTX_BIN) and os.access(MEDIAMTX_BIN, os.X_OK)

_CAM_PROCS: dict[str, subprocess.Popen] = {}
_CAM_LOCK = threading.Lock()
_DEVICE_NAME = "rasppi"
LOCAL_RECORDINGS_DIR = Path(os.getenv("LOCAL_RECORDINGS_DIR", str(Path(__file__).parent / "recordings")))
CENTRAL_SERVER_URL = ""

_REC_LOCK = threading.Lock()
_REC_SESSION: dict | None = None


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
        "-video_size", "640x480",
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


def _safe_name(raw: str) -> str:
    cleaned = "".join(c for c in raw if c.isalnum() or c in "-_")
    return cleaned or "unnamed"


def _start_recording_ffmpeg(camera_name: str, dest: Path) -> subprocess.Popen:
    if _WEBRTC_MODE:
        rtsp_url = f"rtsp://127.0.0.1:{MEDIAMTX_RTSP_PORT}/{camera_name}"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-map", "0:v:0",
            "-c:v", "copy",
            "-an",
            "-f", "matroska",
            str(dest),
        ]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0)

    dev = f"/dev/{camera_name}"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", "v4l2", "-input_format", "mjpeg",
        "-framerate", "15",
        "-video_size", "640x480",
        "-i", dev,
        "-c:v", "copy",
        "-an",
        "-f", "matroska",
        str(dest),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0)


def _stop_recording_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=8)
        return
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=3)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _upload_recording(upload_url: str, session_id: str, camera_name: str, fpath: Path) -> dict:
    if upload_url:
        parsed_upload = urllib.parse.urlsplit(upload_url)
        if parsed_upload.hostname in {"localhost", "127.0.0.1", "::1"}:
            upload_url = ""
    if not upload_url:
        upload_url = urllib.parse.urljoin(CENTRAL_SERVER_URL.rstrip("/") + "/", "api/recording/video")
    parsed = urllib.parse.urlsplit(upload_url)
    query = urllib.parse.urlencode(
        {
            "device": _DEVICE_NAME,
            "cam": camera_name,
            "session": session_id,
            "filename": fpath.name,
        }
    )
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}&{query}"
    else:
        path += f"?{query}"

    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.netloc, timeout=120)
    try:
        with open(fpath, "rb") as fh:
            conn.request(
                "POST",
                path,
                body=fh,
                headers={
                    "Content-Type": "video/x-matroska",
                    "Content-Length": str(fpath.stat().st_size),
                },
            )
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
            if resp.status >= 400:
                raise RuntimeError(f"dashboard upload http {resp.status}: {raw[:300]}")
            return {"ok": True, "status": resp.status, "file": str(fpath), "response": raw[:300]}
    finally:
        conn.close()


app = Flask(__name__)


@app.route("/api/cameras")
def api_cameras():
    return jsonify(discover_camera_names(max_cameras=MAX_CAMERAS))


@app.route("/api/recording/start", methods=["POST"])
def api_recording_start():
    global _REC_SESSION

    body = request.get_json(silent=True) or {}
    session_id = _safe_name(str(body.get("session") or time.strftime("%Y-%m-%d_%H-%M-%S")))
    upload_url = str(body.get("upload_url") or "")

    with _REC_LOCK:
        if _REC_SESSION is not None:
            return jsonify({"ok": False, "error": "already recording", "session": _REC_SESSION["id"]}), 409

        session_dir = LOCAL_RECORDINGS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        camera_names = _fetch_local_camera_names(CAM_PORT, max_cameras=MAX_CAMERAS)

        procs: dict[str, subprocess.Popen] = {}
        files: dict[str, str] = {}
        errors: dict[str, str] = {}
        for camera_name in camera_names:
            if _WEBRTC_MODE:
                _get_ffmpeg_proc(camera_name)
            dest = session_dir / f"{_safe_name(camera_name)}.mkv"
            try:
                procs[camera_name] = _start_recording_ffmpeg(camera_name, dest)
                files[camera_name] = str(dest)
            except Exception as exc:
                errors[camera_name] = str(exc)

        _REC_SESSION = {
            "id": session_id,
            "dir": str(session_dir),
            "upload_url": upload_url,
            "started_at": time.time(),
            "procs": procs,
            "files": files,
        }

    print(f"[RasPi recording] started {session_id}: {list(files)}")
    return jsonify({"ok": True, "session": session_id, "dir": str(session_dir), "cameras": list(files), "errors": errors})


@app.route("/api/recording/stop", methods=["POST"])
def api_recording_stop():
    global _REC_SESSION

    body = request.get_json(silent=True) or {}
    expected_session = str(body.get("session") or "")

    with _REC_LOCK:
        if _REC_SESSION is None:
            return jsonify({"ok": False, "error": "not recording"}), 409
        sess = _REC_SESSION
        if expected_session and expected_session != sess["id"]:
            return jsonify({"ok": False, "error": "session mismatch", "session": sess["id"]}), 409
        _REC_SESSION = None

    for proc in sess["procs"].values():
        _stop_recording_proc(proc)

    uploads: list[dict] = []
    for camera_name, raw_path in sess["files"].items():
        fpath = Path(raw_path)
        if not fpath.exists() or fpath.stat().st_size <= 0:
            uploads.append({"camera": camera_name, "ok": False, "error": "empty or missing recording", "file": str(fpath)})
            continue
        try:
            result = _upload_recording(sess["upload_url"], sess["id"], camera_name, fpath)
            result["camera"] = camera_name
            uploads.append(result)
        except Exception as exc:
            uploads.append({"camera": camera_name, "ok": False, "error": str(exc), "file": str(fpath)})

    print(f"[RasPi recording] stopped {sess['id']}: {uploads}")
    return jsonify({"ok": True, "session": sess["id"], "dir": sess["dir"], "uploads": uploads})


@app.route("/api/power", methods=["POST"])
def api_power():
    body = request.get_json(silent=True) or {}
    action = str(body.get("action") or "").strip().lower()
    if action not in ("shutdown", "reboot"):
        return jsonify({"error": "invalid action"}), 400

    def _run():
        time.sleep(1.5)
        subprocess.run(["shutdown", "-h", "now"] if action == "shutdown" else ["reboot"])

    threading.Thread(target=_run, daemon=True).start()
    print(f"[power] {action} requested")
    return jsonify({"ok": True, "action": action})


def _start_control_api() -> None:
    threading.Thread(
        target=app.run,
        kwargs={"host": "0.0.0.0", "port": CAM_PORT, "threaded": True, "use_reloader": False},
        daemon=True,
    ).start()


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
    CENTRAL_SERVER_URL = args.server

    camera_names = discover_camera_names(max_cameras=args.max_cameras)
    if not camera_names:
        print("WARNING: no /dev/video* devices found.")

    camera_proc = None

    if _WEBRTC_MODE:
        print(f"[RasPi] WebRTC mode — MediaMTX RTSP:{MEDIAMTX_RTSP_PORT} WebRTC:{MEDIAMTX_WEBRTC_PORT}")
        _start_control_api()
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
