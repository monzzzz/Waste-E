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
import http.client
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from flask import Flask, jsonify, request

from arm import ArmController, LEROBOT_OK
from camera import CameraManager
from recording import RecordingManager, _safe_name

# ── Config ───────────────────────────────────────────────────────────────────

CAM_PORT      = int(os.getenv("CAM_PORT", "5000"))
MAX_CAMERAS   = 3
SEND_HZ       = 2.0
RETRY_SECONDS = 5.0

_central_url  = ""   # set in main() from --server arg

# ── Managers ─────────────────────────────────────────────────────────────────

arm       = ArmController()
cameras   = CameraManager(max_cameras=MAX_CAMERAS, cam_port=CAM_PORT)
recording = RecordingManager(cameras=cameras, get_central_url=lambda: _central_url)

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/api/cameras")
def api_cameras():
    return jsonify(cameras.discover())


@app.route("/api/recording/start", methods=["POST"])
def api_recording_start():
    body       = request.get_json(silent=True) or {}
    session_id = _safe_name(str(body.get("session") or time.strftime("%Y-%m-%d_%H-%M-%S")))
    upload_url = str(body.get("upload_url") or "")
    result     = recording.start(session_id, upload_url)
    return jsonify(result), 200 if result["ok"] else 409


@app.route("/api/recording/stop", methods=["POST"])
def api_recording_stop():
    body             = request.get_json(silent=True) or {}
    expected_session = str(body.get("session") or "")
    result           = recording.stop(expected_session)
    return jsonify(result), 200 if result["ok"] else 409


@app.route("/api/power", methods=["POST"])
def api_power():
    body   = request.get_json(silent=True) or {}
    action = str(body.get("action") or "").strip().lower()
    if action not in ("shutdown", "reboot"):
        return jsonify({"error": "invalid action"}), 400

    def _run():
        time.sleep(1.5)
        subprocess.run(["shutdown", "-h", "now"] if action == "shutdown" else ["reboot"])

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "action": action})


@app.route("/api/arm/status")
def api_arm_status():
    if not LEROBOT_OK:
        return jsonify({"error": "lerobot not installed"}), 503
    return jsonify(arm.status())


@app.route("/api/arm/connect", methods=["POST"])
def api_arm_connect():
    if not LEROBOT_OK:
        return jsonify({"error": "lerobot not installed"}), 503
    if arm.is_connected:
        return jsonify({"ok": True, "message": "already connected", "port": arm.port})
    body = request.get_json(silent=True) or {}
    arm.connect_async(port=body.get("port"))
    return jsonify({"ok": True, "message": "connecting", "port": arm.port})


@app.route("/api/arm/disconnect", methods=["POST"])
def api_arm_disconnect():
    if not LEROBOT_OK:
        return jsonify({"error": "lerobot not installed"}), 503
    arm.disconnect()
    return jsonify({"ok": True})


@app.route("/api/arm/action", methods=["POST"])
def api_arm_action():
    if not LEROBOT_OK:
        return jsonify({"error": "lerobot not installed"}), 503
    if not arm.is_connected:
        return jsonify({"error": "arm not connected"}), 503
    body = request.get_json(silent=True) or {}
    try:
        action_dict = arm.send_action(body.get("action"))
        return jsonify({"ok": True, "action": action_dict})
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/arm/observation")
def api_arm_observation():
    if not LEROBOT_OK:
        return jsonify({"error": "lerobot not installed"}), 503
    if not arm.is_connected:
        return jsonify({"error": "arm not connected"}), 503
    try:
        return jsonify(arm.get_observation())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Dashboard client ──────────────────────────────────────────────────────────

class DashboardClient:
    def __init__(self, server_url: str, my_ip: str) -> None:
        self.server_url = server_url
        self.my_ip      = my_ip

    def run_loop(self) -> None:
        poster = _PersistentPoster(self.server_url)
        while True:
            reg = {
                "device":        "rasppi",
                "ip":            self.my_ip,
                "cam_port":      CAM_PORT,
                "cameras":       cameras.fetch_local_names(),
                "webrtc_port":   cameras.webrtc_port if cameras.webrtc_mode else None,
                "arm_port":      arm.port,
                "arm_available": LEROBOT_OK,
            }
            try:
                poster.post("/api/register", reg)
                print(f"[RasPi sender] Registered: {reg}")
            except Exception as exc:
                print(f"[RasPi sender] Registration failed: {exc} — retrying in {RETRY_SECONDS}s")
                time.sleep(RETRY_SECONDS)
                poster = _PersistentPoster(self.server_url)
                continue

            while True:
                try:
                    payload = {
                        "device":      "rasppi",
                        "ts":          time.time(),
                        "cam_port":    CAM_PORT,
                        "cameras":     cameras.fetch_local_names(),
                        "webrtc_port": cameras.webrtc_port if cameras.webrtc_mode else None,
                        "arm":         {"connected": arm.is_connected, "port": arm.port},
                    }
                    poster.post("/api/data", payload)
                except Exception as exc:
                    print(f"[RasPi sender] Data POST failed: {exc} — will re-register")
                    break
                time.sleep(1.0 / SEND_HZ)


class _PersistentPoster:
    def __init__(self, server_url: str, timeout: float = 5.0) -> None:
        parsed        = urllib.parse.urlsplit(server_url)
        self._host    = parsed.netloc
        self._https   = parsed.scheme == "https"
        self._timeout = timeout
        self._conn: http.client.HTTPConnection | None = None

    def post(self, path: str, payload: dict) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        for attempt in range(2):
            try:
                if self._conn is None:
                    self._conn = (
                        http.client.HTTPSConnection(self._host, timeout=self._timeout)
                        if self._https
                        else http.client.HTTPConnection(self._host, timeout=self._timeout)
                    )
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


# ── Entry point ───────────────────────────────────────────────────────────────

def _start_flask() -> None:
    threading.Thread(
        target=app.run,
        kwargs={"host": "0.0.0.0", "port": CAM_PORT, "threaded": True, "use_reloader": False},
        daemon=True,
    ).start()


def _start_camera_server(script_path: Path) -> subprocess.Popen:
    if not script_path.exists():
        raise FileNotFoundError(f"Camera stream script not found: {script_path}")
    env = os.environ.copy()
    env["CAM_PORT"] = str(CAM_PORT)
    return subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent), env=env,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi data sender")
    parser.add_argument("--server",        default="http://192.168.1.100:9000")
    parser.add_argument("--my-ip",         default="192.168.1.11")
    parser.add_argument("--cam-port",      type=int, default=CAM_PORT)
    parser.add_argument("--camera-script", default="camera_stream.py")
    parser.add_argument("--max-cameras",   type=int, default=MAX_CAMERAS)
    args = parser.parse_args()

    _central_url         = args.server
    CAM_PORT             = args.cam_port
    cameras.max_cameras  = args.max_cameras
    cameras.cam_port     = args.cam_port

    camera_names = cameras.discover()
    if not camera_names:
        print("WARNING: no /dev/video* devices found.")

    camera_proc = None

    if cameras.webrtc_mode:
        print(f"[RasPi] WebRTC mode — RTSP:{cameras.rtsp_port} WebRTC:{cameras.webrtc_port}")
        _start_flask()
        cameras.start_mediamtx()
        time.sleep(1.5)
        for name in camera_names:
            cameras.get_or_start_ffmpeg(name)
        cameras.start_watchdog(camera_names)
    else:
        print("[RasPi] MJPEG mode — starting camera_stream.py")
        camera_proc = _start_camera_server(Path(__file__).parent / args.camera_script)
        time.sleep(1.5)

    print(f"[RasPi] Cameras: {camera_names}")
    print(f"[RasPi] Central server: {args.server}")

    client = DashboardClient(server_url=args.server, my_ip=args.my_ip)
    try:
        thread = threading.Thread(target=client.run_loop, daemon=True)
        thread.start()
        thread.join()
    except KeyboardInterrupt:
        print("[RasPi] Interrupted, shutting down")
    finally:
        arm.disconnect()
        if camera_proc and camera_proc.poll() is None:
            camera_proc.terminate()
            camera_proc.wait(timeout=5)
