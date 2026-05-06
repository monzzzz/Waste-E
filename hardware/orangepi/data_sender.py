#!/usr/bin/env python3
"""
Orange Pi data sender for Waste-E.

Serves camera streams, exposes sensor/drive APIs, and registers with the
central dashboard.

WebRTC mode (mediamtx binary present):
    MediaMTX + ffmpeg → RTSP → browser via WHEP.

MJPEG fallback (no mediamtx binary):
    ffmpeg pipes MJPEG frames directly from each camera.

Pass --dashboard <url> when dashboard.py is already running on this device
to avoid competing for serial ports and to proxy drive commands through it.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import threading
import time
import urllib.parse
from typing import Optional

from flask import Flask, Response, abort, request

from motor_manager import MotorManager
from recording import RecordingManager, _safe_name
from sensors import SensorManager
from stream import StreamManager

# ── Config ────────────────────────────────────────────────────────────────────

CAM_PORT  = int(os.getenv("CAM_PORT", "8890"))
SEND_HZ   = 5.0
RETRY_S   = 5.0

_central_url = ""

# ── Managers ──────────────────────────────────────────────────────────────────

streams   = StreamManager()
motors    = MotorManager()
sensors   = SensorManager(send_hz=SEND_HZ, motor_manager=motors)
recording = RecordingManager(streams=streams, get_central_url=lambda: _central_url)

CAM_DEVS: list[str] = []   # populated in main() after discovery

# ── Flask ─────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/cam/<name>")
def cam_stream(name: str):
    if streams.webrtc_mode:
        abort(404)
    dev = f"/dev/{name}"
    if dev not in CAM_DEVS or not os.path.exists(dev):
        abort(404)
    return Response(streams.mjpeg_gen(dev), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    return json.dumps(sensors.get_state())


@app.route("/api/cameras")
def api_cameras():
    return json.dumps([os.path.basename(d) for d in CAM_DEVS if os.path.exists(d)])


@app.route("/api/power", methods=["POST"])
def api_power():
    body   = request.get_json(silent=True) or {}
    action = str(body.get("action") or "").strip().lower()
    if action not in ("shutdown", "reboot"):
        return json.dumps({"error": "invalid action"}), 400

    def _run():
        time.sleep(1.5)
        subprocess.run(["shutdown", "-h", "now"] if action == "shutdown" else ["reboot"])

    threading.Thread(target=_run, daemon=True).start()
    return json.dumps({"ok": True, "action": action})


@app.route("/api/recording/start", methods=["POST"])
def api_recording_start():
    body       = request.get_json(silent=True) or {}
    session_id = _safe_name(str(body.get("session") or time.strftime("%Y-%m-%d_%H-%M-%S")))
    upload_url = str(body.get("upload_url") or "")
    result     = recording.start(session_id, upload_url, CAM_DEVS)
    return json.dumps(result), 200 if result["ok"] else 409


@app.route("/api/recording/stop", methods=["POST"])
def api_recording_stop():
    body             = request.get_json(silent=True) or {}
    expected_session = str(body.get("session") or "")
    result           = recording.stop(expected_session)
    return json.dumps(result), 200 if result["ok"] else 409


@app.route("/api/drive", methods=["POST"])
def api_drive():
    body   = request.get_json(silent=True) or {}
    action = str(body.get("action") or "stop").strip().lower()
    body, status = motors.apply_action(action)
    return body, status


@app.route("/api/speed", methods=["POST"])
def api_speed():
    body = request.get_json(silent=True) or {}
    try:
        speed = max(0.0, min(1.0, float(body.get("speed", 1.0))))
    except (TypeError, ValueError):
        return json.dumps({"error": "speed must be 0.0–1.0"}), 400

    resp_body, status = motors.set_speed(speed)
    if status == 200:
        sensors.update_motor_speed(speed)
    return resp_body, status


# ── Dashboard client ──────────────────────────────────────────────────────────

class DashboardClient:
    def __init__(self, server_url: str, my_ip: str, drive_port: int) -> None:
        self.server_url = server_url
        self.my_ip      = my_ip
        self.drive_port = drive_port

    def run_loop(self) -> None:
        cam_names = [os.path.basename(d) for d in CAM_DEVS]
        reg       = {
            "device":         "orangepi",
            "ip":             self.my_ip,
            "cam_port":       CAM_PORT,
            "cameras":        cam_names,
            "drive_port":     self.drive_port,
            "dashboard_port": self.drive_port,
            "webrtc_port":    streams.webrtc_port if streams.webrtc_mode else None,
        }
        poster = _PersistentPoster(self.server_url)

        while True:
            try:
                poster.post("/api/register", reg)
                print(f"[sender] Registered — cameras: {cam_names} drive_port: {self.drive_port}")
            except Exception as e:
                print(f"[sender] Registration failed: {e} — retrying in {RETRY_S}s")
                time.sleep(RETRY_S)
                poster = _PersistentPoster(self.server_url)
                continue

            while True:
                try:
                    payload = {
                        "device":         "orangepi",
                        "ts":             time.time(),
                        "cam_port":       CAM_PORT,
                        "cameras":        cam_names,
                        "drive_port":     self.drive_port,
                        "dashboard_port": self.drive_port,
                        "sensors":        sensors.get_state(),
                    }
                    poster.post("/api/data", payload)
                except Exception as e:
                    print(f"[sender] POST failed: {e} — re-registering")
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

def _extract_port(url: Optional[str], fallback: int) -> int:
    if not url:
        return fallback
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.port:
            return parsed.port
        return 443 if parsed.scheme == "https" else 80
    except Exception:
        return fallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orange Pi data sender")
    parser.add_argument("--server",    default="http://192.168.1.100:9000")
    parser.add_argument("--my-ip",     default="192.168.1.10")
    parser.add_argument("--dashboard", default=None,
                        help="Local dashboard.py URL to proxy drive/sensors through")
    parser.add_argument("--cam-port",  type=int, default=CAM_PORT)
    args = parser.parse_args()

    CAM_PORT     = args.cam_port
    _central_url = args.server

    dashboard_drive_url: Optional[str] = None
    drive_port = CAM_PORT

    if args.dashboard:
        drive_port          = _extract_port(args.dashboard, 8888)
        dashboard_drive_url = f"{args.dashboard.rstrip('/')}/api/drive"

    # Re-init managers with runtime config
    motors.__init__(dashboard_drive_url=dashboard_drive_url)
    sensors.__init__(send_hz=SEND_HZ, dashboard_url=args.dashboard, motor_manager=motors)

    CAM_DEVS[:] = streams.discover()

    # Start camera infrastructure
    mediamtx_proc = streams.start_mediamtx()
    if mediamtx_proc:
        time.sleep(1.5)
        print(f"[OrangePi] WebRTC mode — RTSP:{streams.rtsp_port} WebRTC:{streams.webrtc_port}")
        for dev in CAM_DEVS:
            if os.path.exists(dev):
                print(f"[OrangePi] Starting ffmpeg → RTSP for {dev}")
                streams.get_or_start_proc(dev)
    else:
        print("[OrangePi] MJPEG mode — MediaMTX binary not found")

    print(f"[OrangePi] Cameras: {CAM_DEVS or 'none'}")
    print(f"[OrangePi] Camera server → http://0.0.0.0:{CAM_PORT}")
    print(f"[OrangePi] Central server: {args.server}")
    print(f"[OrangePi] Drive port: {drive_port}")

    if args.dashboard:
        print(f"[OrangePi] Sensor/drive source: {args.dashboard} (passthrough)")
    else:
        from sensors import _HAS_GPS, _HAS_IMU, _HAS_ENC
        from motor_manager import _HAS_MOTOR
        print(f"[OrangePi] GPS:     {'OK' if _HAS_GPS   else 'MISSING'}")
        print(f"[OrangePi] IMU:     {'OK' if _HAS_IMU   else 'MISSING'}")
        print(f"[OrangePi] Encoder: {'OK' if _HAS_ENC   else 'MISSING'}")
        print(f"[OrangePi] Motor:   {'OK' if _HAS_MOTOR else 'MISSING'}")

    motors.start()
    sensors.start()

    client = DashboardClient(server_url=args.server, my_ip=args.my_ip, drive_port=drive_port)
    threading.Thread(target=client.run_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=CAM_PORT, threaded=True, use_reloader=False)
