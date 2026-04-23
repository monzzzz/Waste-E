#!/usr/bin/env python3
"""
Orange Pi Data Sender
- Serves 4 camera MJPEG streams on --cam-port (default 8890)
- Reads GPS/IMU data (direct or via dashboard.py API)
- Exposes /api/drive for central motor control
- Registers with and streams data to the central dashboard server

Run:
    python data_sender.py --server http://192.168.1.100:9000 --my-ip 192.168.1.10

If dashboard.py is already running on this device, pass --dashboard to avoid
competing for serial ports and to proxy drive commands to dashboard.py:
    python data_sender.py --server http://... --dashboard http://localhost:8888
"""
from __future__ import annotations

import argparse
import copy
import glob
import http.client
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from flask import Flask, Response, abort, request

# ── sensor imports (graceful fallback) ──────────────────────────────────────
try:
    from GPS import NEO8M_GPS
    _HAS_GPS = True
except Exception:
    _HAS_GPS = False

try:
    from IMUsensor import BNO055_IMU
    _HAS_IMU = True
except Exception:
    _HAS_IMU = False

try:
    from motorencoder import AS5600Encoder
    _HAS_ENC = True
except Exception:
    _HAS_ENC = False

try:
    from motor_driver import MotorDriver
    _HAS_MOTOR = True
except Exception:
    _HAS_MOTOR = False

# ── config ───────────────────────────────────────────────────────────────────
GPS_PORT = os.getenv("GPS_PORT", "/dev/ttyS0")
IMU_PORT = os.getenv("IMU_PORT", "/dev/ttyS6")
CAM_PORT = int(os.getenv("CAM_PORT", "8890"))
ROTATE_180 = {"/dev/video0", "/dev/video2", "/dev/video6"}
CAM_W, CAM_H, CAM_FPS, CAM_Q = 1280, 720, 10, 2
SEND_HZ = 10.0
DRIVE_ACTIONS = {"forward", "backward", "left", "right", "stop"}

DASHBOARD_URL: Optional[str] = None
DASHBOARD_DRIVE_URL: Optional[str] = None
DRIVE_PORT: int = CAM_PORT

# ── camera discovery ─────────────────────────────────────────────────────────
_VIDEO_RE = re.compile(r"^video\d+$")


def _video_key(p: str):
    m = re.search(r"(\d+)$", p)
    return (int(m.group(1)), p) if m else (10**9, p)


def _is_primary(dev: Path) -> bool:
    f = Path("/sys/class/video4linux") / dev.name / "index"
    try:
        return f.read_text().strip() == "0"
    except OSError:
        return False


def discover_cameras() -> list[str]:
    return [
        str(p)
        for p in sorted((Path(x) for x in glob.glob("/dev/video*")), key=lambda p: _video_key(str(p)))
        if _VIDEO_RE.match(p.name) and _is_primary(p)
    ]


CAM_DEVS = discover_cameras()
_CAM_PROCS: dict[str, subprocess.Popen] = {}
_CAM_LOCK = threading.Lock()


def _start_ffmpeg(dev: str) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "v4l2",
        "-input_format",
        "mjpeg",
        "-framerate",
        str(CAM_FPS),
        "-video_size",
        f"{CAM_W}x{CAM_H}",
        "-i",
        dev,
    ]
    if dev in ROTATE_180:
        cmd += ["-vf", "hflip,vflip"]
    cmd += ["-f", "mjpeg", "-q:v", str(CAM_Q), "pipe:1"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)


def _get_proc(dev: str) -> subprocess.Popen:
    with _CAM_LOCK:
        p = _CAM_PROCS.get(dev)
        if p is None or p.poll() is not None:
            _CAM_PROCS[dev] = _start_ffmpeg(dev)
        return _CAM_PROCS[dev]


def _mjpeg_gen(dev: str):
    p = _get_proc(dev)
    buf = b""
    while True:
        chunk = p.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            s = buf.find(b"\xff\xd8")
            if s == -1:
                break
            e = buf.find(b"\xff\xd9", s + 2)
            if e == -1:
                break
            frame = buf[s : e + 2]
            buf = buf[e + 2 :]
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                + frame
                + b"\r\n"
            )


# ── state ────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_sensor_state: dict = {
    "gps": {
        "lat": None,
        "lon": None,
        "alt": None,
        "speed": None,
        "heading": None,
        "satellites": None,
        "fix": False,
    },
    "imu": {
        "heading": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "qw": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "temp": None,
        "cal_sys": 0,
        "cal_gyro": 0,
        "cal_accel": 0,
        "cal_mag": 0,
    },
    "encoders": {"left_rpm": None, "right_rpm": None, "left_angle": None, "right_angle": None},
    "motor_online": False,
    "motor_source": "none",
    "ts": 0.0,
}

_active_motor: Optional[object] = None
_motor_lock = threading.Lock()
_motor_last_error: Optional[str] = None


# ── helpers ──────────────────────────────────────────────────────────────────
def _extract_port(url: Optional[str], fallback: int) -> int:
    if not url:
        return fallback
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.port:
            return parsed.port
        if parsed.scheme == "https":
            return 443
        if parsed.scheme == "http":
            return 80
    except Exception:
        pass
    return fallback


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


class _PersistentReader:
    """Reuses a single TCP connection for repeated GETs from the same host."""

    def __init__(self, timeout: float = 2.5):
        self._timeout = timeout
        self._host: str = ""
        self._conn: http.client.HTTPConnection | None = None

    def get_json(self, url: str) -> dict:
        parsed = urllib.parse.urlsplit(url)
        host = parsed.netloc
        path = parsed.path or "/"
        if host != self._host:
            self._conn = None
            self._host = host
        for attempt in range(2):
            try:
                if self._conn is None:
                    self._conn = http.client.HTTPConnection(host, timeout=self._timeout)
                self._conn.request("GET", path, headers={"Connection": "keep-alive"})
                resp = self._conn.getresponse()
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw) if raw else {}
                return data if isinstance(data, dict) else {}
            except Exception:
                self._conn = None
                if attempt == 0:
                    continue
                return {}
        return {}


def _read_json_from_url(url: str, timeout: float = 2.5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw) if raw else {}
    return data if isinstance(data, dict) else {}


# ── sensor + motor threads ───────────────────────────────────────────────────
def _sensor_loop(dashboard_url: Optional[str]):
    gps: Optional[object] = None
    imu: Optional[object] = None
    enc_left: Optional[object] = None
    enc_right: Optional[object] = None
    _enc_prev: dict = {"left": None, "right": None, "ts": None}
    _reader = _PersistentReader(timeout=2.0)

    if not dashboard_url:
        if _HAS_GPS:
            try:
                gps = NEO8M_GPS(port=GPS_PORT)
                if not gps.connect():
                    gps = None
                else:
                    print(f"[GPS] connected on {GPS_PORT}")
            except Exception as e:
                print(f"[GPS] error: {e}")

        if _HAS_IMU:
            try:
                imu = BNO055_IMU(uart_port=IMU_PORT)
                if not imu.connect():
                    imu = None
                else:
                    print(f"[IMU] connected on {IMU_PORT}")
            except Exception as e:
                print(f"[IMU] error: {e}")

        if _HAS_ENC:
            try:
                enc_left = AS5600Encoder(bus_id=1, address=0x36)
                enc_left.open()
                print("[ENC] left encoder opened (bus 1, addr 0x36)")
            except Exception as e:
                print(f"[ENC] left encoder: {e}")
            try:
                enc_right = AS5600Encoder(bus_id=0, address=0x36)
                enc_right.open()
                print("[ENC] right encoder opened (bus 0, addr 0x36)")
            except Exception as e:
                print(f"[ENC] right encoder: {e}")

    while True:
        with _state_lock:
            state = copy.deepcopy(_sensor_state)

        if dashboard_url:
            try:
                dash = _reader.get_json(f"{dashboard_url.rstrip('/')}/api/state")
                g = dash.get("gps", {})
                state["gps"].update(
                    {
                        "lat": g.get("lat"),
                        "lon": g.get("lon"),
                        "alt": g.get("alt"),
                        "speed": g.get("speed"),
                        "heading": g.get("heading"),
                        "satellites": g.get("satellites"),
                        "fix": bool(g.get("fix")),
                    }
                )

                im = dash.get("imu", {})
                state["imu"].update(
                    {
                        "heading": im.get("heading", 0.0),
                        "roll": im.get("roll", 0.0),
                        "pitch": im.get("pitch", 0.0),
                        "qw": im.get("qw", 1.0),
                        "qx": im.get("qx", 0.0),
                        "qy": im.get("qy", 0.0),
                        "qz": im.get("qz", 0.0),
                        "temp": im.get("temp"),
                        "cal_sys": im.get("cal_sys", 0),
                        "cal_gyro": im.get("cal_gyro", 0),
                        "cal_accel": im.get("cal_accel", 0),
                        "cal_mag": im.get("cal_mag", 0),
                    }
                )
                enc = dash.get("encoders", {})
                if isinstance(enc, dict):
                    state["encoders"].update({
                        "left_rpm": enc.get("left_rpm"),
                        "right_rpm": enc.get("right_rpm"),
                        "left_angle": enc.get("left_angle"),
                        "right_angle": enc.get("right_angle"),
                    })
                state["motor_online"] = bool(dash.get("motor_online"))
                state["motor_source"] = "dashboard"
            except Exception:
                state["motor_online"] = False
                state["motor_source"] = "dashboard"
        else:
            if gps:
                try:
                    d = gps.get_gps_data()
                    if d:
                        state["gps"].update(
                            {
                                "lat": d.get("latitude"),
                                "lon": d.get("longitude"),
                                "alt": d.get("altitude"),
                                "speed": d.get("speed"),
                                "heading": d.get("heading"),
                                "satellites": d.get("satellites"),
                                "fix": bool(d.get("fix_quality", 0)),
                            }
                        )
                except Exception as e:
                    print(f"[GPS] read error: {e}")

            if imu:
                try:
                    d = imu.get_all_data()
                    if d:
                        euler = d.get("euler", {})
                        quat = d.get("quaternion", {})
                        cal = d.get("calibration", {}) or {}
                        state["imu"].update(
                            {
                                "heading": euler.get("heading") or 0.0,
                                "roll": euler.get("roll") or 0.0,
                                "pitch": euler.get("pitch") or 0.0,
                                "qw": quat.get("w") or 1.0,
                                "qx": quat.get("x") or 0.0,
                                "qy": quat.get("y") or 0.0,
                                "qz": quat.get("z") or 0.0,
                                "temp": d.get("temperature"),
                                "cal_sys": cal.get("system", 0),
                                "cal_gyro": cal.get("gyro", 0),
                                "cal_accel": cal.get("accel", 0),
                                "cal_mag": cal.get("mag", 0),
                            }
                        )
                except Exception as e:
                    print(f"[IMU] read error: {e}")
                    if "Bad file descriptor" in str(e) or "NoneType" in str(e):
                        try:
                            imu.disconnect()
                            time.sleep(2)
                            if imu.connect():
                                print("[IMU] reconnected")
                            else:
                                imu = None
                        except Exception:
                            imu = None

            with _motor_lock:
                state["motor_online"] = _active_motor is not None
            state["motor_source"] = "local"

            now_enc = time.time()
            if enc_left:
                try:
                    la = enc_left.read_angle_deg()
                    state["encoders"]["left_angle"] = round(la, 2)
                    if _enc_prev["left"] is not None and _enc_prev["ts"] is not None:
                        dt = now_enc - _enc_prev["ts"]
                        if dt > 0:
                            d = la - _enc_prev["left"]
                            if d > 180: d -= 360
                            elif d < -180: d += 360
                            state["encoders"]["left_rpm"] = round((d / 360.0) / dt * 60.0, 2)
                    _enc_prev["left"] = la
                except Exception:
                    pass
            if enc_right:
                try:
                    ra = enc_right.read_angle_deg()
                    state["encoders"]["right_angle"] = round(ra, 2)
                    if _enc_prev["right"] is not None and _enc_prev["ts"] is not None:
                        dt = now_enc - _enc_prev["ts"]
                        if dt > 0:
                            d = ra - _enc_prev["right"]
                            if d > 180: d -= 360
                            elif d < -180: d += 360
                            state["encoders"]["right_rpm"] = round((d / 360.0) / dt * 60.0, 2)
                    _enc_prev["right"] = ra
                except Exception:
                    pass
            _enc_prev["ts"] = now_enc

        state["ts"] = time.time()
        with _state_lock:
            _sensor_state.update(state)

        time.sleep(1.0 / SEND_HZ)


def _motor_manager():
    global _active_motor, _motor_last_error
    while True:
        with _motor_lock:
            has_motor = _active_motor is not None

        if has_motor:
            time.sleep(1.0)
            continue

        try:
            m = MotorDriver()
            m.open()
            with _motor_lock:
                _active_motor = m
                _motor_last_error = None
            print("[motor] local motor driver ready")
        except Exception as e:
            with _motor_lock:
                _active_motor = None
                _motor_last_error = str(e)
            print(f"[motor] unavailable: {e} — retrying in 3s")
            time.sleep(3.0)


def _register_and_send_loop(server_url: str, my_ip: str, drive_port: int):
    cam_names = [os.path.basename(d) for d in CAM_DEVS]
    registration = {
        "device": "orangepi",
        "ip": my_ip,
        "cam_port": CAM_PORT,
        "cameras": cam_names,
        "drive_port": drive_port,
        "dashboard_port": drive_port,
    }
    poster = _PersistentPoster(server_url)

    while True:
        try:
            poster.post("/api/register", registration)
            print(f"[sender] Registered with central server — cameras: {cam_names} drive_port: {drive_port}")
        except Exception as e:
            print(f"[sender] Registration failed: {e} — retrying in 5s")
            time.sleep(5)
            continue

        while True:
            try:
                with _state_lock:
                    state = copy.deepcopy(_sensor_state)
                payload = {
                    "device": "orangepi",
                    "ts": time.time(),
                    "cam_port": CAM_PORT,
                    "cameras": cam_names,
                    "drive_port": drive_port,
                    "dashboard_port": drive_port,
                    "sensors": state,
                }
                poster.post("/api/data", payload)
            except Exception as e:
                print(f"[sender] POST failed: {e} — re-registering")
                break
            time.sleep(1.0 / SEND_HZ)


# ── Flask (camera server + local state API) ─────────────────────────────────
app = Flask(__name__)


@app.route("/cam/<name>")
def cam_stream(name):
    dev = f"/dev/{name}"
    if dev not in CAM_DEVS or not os.path.exists(dev):
        abort(404)
    return Response(_mjpeg_gen(dev), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    with _state_lock:
        return json.dumps(_sensor_state)


@app.route("/api/cameras")
def api_cameras():
    return json.dumps([os.path.basename(d) for d in CAM_DEVS if os.path.exists(d)])


@app.route("/api/drive", methods=["POST"])
def api_drive():
    global _motor_last_error

    body = request.get_json(silent=True) or {}
    action = str(body.get("action") or "stop").strip().lower()
    if action not in DRIVE_ACTIONS:
        return json.dumps({"error": f"invalid action: {action}"}), 400

    # dashboard passthrough mode
    if DASHBOARD_DRIVE_URL:
        req_data = json.dumps({"action": action}).encode("utf-8")
        req = urllib.request.Request(
            DASHBOARD_DRIVE_URL,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return raw or json.dumps({"ok": True, "action": action}), resp.getcode()
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            return raw or json.dumps({"error": f"dashboard http {exc.code}"}), exc.code
        except Exception as exc:
            return json.dumps({"error": f"dashboard drive proxy failed: {exc}"}), 502

    # local motor mode
    with _motor_lock:
        driver = _active_motor

    if driver is None:
        return json.dumps({"error": f"motor driver unavailable: {_motor_last_error or 'not initialized'}"}), 503

    try:
        driver.apply_action(action)
        return json.dumps({"ok": True, "action": action})
    except Exception as exc:
        _motor_last_error = str(exc)
        return json.dumps({"error": str(exc)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orange Pi data sender")
    parser.add_argument("--server", default="http://192.168.1.100:9000", help="Central dashboard server URL")
    parser.add_argument("--my-ip", default="192.168.1.10", help="This device's IP address")
    parser.add_argument(
        "--dashboard",
        default=None,
        help="Local dashboard.py URL to avoid serial-port conflicts (e.g. http://localhost:8888)",
    )
    parser.add_argument("--cam-port", type=int, default=CAM_PORT, help=f"Local camera server port (default {CAM_PORT})")
    args = parser.parse_args()

    CAM_PORT = args.cam_port
    DASHBOARD_URL = args.dashboard

    if DASHBOARD_URL:
        DRIVE_PORT = _extract_port(DASHBOARD_URL, 8888)
        DASHBOARD_DRIVE_URL = f"{DASHBOARD_URL.rstrip('/')}/api/drive"
    else:
        DRIVE_PORT = CAM_PORT
        DASHBOARD_DRIVE_URL = None

    print(f"[OrangePi] Cameras found: {CAM_DEVS or 'none'}")
    print(f"[OrangePi] Camera server → http://0.0.0.0:{CAM_PORT}")
    print(f"[OrangePi] Central server: {args.server}")
    print(f"[OrangePi] Drive endpoint published on port: {DRIVE_PORT}")

    if DASHBOARD_URL:
        print(f"[OrangePi] Sensor source: {DASHBOARD_URL} (dashboard passthrough)")
        print(f"[OrangePi] Drive source: {DASHBOARD_DRIVE_URL} (proxied)")
    else:
        if _HAS_MOTOR:
            threading.Thread(target=_motor_manager, daemon=True).start()
        else:
            print("[OrangePi] MotorDriver import failed; /api/drive will return 503")

    threading.Thread(target=_sensor_loop, args=(DASHBOARD_URL,), daemon=True).start()
    threading.Thread(target=_register_and_send_loop, args=(args.server, args.my_ip, DRIVE_PORT), daemon=True).start()

    app.run(host="0.0.0.0", port=CAM_PORT, threaded=True, use_reloader=False)
