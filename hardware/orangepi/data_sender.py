#!/usr/bin/env python3
"""
Orange Pi Data Sender
- Serves 4 camera MJPEG streams on --cam-port (default 8890)
- Reads GPS, IMU, encoder data (direct or via dashboard.py API)
- Registers with and streams data to the central dashboard server

Run:
    python data_sender.py --server http://192.168.1.100:9000 --my-ip 192.168.1.10

If dashboard.py is already running on this device, pass --dashboard to avoid
competing for the same serial ports:
    python data_sender.py --server http://... --dashboard http://localhost:8888
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

from flask import Flask, Response, abort

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

# ── config ────────────────────────────────────────────────────────────────────
GPS_PORT   = os.getenv("GPS_PORT", "/dev/ttyS0")
IMU_PORT   = os.getenv("IMU_PORT", "/dev/ttyS6")
CAM_PORT   = int(os.getenv("CAM_PORT", "8890"))
ROTATE_180 = {"/dev/video0", "/dev/video2", "/dev/video6"}
CAM_W, CAM_H, CAM_FPS, CAM_Q = 1280, 720, 10, 2
SEND_HZ    = 5.0

# ── camera discovery ──────────────────────────────────────────────────────────
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
        str(p) for p in
        sorted((Path(x) for x in glob.glob("/dev/video*")), key=lambda p: _video_key(str(p)))
        if _VIDEO_RE.match(p.name) and _is_primary(p)
    ]

CAM_DEVS = discover_cameras()
_CAM_PROCS: dict[str, subprocess.Popen] = {}
_CAM_LOCK = threading.Lock()

def _start_ffmpeg(dev: str) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "v4l2", "-input_format", "mjpeg",
        "-framerate", str(CAM_FPS), "-video_size", f"{CAM_W}x{CAM_H}",
        "-i", dev,
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
            if s == -1: break
            e = buf.find(b"\xff\xd9", s + 2)
            if e == -1: break
            frame = buf[s:e + 2]
            buf = buf[e + 2:]
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                   + frame + b"\r\n")

# ── sensor state ─────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_sensor_state: dict = {
    "gps": {"lat": None, "lon": None, "alt": None, "speed": None,
            "heading": None, "satellites": None, "fix": False},
    "imu": {"heading": 0.0, "roll": 0.0, "pitch": 0.0,
            "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
            "temp": None, "cal_sys": 0, "cal_gyro": 0,
            "cal_accel": 0, "cal_mag": 0},
    "encoders": {"left_rpm": None, "right_rpm": None,
                 "left_angle": None, "right_angle": None},
    "ts": 0.0,
}


def _sensor_loop(dashboard_url: Optional[str]):
    gps: Optional[object] = None
    imu: Optional[object] = None

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

    while True:
        with _state_lock:
            state = json.loads(json.dumps(_sensor_state))

        if dashboard_url:
            try:
                url = f"{dashboard_url.rstrip('/')}/api/state"
                with urllib.request.urlopen(url, timeout=2) as resp:
                    dash = json.loads(resp.read().decode())
                g = dash.get("gps", {})
                state["gps"].update({
                    "lat": g.get("lat"), "lon": g.get("lon"),
                    "alt": g.get("alt"), "speed": g.get("speed"),
                    "heading": g.get("heading"), "satellites": g.get("satellites"),
                    "fix": bool(g.get("fix")),
                })
                im = dash.get("imu", {})
                state["imu"].update({
                    "heading": im.get("heading", 0.0),
                    "roll":    im.get("roll",    0.0),
                    "pitch":   im.get("pitch",   0.0),
                    "qw": im.get("qw", 1.0), "qx": im.get("qx", 0.0),
                    "qy": im.get("qy", 0.0), "qz": im.get("qz", 0.0),
                    "temp":      im.get("temp"),
                    "cal_sys":   im.get("cal_sys",   0),
                    "cal_gyro":  im.get("cal_gyro",  0),
                    "cal_accel": im.get("cal_accel", 0),
                    "cal_mag":   im.get("cal_mag",   0),
                })
            except Exception:
                pass
        else:
            if gps:
                try:
                    d = gps.get_gps_data()
                    if d:
                        state["gps"].update({
                            "lat": d.get("latitude"), "lon": d.get("longitude"),
                            "alt": d.get("altitude"), "speed": d.get("speed"),
                            "heading": d.get("heading"), "satellites": d.get("satellites"),
                            "fix": bool(d.get("fix_quality", 0)),
                        })
                except Exception as e:
                    print(f"[GPS] read error: {e}")

            if imu:
                try:
                    d = imu.get_all_data()
                    if d:
                        euler = d.get("euler", {})
                        quat  = d.get("quaternion", {})
                        cal   = d.get("calibration", {}) or {}
                        state["imu"].update({
                            "heading": euler.get("heading") or 0.0,
                            "roll":    euler.get("roll")    or 0.0,
                            "pitch":   euler.get("pitch")   or 0.0,
                            "qw": quat.get("w") or 1.0,
                            "qx": quat.get("x") or 0.0,
                            "qy": quat.get("y") or 0.0,
                            "qz": quat.get("z") or 0.0,
                            "temp":      d.get("temperature"),
                            "cal_sys":   cal.get("system",  0),
                            "cal_gyro":  cal.get("gyro",    0),
                            "cal_accel": cal.get("accel",   0),
                            "cal_mag":   cal.get("mag",     0),
                        })
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

        state["ts"] = time.time()
        with _state_lock:
            _sensor_state.update(state)

        time.sleep(1.0 / SEND_HZ)


def _register_and_send_loop(server_url: str, my_ip: str):
    cam_names = [os.path.basename(d) for d in CAM_DEVS if os.path.exists(d)]
    registration = {
        "device": "orangepi",
        "ip": my_ip,
        "cam_port": CAM_PORT,
        "cameras": cam_names,
    }

    while True:
        try:
            data = json.dumps(registration).encode()
            req = urllib.request.Request(
                f"{server_url}/api/register", data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
            print(f"[sender] Registered with central server — cameras: {cam_names}")
        except Exception as e:
            print(f"[sender] Registration failed: {e} — retrying in 5s")
            time.sleep(5)
            continue

        # Streaming loop — break back out to re-register on failure
        while True:
            try:
                with _state_lock:
                    state = json.loads(json.dumps(_sensor_state))
                payload = {"device": "orangepi", "ts": time.time(), "sensors": state}
                data = json.dumps(payload).encode()
                req = urllib.request.Request(
                    f"{server_url}/api/data", data=data,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=3):
                    pass
            except Exception as e:
                print(f"[sender] POST failed: {e} — re-registering")
                break
            time.sleep(1.0 / SEND_HZ)


# ── Flask (camera server + local state API) ──────────────────────────────────
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orange Pi data sender")
    parser.add_argument("--server",    default="http://192.168.1.100:9000",
                        help="Central dashboard server URL")
    parser.add_argument("--my-ip",     default="192.168.1.10",
                        help="This device's IP address")
    parser.add_argument("--dashboard", default=None,
                        help="Local dashboard.py URL to avoid serial-port conflicts "
                             "(e.g. http://localhost:8888)")
    parser.add_argument("--cam-port",  type=int, default=CAM_PORT,
                        help=f"Local camera server port (default {CAM_PORT})")
    args = parser.parse_args()
    CAM_PORT = args.cam_port

    print(f"[OrangePi] Cameras found: {CAM_DEVS or 'none'}")
    print(f"[OrangePi] Camera server → http://0.0.0.0:{CAM_PORT}")
    print(f"[OrangePi] Central server: {args.server}")
    if args.dashboard:
        print(f"[OrangePi] Sensor source: {args.dashboard} (dashboard passthrough)")

    threading.Thread(target=_sensor_loop,         args=(args.dashboard,),           daemon=True).start()
    threading.Thread(target=_register_and_send_loop, args=(args.server, args.my_ip), daemon=True).start()

    app.run(host="0.0.0.0", port=CAM_PORT, threaded=True, use_reloader=False)
