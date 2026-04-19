#!/usr/bin/env python3
"""
Waste-E Sensor Dashboard
Run:  python dashboard.py
Open: http://<orangepi-ip>:8888
"""
from __future__ import annotations

import atexit
import glob
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from flask import Flask, Response, render_template_string, abort, request

try:
    from coverage_planner import plan_coverage
    _HAS_COVERAGE = True
except Exception:
    _HAS_COVERAGE = False

# ── sensor imports (graceful fallback if hardware missing) ───────────────────
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
    from motor_driver import MotorDriver
    _HAS_MOTOR = True
except Exception:
    _HAS_MOTOR = False

# ── config ───────────────────────────────────────────────────────────────────
GPS_PORT   = os.getenv("GPS_PORT",  "/dev/ttyS0")
IMU_PORT   = os.getenv("IMU_PORT",  "/dev/ttyS6")
DASH_PORT  = int(os.getenv("DASH_PORT", "8888"))
CAM_PORT   = int(os.getenv("CAM_PORT",  "8889"))   # internal MJPEG port

ROTATE_180 = {"/dev/video0", "/dev/video2", "/dev/video6"}
CAM_W, CAM_H, CAM_FPS, CAM_Q = 1280, 720, 10, 2

# ── camera discovery & streaming ─────────────────────────────────────────────
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
_CAM_LOCK  = threading.Lock()

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
    "ts": 0.0,
}
_sse_queues: list[queue.SimpleQueue] = []
_sse_lock = threading.Lock()

def _broadcast(payload: str):
    with _sse_lock:
        dead = []
        for q in _sse_queues:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            _sse_queues.remove(q)

_active_gps: Optional["NEO8M_GPS"] = None
_active_imu: Optional["BNO055_IMU"] = None
_active_motor: Optional["MotorDriver"] = None
_motor_last_cmd: float = 0.0   # epoch-time of last /api/drive call

def _cleanup_sensors():
    """Release serial ports and motors — called on exit."""
    global _active_gps, _active_imu, _active_motor
    if _active_gps:
        try: _active_gps.disconnect()
        except Exception: pass
        _active_gps = None
    if _active_imu:
        try: _active_imu.disconnect()
        except Exception: pass
        _active_imu = None
    if _active_motor:
        try: _active_motor.close()
        except Exception: pass
        _active_motor = None
    print("[Waste-E] Serial ports released.")

atexit.register(_cleanup_sensors)
signal.signal(signal.SIGTERM, lambda *_: (_cleanup_sensors(), exit(0)))
signal.signal(signal.SIGINT,  lambda *_: (_cleanup_sensors(), exit(0)))

def _sensor_loop():
    global _active_gps, _active_imu
    gps: Optional[NEO8M_GPS] = None
    imu: Optional[BNO055_IMU] = None

    if _HAS_GPS:
        try:
            gps = NEO8M_GPS(port=GPS_PORT)
            if not gps.connect():
                print(f"[GPS] failed to connect on {GPS_PORT}")
                gps = None
            else:
                _active_gps = gps
                print(f"[GPS] connected on {GPS_PORT}")
        except Exception as e:
            print(f"[GPS] error: {e}")

    if _HAS_IMU:
        try:
            imu = BNO055_IMU(uart_port=IMU_PORT)
            if not imu.connect():
                print(f"[IMU] failed to connect on {IMU_PORT}")
                imu = None
            else:
                _active_imu = imu
                print(f"[IMU] connected on {IMU_PORT}")
        except Exception as e:
            print(f"[IMU] error: {e}")

    while True:
        with _state_lock:
            state = json.loads(json.dumps(_sensor_state))  # deep copy

        # GPS
        if gps:
            try:
                d = gps.get_gps_data()
                if d:
                    state["gps"].update({
                        "lat":       d.get("latitude"),
                        "lon":       d.get("longitude"),
                        "alt":       d.get("altitude"),
                        "speed":     d.get("speed"),
                        "heading":   d.get("heading"),
                        "satellites":d.get("satellites"),
                        "fix":       bool(d.get("fix_quality", 0)),
                    })
            except Exception as e:
                print(f"[GPS] read error: {e}")

        # IMU
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
                # Bad file descriptor / UART error — sensor dropped; reconnect
                if "Bad file descriptor" in str(e) or "UART" in str(e) or "NoneType" in str(e):
                    print("[IMU] connection lost, reconnecting…")
                    imu.disconnect()
                    time.sleep(2)
                    if imu.connect():
                        _active_imu = imu
                        print("[IMU] reconnected")
                    else:
                        imu = None
                        _active_imu = None

        state["ts"] = time.time()
        state["motor_online"] = _active_motor is not None
        with _state_lock:
            _sensor_state.update(state)

        _broadcast("data: " + json.dumps(state) + "\n\n")
        time.sleep(0.2)   # 5 Hz

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    cam_names = [os.path.basename(d) for d in CAM_DEVS if os.path.exists(d)]
    return render_template_string(HTML,
        cam_names=cam_names, cam_port=CAM_PORT,
        cam_w=CAM_W, cam_h=CAM_H, cam_fps=CAM_FPS, cam_q=CAM_Q,
    )

@app.route("/cam/<name>")
def cam_stream(name):
    dev = f"/dev/{name}"
    if dev not in CAM_DEVS or not os.path.exists(dev):
        abort(404)
    return Response(_mjpeg_gen(dev), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/state")
def api_state():
    with _state_lock:
        data = dict(_sensor_state)
    data["motor_online"] = _active_motor is not None
    return json.dumps(data)

@app.route("/api/drive", methods=["POST"])
def api_drive():
    global _motor_last_cmd
    if not _active_motor:
        return json.dumps({"error": "motor driver not available"}), 503
    body   = request.get_json(force=True) or {}
    action = body.get("action", "stop")
    if action not in ("forward", "backward", "left", "right", "stop"):
        return json.dumps({"error": "invalid action"}), 400
    _motor_last_cmd = time.time()
    try:
        _active_motor.apply_action(action)
    except Exception as e:
        return json.dumps({"error": str(e)}), 500
    return json.dumps({"ok": True, "action": action})

@app.route("/api/coverage", methods=["POST"])
def api_coverage():
    if not _HAS_COVERAGE:
        return json.dumps({"error": "coverage_planner not available"}), 500
    try:
        body = request.get_json(force=True)
        result = plan_coverage(
            polygon_latlon=body["polygon"],
            strip_width_m=float(body.get("strip_width_m", 0.6)),
            angle_deg=float(body.get("angle_deg", 0.0)),
        )
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)}), 400

@app.route("/api/stream")
def api_stream():
    q: queue.SimpleQueue = queue.SimpleQueue()
    with _sse_lock:
        _sse_queues.append(q)

    def gen():
        yield ": connected\n\n"
        while True:
            try:
                msg = q.get(timeout=30)
                yield msg
            except queue.Empty:
                yield ": ping\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── HTML / JS / CSS (single-file) ─────────────────────────────────────────────
HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Waste-E Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<style>
/* ─── reset & tokens ──────────────────────────────────────────────────────── */
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#060810;
  --surface:#0b0e18;
  --surface2:#0f1422;
  --border:#1a2035;
  --border-hi:#2a3555;
  --cyan:#00e5ff;
  --violet:#7c3aed;
  --emerald:#10b981;
  --amber:#f59e0b;
  --rose:#f43f5e;
  --text:#dde6f5;
  --muted:#4a5878;
  --muted2:#2a3555;
  --r:14px;
  --gap:10px;
  --mono:'JetBrains Mono',monospace;
}
html,body{
  height:100%;
  background:var(--bg);
  color:var(--text);
  font-family:'Inter',system-ui,sans-serif;
  font-size:13px;
  overflow:hidden;
}

/* ─── scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border-hi);border-radius:2px}

/* ─── layout ──────────────────────────────────────────────────────────────── */
#root{
  display:grid;
  height:100vh;
  padding:var(--gap);
  gap:var(--gap);
  grid-template-rows:52px 1fr 300px;
  grid-template-columns:1fr 360px;
  grid-template-areas:
    "hdr  hdr"
    "map  imu"
    "cams ctrl";
}

/* ─── header ──────────────────────────────────────────────────────────────── */
#hdr{
  grid-area:hdr;
  display:flex;
  align-items:center;
  gap:14px;
  padding:0 20px;
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:var(--r);
  position:relative;
  overflow:hidden;
}
/* animated top-border accent */
#hdr::before{
  content:'';
  position:absolute;
  top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--violet),var(--cyan),var(--emerald),var(--cyan),var(--violet));
  background-size:300% 100%;
  animation:gradshift 6s linear infinite;
}
@keyframes gradshift{to{background-position:300% 0}}

/* logo */
.logo{display:flex;align-items:center;gap:8px}
.logo-icon{
  width:32px;height:32px;border-radius:8px;
  background:linear-gradient(135deg,var(--violet),var(--cyan));
  display:flex;align-items:center;justify-content:center;
  font-size:16px;flex-shrink:0;
  box-shadow:0 0 12px #7c3aed60;
}
.logo-text{font-size:17px;font-weight:800;letter-spacing:-.01em}
.logo-text em{
  font-style:normal;
  background:linear-gradient(90deg,var(--cyan),var(--violet));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}

/* status pills */
.pill{
  display:flex;align-items:center;gap:6px;
  padding:4px 12px;border-radius:99px;
  font-size:11px;font-weight:600;letter-spacing:.02em;
  border:1px solid transparent;
  transition:all .3s;
}
.pill-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.pill.ok   {background:#0a2a1e;border-color:#10b98140;color:var(--emerald)}
.pill.ok   .pill-dot{background:var(--emerald);box-shadow:0 0 6px var(--emerald);animation:pulse-g 2s infinite}
.pill.warn {background:#2a1a00;border-color:#f59e0b40;color:var(--amber)}
.pill.warn .pill-dot{background:var(--amber);box-shadow:0 0 6px var(--amber);animation:pulse-a 2s infinite}
.pill.bad  {background:#2a0a10;border-color:#f43f5e40;color:var(--rose)}
.pill.bad  .pill-dot{background:var(--rose);animation:none}
@keyframes pulse-g{0%,100%{box-shadow:0 0 4px var(--emerald)}50%{box-shadow:0 0 10px var(--emerald)}}
@keyframes pulse-a{0%,100%{box-shadow:0 0 4px var(--amber)}50%{box-shadow:0 0 10px var(--amber)}}

/* right section of header */
#hdr-right{
  margin-left:auto;
  display:flex;align-items:center;gap:12px;
}
/* encoding chip */
#enc-chip{
  display:flex;align-items:center;gap:6px;
  padding:4px 12px;border-radius:8px;
  background:var(--surface2);border:1px solid var(--border-hi);
  font-family:var(--mono);font-size:10px;color:var(--muted);
}
#enc-chip span{color:var(--cyan);font-weight:600}
#enc-chip .sep{color:var(--border-hi);margin:0 2px}
#clock{
  font-family:var(--mono);font-size:12px;color:var(--muted);
  min-width:72px;text-align:right;
}

/* ─── generic panel ───────────────────────────────────────────────────────── */
.panel{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:var(--r);
  overflow:hidden;
  display:flex;flex-direction:column;
}
.phdr{
  display:flex;align-items:center;gap:8px;
  padding:9px 14px;
  border-bottom:1px solid var(--border);
  flex-shrink:0;
  background:var(--surface2);
}
.phdr-icon{
  width:26px;height:26px;border-radius:7px;
  display:flex;align-items:center;justify-content:center;
  font-size:13px;flex-shrink:0;
}
.phdr-icon.cyan  {background:#00e5ff15;border:1px solid #00e5ff30}
.phdr-icon.violet{background:#7c3aed15;border:1px solid #7c3aed30}
.phdr-icon.green {background:#10b98115;border:1px solid #10b98130}
.phdr h2{
  font-size:11px;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);
}
.phdr-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.pbody{flex:1;overflow:auto;min-height:0;position:relative}

/* ─── GPS map ─────────────────────────────────────────────────────────────── */
#map-panel{grid-area:map}
#map{width:100%;height:100%}
.leaflet-container{background:#070a12}
.leaflet-tile-pane{filter:invert(1) hue-rotate(180deg) brightness(.82) saturate(.6)}
.leaflet-control-zoom a{
  background:var(--surface2)!important;color:var(--text)!important;
  border-color:var(--border)!important;
}

/* GPS stats bar */
#gps-bar{
  display:flex;gap:0;
  border-top:1px solid var(--border);
  flex-shrink:0;
  background:var(--surface2);
}
.gv{
  flex:1;display:flex;flex-direction:column;gap:2px;
  padding:8px 12px;
  border-right:1px solid var(--border);
}
.gv:last-child{border-right:none}
.gv-lbl{
  font-size:9px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);
}
.gv-val{
  font-family:var(--mono);font-size:12px;font-weight:600;
  color:var(--text);
}
.gv-val.hi{color:var(--cyan)}

/* ─── IMU panel ───────────────────────────────────────────────────────────── */
#imu-panel{grid-area:imu;display:flex;flex-direction:column}
#imu-body{
  flex:1;display:grid;
  grid-template-rows:1fr 1fr;
  min-height:0;padding:10px;gap:10px;
}

/* compass section */
#compass-wrap{
  display:flex;gap:12px;align-items:center;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:10px;padding:10px 14px;
}
#compass-canvas{flex-shrink:0;border-radius:50%}
#compass-info{display:flex;flex-direction:column;gap:6px;flex:1;min-width:0}
#heading-big{
  font-family:var(--mono);font-size:32px;font-weight:700;
  color:var(--cyan);line-height:1;letter-spacing:-.02em;
}
#heading-dir{font-size:11px;color:var(--muted);font-weight:600;letter-spacing:.06em}
.imu-stat{display:flex;flex-direction:column;gap:2px}
.imu-stat-lbl{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.imu-stat-val{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--text)}

/* 3D section */
#three-wrap{
  display:flex;flex-direction:column;align-items:center;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:10px;overflow:hidden;position:relative;
}
#three-canvas{width:100%;flex:1;min-height:0;display:block}
#euler-strip{
  display:flex;width:100%;border-top:1px solid var(--border);
  background:var(--surface);flex-shrink:0;
}
.eitem{
  flex:1;display:flex;flex-direction:column;align-items:center;
  padding:5px 0;gap:1px;border-right:1px solid var(--border);
}
.eitem:last-child{border-right:none}
.eitem-lbl{font-size:9px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.eitem-val{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--text)}
.eitem-val.cyan{color:var(--cyan)}

/* calibration footer */
#cal-bar{
  display:flex;align-items:center;gap:8px;
  padding:7px 14px;border-top:1px solid var(--border);
  flex-shrink:0;background:var(--surface2);
}
#cal-bar-lbl{font-size:10px;font-weight:700;letter-spacing:.06em;color:var(--muted);text-transform:uppercase;margin-right:2px}
.cdot{
  display:flex;align-items:center;justify-content:center;
  width:28px;height:22px;border-radius:5px;
  font-size:10px;font-weight:700;font-family:var(--mono);
  border:1px solid transparent;transition:all .4s;
}
.cdot.c0{background:#111827;border-color:#1f2937;color:#374151}
.cdot.c1{background:#1c0a00;border-color:#78350f;color:#d97706}
.cdot.c2{background:#052e16;border-color:#065f46;color:#34d399}
.cdot.c3{background:#022c22;border-color:#10b981;color:#10b981;box-shadow:0 0 8px #10b98150}
#imu-temp-val{
  margin-left:auto;
  font-family:var(--mono);font-size:11px;color:var(--muted);
}

/* ─── cameras ─────────────────────────────────────────────────────────────── */
#cams-panel{grid-area:cams}
#cams-grid{
  display:flex;height:100%;
  gap:8px;padding:8px;
  overflow-x:auto;overflow-y:hidden;
}
.cam-card{
  flex-shrink:0;
  height:100%;
  aspect-ratio:16/9;
  background:#03040a;
  border:1px solid var(--border);
  border-radius:10px;
  overflow:hidden;
  display:flex;flex-direction:column;
  position:relative;
  transition:border-color .2s;
}
.cam-card:hover{border-color:var(--border-hi)}
.cam-lbl{
  position:absolute;top:0;left:0;right:0;
  display:flex;align-items:center;gap:6px;
  padding:5px 10px;
  background:linear-gradient(180deg,rgba(0,0,0,.8) 0%,transparent 100%);
  font-size:10px;font-weight:600;letter-spacing:.04em;color:#fff;
  z-index:2;
}
.cam-lbl-dot{
  width:5px;height:5px;border-radius:50%;
  background:var(--emerald);box-shadow:0 0 5px var(--emerald);
  animation:pulse-g 2s infinite;
}
.cam-enc{
  position:absolute;bottom:0;left:0;right:0;
  padding:4px 10px;
  background:linear-gradient(0deg,rgba(0,0,0,.75) 0%,transparent 100%);
  font-family:var(--mono);font-size:9px;color:rgba(255,255,255,.5);
  z-index:2;text-align:right;
}
.cam-card img{
  width:100%;height:100%;object-fit:cover;display:block;
  position:absolute;top:0;left:0;
}
.no-cams{
  display:flex;align-items:center;justify-content:center;
  width:100%;color:var(--muted);font-size:13px;gap:8px;
}

/* ─── live dot in panel headers ──────────────────────────────────────────── */
.live-dot{
  width:6px;height:6px;border-radius:50%;
  background:var(--emerald);box-shadow:0 0 6px var(--emerald);
  animation:pulse-g 2s infinite;flex-shrink:0;
}

/* ─── coverage button ─────────────────────────────────────────────────────── */
#cov-btn{
  display:flex;align-items:center;gap:6px;
  padding:5px 12px;border-radius:8px;border:1px solid var(--border-hi);
  background:var(--surface2);color:var(--text);
  font-size:11px;font-weight:600;cursor:pointer;
  transition:all .2s;white-space:nowrap;
}
#cov-btn:hover{background:var(--violet);border-color:var(--violet);color:#fff;box-shadow:0 0 12px #7c3aed60}
#cov-btn.active{background:var(--violet);border-color:var(--violet);color:#fff;box-shadow:0 0 12px #7c3aed80}
#cov-btn svg{flex-shrink:0}

/* ─── path planning side panel ┤ draw + coverage ─────────────────────────── */
#path-panel{
  position:absolute;top:50px;right:10px;z-index:500;
  width:260px;
  background:var(--surface);border:1px solid var(--border-hi);border-radius:12px;
  box-shadow:0 8px 32px rgba(0,0,0,.6);
  display:none;flex-direction:column;overflow:hidden;
}
#path-panel.open{display:flex}
#path-panel-hdr{
  padding:10px 14px;background:var(--surface2);
  border-bottom:1px solid var(--border);
  font-size:12px;font-weight:700;letter-spacing:.04em;color:var(--cyan);
  display:flex;align-items:center;gap:8px;
}
#path-panel-tabs{
  display:flex;border-bottom:1px solid var(--border);
  background:var(--surface2);
}
.path-tab{
  flex:1;padding:8px;background:transparent;border:none;color:var(--muted);
  font-size:11px;font-weight:600;text-align:center;cursor:pointer;
  border-bottom:2px solid transparent;transition:all .2s;
}
.path-tab:hover{color:var(--text)}
.path-tab.active{
  color:var(--cyan);border-bottom-color:var(--cyan);
  background:var(--surface);
}
.path-mode{display:none}
.path-mode.active{display:block}
#draw-panel-body,#cov-panel-body{padding:12px;display:flex;flex-direction:column;gap:10px}
.cov-field{display:flex;flex-direction:column;gap:4px}
.cov-field label{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.cov-field input[type=range]{
  width:100%;accent-color:var(--violet);cursor:pointer;
}
.cov-field .val{font-family:var(--mono);font-size:12px;color:var(--cyan);font-weight:600}
.cov-field input[type=number]{
  width:100%;background:var(--surface2);border:1px solid var(--border-hi);
  border-radius:6px;padding:5px 8px;color:var(--text);font-family:var(--mono);font-size:12px;
  outline:none;
}
.cov-field input[type=number]:focus{border-color:var(--violet)}
.cov-divider{height:1px;background:var(--border);margin:2px 0}
#cov-draw-btn,#cov-clear-btn,#cov-gen-btn,#draw-finish-btn,#draw-clear-btn{
  padding:8px;border-radius:8px;border:none;cursor:pointer;
  font-size:12px;font-weight:600;width:100%;transition:all .2s;
}
#cov-draw-btn,#draw-finish-btn{background:var(--surface2);color:var(--text);border:1px solid var(--border-hi)}
#cov-draw-btn:hover,#draw-finish-btn:hover{border-color:var(--cyan);color:var(--cyan)}
#cov-draw-btn.drawing{background:#00e5ff20;border-color:var(--cyan);color:var(--cyan)}
#cov-gen-btn{
  background:linear-gradient(135deg,var(--violet),#6d28d9);color:#fff;
  box-shadow:0 2px 12px #7c3aed40;
}
#cov-gen-btn:hover{box-shadow:0 2px 18px #7c3aed70}
#cov-gen-btn:disabled{opacity:.4;cursor:not-allowed;box-shadow:none}
#draw-finish-btn:disabled{opacity:.4;cursor:not-allowed}
#cov-clear-btn,#draw-clear-btn{background:transparent;color:var(--rose);border:1px solid #f43f5e30;font-size:11px}
#cov-clear-btn:hover,#draw-clear-btn:hover{background:#f43f5e15}
#cov-result,#draw-result{
  font-size:11px;color:var(--muted);
  background:var(--surface2);border-radius:6px;padding:8px;
  line-height:1.6;display:none;
}
#cov-result.show,#draw-result.show{display:block}
#cov-result strong,#draw-result strong{color:var(--cyan)}
/* Leaflet draw overrides */
.leaflet-draw-toolbar a{background-color:var(--surface2)!important;border-color:var(--border)!important}

/* ─── drive joystick ──────────────────────────────────────────────────── */
#ctrl-panel{grid-area:ctrl}
#joy-wrap{
  flex:1;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:10px;
  padding:10px;
}
#joy-ring{
  width:210px;height:210px;border-radius:50%;
  background:radial-gradient(circle at 50% 50%,#0f1422 55%,#070a12 100%);
  border:2px solid var(--border-hi);
  position:relative;
  touch-action:none;user-select:none;cursor:grab;
  box-shadow:0 0 0 1px var(--border),inset 0 0 40px rgba(0,0,0,.5);
  flex-shrink:0;
}
/* crosshair guide lines */
#joy-ring::before,#joy-ring::after{
  content:'';position:absolute;top:50%;left:50%;
  background:var(--border);pointer-events:none;
}
#joy-ring::before{width:1px;height:65%;transform:translate(-50%,-50%)}
#joy-ring::after {width:65%;height:1px;transform:translate(-50%,-50%)}
#joy-thumb{
  width:66px;height:66px;border-radius:50%;
  background:radial-gradient(circle at 38% 38%,var(--cyan),var(--violet));
  position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);
  box-shadow:0 0 14px #7c3aed70,0 0 4px #00e5ff50;
  pointer-events:none;
  will-change:transform;
}
#joy-thumb.active{box-shadow:0 0 28px #7c3aed,0 0 10px #00e5ff}
.joy-dir{
  position:absolute;font-size:9px;font-weight:700;
  letter-spacing:.08em;color:var(--muted);text-transform:uppercase;
  pointer-events:none;
}
.joy-dir.n{top:8px; left:50%;transform:translateX(-50%)}
.joy-dir.s{bottom:8px;left:50%;transform:translateX(-50%)}
.joy-dir.w{left:8px; top:50%;transform:translateY(-50%)}
.joy-dir.e{right:8px;top:50%;transform:translateY(-50%)}
#drive-status{
  font-family:var(--mono);font-size:13px;font-weight:700;
  letter-spacing:.12em;color:var(--muted);text-transform:uppercase;
  text-align:center;height:18px;
}
</style>
</head>
<body>
<div id="root">

<!-- ── header ──────────────────────────────────────────────────────────── -->
<div id="hdr">
  <div class="logo">
    <div class="logo-icon">🤖</div>
    <div class="logo-text">Waste<em>-E</em></div>
  </div>

  <div style="width:1px;height:24px;background:var(--border);margin:0 2px"></div>

  <div class="pill bad" id="gps-badge">
    <div class="pill-dot"></div>GPS: no fix
  </div>
  <div class="pill warn" id="imu-badge">
    <div class="pill-dot"></div>IMU: —
  </div>

  <div id="hdr-right">
    <!-- encoding info -->
    <div id="enc-chip">
      <span>MJPEG</span>
      <span class="sep">·</span>
      <span>{{ cam_w }}×{{ cam_h }}</span>
      <span class="sep">·</span>
      <span>{{ cam_fps }} fps</span>
      <span class="sep">·</span>
      Q<span>{{ cam_q }}</span>
    </div>
    <div id="clock">—</div>
  </div>
</div>

<!-- ── GPS map ──────────────────────────────────────────────────────────── -->
<div class="panel" id="map-panel">
  <div class="phdr">
    <div class="phdr-icon cyan">🗺</div>
    <h2>GPS Location</h2>
    <div class="phdr-right">
      <button id="path-btn" title="Path planning">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 3h18v18H3z"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>
        </svg>
        Path Planning
      </button>
      <div class="live-dot" id="gps-live-dot" style="background:var(--rose);box-shadow:none;animation:none"></div>
    </div>
  </div>
  <div class="pbody">
    <div id="map"></div>
    <!-- path planning side panel -->
    <div id="path-panel">
      <div id="path-panel-hdr">📍 Path Planning</div>
      <div id="path-panel-tabs">
        <button class="path-tab active" data-mode="draw">Draw Path</button>
        <button class="path-tab" data-mode="coverage">Coverage Area</button>
      </div>
      <!-- Draw Path Mode -->
      <div id="draw-mode" class="path-mode active">
        <div id="draw-panel-body">
          <div style="font-size:11px;color:var(--muted);margin-bottom:8px">
            📍 Click on the map to add waypoints. Double-click to finish.
          </div>
          <div id="draw-waypoints" style="font-size:10px;color:var(--muted);background:var(--surface2);border-radius:6px;padding:8px;max-height:80px;overflow-y:auto;margin-bottom:8px"></div>
          <button id="draw-finish-btn" disabled>✅ Send Path to Robot</button>
          <button id="draw-clear-btn">✕ Clear Path</button>
          <div id="draw-result"></div>
        </div>
      </div>
      <!-- Coverage Area Mode -->
      <div id="coverage-mode" class="path-mode" style="display:none">
        <div id="cov-panel-body">
          <div class="cov-field">
            <label>Strip Width</label>
            <input type="range" id="cov-width" min="0.2" max="3.0" step="0.1" value="0.6"/>
            <span class="val" id="cov-width-val">0.6 m</span>
          </div>
          <div class="cov-field">
            <label>Sweep Angle</label>
            <input type="range" id="cov-angle" min="0" max="175" step="5" value="0"/>
            <span class="val" id="cov-angle-val">0°</span>
          </div>
          <div class="cov-divider"></div>
          <button id="cov-draw-btn">✏️ Draw Rectangle</button>
          <button id="cov-gen-btn" disabled>⚡ Generate Path</button>
          <button id="cov-clear-btn">✕ Clear</button>
          <div id="cov-result"></div>
        </div>
      </div>
    </div>
  </div>
  <div id="gps-bar">
    <div class="gv"><div class="gv-lbl">Latitude</div> <div class="gv-val hi" id="g-lat">—</div></div>
    <div class="gv"><div class="gv-lbl">Longitude</div><div class="gv-val hi" id="g-lon">—</div></div>
    <div class="gv"><div class="gv-lbl">Altitude</div> <div class="gv-val"    id="g-alt">—</div></div>
    <div class="gv"><div class="gv-lbl">Speed</div>    <div class="gv-val"    id="g-spd">—</div></div>
    <div class="gv"><div class="gv-lbl">Heading</div>  <div class="gv-val"    id="g-hdg">—</div></div>
    <div class="gv"><div class="gv-lbl">Satellites</div><div class="gv-val"   id="g-sat">—</div></div>
  </div>
</div>

<!-- ── IMU ──────────────────────────────────────────────────────────────── -->
<div class="panel" id="imu-panel">
  <div class="phdr">
    <div class="phdr-icon violet">🧭</div>
    <h2>IMU Orientation</h2>
    <div class="phdr-right">
      <div class="live-dot" id="imu-live-dot" style="background:var(--amber);box-shadow:none;animation:none"></div>
    </div>
  </div>

  <div id="imu-body">
    <!-- compass + info -->
    <div id="compass-wrap">
      <canvas id="compass-canvas" width="130" height="130"></canvas>
      <div id="compass-info">
        <div id="heading-big">0.0°</div>
        <div id="heading-dir">North</div>
        <div style="height:8px"></div>
        <div class="imu-stat">
          <div class="imu-stat-lbl">Roll</div>
          <div class="imu-stat-val" id="s-rol">0.0°</div>
        </div>
        <div class="imu-stat">
          <div class="imu-stat-lbl">Pitch</div>
          <div class="imu-stat-val" id="s-pit">0.0°</div>
        </div>
      </div>
    </div>

    <!-- 3-D -->
    <div id="three-wrap">
      <canvas id="three-canvas" style="flex:1;width:100%"></canvas>
      <div id="euler-strip">
        <div class="eitem"><div class="eitem-lbl">Heading</div><div class="eitem-val cyan" id="e-hdg">0.0°</div></div>
        <div class="eitem"><div class="eitem-lbl">Roll</div>   <div class="eitem-val"      id="e-rol">0.0°</div></div>
        <div class="eitem"><div class="eitem-lbl">Pitch</div>  <div class="eitem-val"      id="e-pit">0.0°</div></div>
      </div>
    </div>
  </div>

  <!-- calibration -->
  <div id="cal-bar">
    <span id="cal-bar-lbl">Cal</span>
    <div class="cdot c0" id="cal-sys"   title="System calibration">SYS</div>
    <div class="cdot c0" id="cal-gyro"  title="Gyroscope">GYR</div>
    <div class="cdot c0" id="cal-accel" title="Accelerometer">ACC</div>
    <div class="cdot c0" id="cal-mag"   title="Magnetometer">MAG</div>
    <div id="imu-temp-val"></div>
  </div>
</div>

<!-- ── cameras ──────────────────────────────────────────────────────────── -->
<div class="panel" id="cams-panel">
  <div class="phdr">
    <div class="phdr-icon green">📷</div>
    <h2>Camera Feeds</h2>
    <div class="phdr-right">
      <span style="font-size:10px;color:var(--muted)" id="cam-count"></span>
    </div>
  </div>
  <div class="pbody" style="overflow:hidden">
    <div id="cams-grid">
    {% if cam_names %}
      {% for name in cam_names %}
      <div class="cam-card">
        <div class="cam-lbl"><span class="cam-lbl-dot"></span>{{ name }}</div>
        <img src="/cam/{{ name }}" alt="{{ name }}"
             onerror="this.style.opacity='.15'" loading="lazy"/>
        <div class="cam-enc">MJPEG · {{ cam_w }}×{{ cam_h }} · {{ cam_fps }}fps</div>
      </div>
      {% endfor %}
    {% else %}
      <div class="no-cams">🎥 No cameras detected</div>
    {% endif %}
    </div>
  </div>
</div>

<!-- ── drive joystick ────────────────────────────────────────────────── -->
<div class="panel" id="ctrl-panel">
  <div class="phdr">
    <div class="phdr-icon violet">🕹</div>
    <h2>Drive Control</h2>
    <div class="phdr-right">
      <div id="drive-status">STOPPED</div>
    </div>
  </div>
  <div id="joy-wrap">
    <div id="joy-ring">
      <span class="joy-dir n">FWD</span>
      <span class="joy-dir s">BWD</span>
      <span class="joy-dir w">L</span>
      <span class="joy-dir e">R</span>
      <div id="joy-thumb"></div>
    </div>
  </div>
</div>

</div><!-- #root -->
<script>
// ══════════════════════════════════════════════════════
// MAP
// ══════════════════════════════════════════════════════
const map = L.map('map',{center:[13.75,100.52],zoom:16,zoomControl:true,attributionControl:false});
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19}).addTo(map);

const robotIcon = L.divIcon({
  className:'',
  html:`<div style="
    width:40px;height:40px;border-radius:50%;
    background:conic-gradient(from 180deg,#7c3aed,#00e5ff,#7c3aed);
    border:3px solid rgba(255,255,255,.9);
    box-shadow:0 0 0 4px #00e5ff40,0 4px 20px #00e5ff60;
    display:flex;align-items:center;justify-content:center;font-size:20px;
  ">🤖</div>`,
  iconSize:[40,40],iconAnchor:[20,20],
});
let marker=null, mapReady=false;
let trail=[], trailLine=null;

function updateMap(lat,lon){
  if(!lat||!lon)return;
  if(!mapReady){map.setView([lat,lon],17);mapReady=true;}
  if(!marker){marker=L.marker([lat,lon],{icon:robotIcon}).addTo(map);}
  else{marker.setLatLng([lat,lon]);}
  // breadcrumb trail (last 120 pts)
  trail.push([lat,lon]);
  if(trail.length>120)trail.shift();
  if(trailLine)map.removeLayer(trailLine);
  trailLine=L.polyline(trail,{color:'#00e5ff',weight:2,opacity:.45,dashArray:'4 4'}).addTo(map);
}

// ══════════════════════════════════════════════════════
// COMPASS  (2-D canvas)
// ══════════════════════════════════════════════════════
const cc=document.getElementById('compass-canvas');
const ctx=cc.getContext('2d');
const R=cc.width/2;

const DIRS=[
  {l:'N',a:-90,col:'#f43f5e'},{l:'NE',a:-45,col:'#4a5878',sm:true},
  {l:'E',a:0,col:'#94a3b8'},{l:'SE',a:45,col:'#4a5878',sm:true},
  {l:'S',a:90,col:'#94a3b8'},{l:'SW',a:135,col:'#4a5878',sm:true},
  {l:'W',a:180,col:'#94a3b8'},{l:'NW',a:-135,col:'#4a5878',sm:true},
];

function headingToDir(h){
  const dirs=['N','NE','E','SE','S','SW','W','NW'];
  return dirs[Math.round(((h%360)+360)%360/45)%8];
}

function drawCompass(hdg){
  const W=cc.width,H=cc.height;
  ctx.clearRect(0,0,W,H);

  // background glow
  const grd=ctx.createRadialGradient(R,R,R*.3,R,R,R);
  grd.addColorStop(0,'#00e5ff08');grd.addColorStop(1,'transparent');
  ctx.fillStyle=grd;ctx.beginPath();ctx.arc(R,R,R,0,Math.PI*2);ctx.fill();

  // outer ring
  ctx.beginPath();ctx.arc(R,R,R-2,0,Math.PI*2);
  ctx.strokeStyle='#1a2035';ctx.lineWidth=2;ctx.stroke();

  // degree ticks
  for(let i=0;i<72;i++){
    const a=(i*5-90)*Math.PI/180;
    const isMaj=i%18===0, isMed=i%9===0;
    const len=isMaj?14:isMed?9:5;
    const ro=R-2, ri=ro-len;
    ctx.beginPath();
    ctx.moveTo(R+Math.cos(a)*ro,R+Math.sin(a)*ro);
    ctx.lineTo(R+Math.cos(a)*ri,R+Math.sin(a)*ri);
    ctx.strokeStyle=isMaj?'#00e5ff':isMed?'#2a3555':'#1a2035';
    ctx.lineWidth=isMaj?2:1;ctx.stroke();
  }

  // cardinal & intercardinal labels
  DIRS.forEach(({l,a,col,sm})=>{
    const rad=a*Math.PI/180;
    const r2=R-(sm?28:24);
    ctx.font=(sm?'bold 8px':'bold 10px')+' Inter,sans-serif';
    ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillStyle=col;
    ctx.fillText(l,R+Math.cos(rad)*r2,R+Math.sin(rad)*r2);
  });

  // rotating car silhouette
  const hRad=(hdg-90)*Math.PI/180;
  ctx.save();ctx.translate(R,R);ctx.rotate(hRad);

  // shadow
  ctx.shadowColor='#7c3aed';ctx.shadowBlur=8;
  ctx.fillStyle='#7c3aed25';ctx.strokeStyle='#7c3aed90';ctx.lineWidth=1.5;
  ctx.beginPath();ctx.roundRect(-13,-20,26,40,3);ctx.fill();ctx.stroke();
  ctx.shadowBlur=0;

  // front headlight glow
  ctx.fillStyle='#00e5ff30';ctx.fillRect(-13,-20,26,6);

  // nose arrow
  ctx.fillStyle='#00e5ff';ctx.shadowColor='#00e5ff';ctx.shadowBlur=6;
  ctx.beginPath();ctx.moveTo(0,-27);ctx.lineTo(-7,-17);ctx.lineTo(7,-17);
  ctx.closePath();ctx.fill();ctx.shadowBlur=0;

  ctx.restore();

  // center pip
  ctx.beginPath();ctx.arc(R,R,4,0,Math.PI*2);
  ctx.fillStyle='#00e5ff';ctx.shadowColor='#00e5ff';ctx.shadowBlur=8;
  ctx.fill();ctx.shadowBlur=0;
}

// ══════════════════════════════════════════════════════
// THREE.JS  3-D car
// ══════════════════════════════════════════════════════
(function initThree(){
  const canvas=document.getElementById('three-canvas');
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.shadowMap.enabled=true;
  renderer.shadowMap.type=THREE.PCFSoftShadowMap;

  const scene=new THREE.Scene();
  const camera=new THREE.PerspectiveCamera(42,2,0.1,100);
  camera.position.set(3.5,2.8,3.5);camera.lookAt(0,0,0);

  // lights
  scene.add(new THREE.AmbientLight(0x8899cc,0.6));
  const sun=new THREE.DirectionalLight(0x00e5ff,1.4);
  sun.position.set(4,6,4);sun.castShadow=true;
  sun.shadow.mapSize.set(512,512);scene.add(sun);
  const fill=new THREE.DirectionalLight(0x7c3aed,0.8);
  fill.position.set(-3,2,-3);scene.add(fill);
  const rim=new THREE.PointLight(0x10b981,0.6,8);
  rim.position.set(0,3,-2);scene.add(rim);

  // ground grid
  const gridHelper=new THREE.GridHelper(8,20,0x1a2035,0x1a2035);
  scene.add(gridHelper);

  // shadow plane
  const planeGeo=new THREE.PlaneGeometry(8,8);
  const planeMat=new THREE.ShadowMaterial({opacity:.35});
  const plane=new THREE.Mesh(planeGeo,planeMat);
  plane.rotation.x=-Math.PI/2;plane.position.y=-.201;plane.receiveShadow=true;
  scene.add(plane);

  // ── car assembly ──
  const car=new THREE.Group();

  // body shell
  const bodyG=new THREE.BoxGeometry(1.3,0.42,2.1);
  const bodyM=new THREE.MeshPhongMaterial({color:0x6d28d9,shininess:120,specular:0x7c3aed});
  const body=new THREE.Mesh(bodyG,bodyM);body.castShadow=true;car.add(body);

  // body edge glow (wireframe)
  car.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(bodyG),
    new THREE.LineBasicMaterial({color:0x7c3aed,transparent:true,opacity:.3})
  ));

  // cab / roof
  const cabG=new THREE.BoxGeometry(1.0,0.38,1.1);
  const cabM=new THREE.MeshPhongMaterial({color:0x4c1d95,shininess:80});
  const cab=new THREE.Mesh(cabG,cabM);cab.castShadow=true;
  cab.position.set(0,.4,-.15);car.add(cab);

  // windshield (glass-like)
  const windG=new THREE.BoxGeometry(.98,.01,.55);
  const windM=new THREE.MeshPhongMaterial({color:0x00e5ff,transparent:true,opacity:.35,shininess:200});
  const wind=new THREE.Mesh(windG,windM);wind.position.set(0,.59,.22);car.add(wind);

  // front light bar
  const fbarG=new THREE.BoxGeometry(1.32,.06,.04);
  const fbarM=new THREE.MeshBasicMaterial({color:0x00e5ff});
  const fbar=new THREE.Mesh(fbarG,fbarM);fbar.position.set(0,.12,1.07);car.add(fbar);

  // rear light bar
  const rbarM=new THREE.MeshBasicMaterial({color:0xf43f5e});
  const rbar=new THREE.Mesh(new THREE.BoxGeometry(1.32,.06,.04),rbarM);
  rbar.position.set(0,.12,-1.07);car.add(rbar);

  // wheels
  const wG=new THREE.CylinderGeometry(.2,.2,.14,20);
  const wM=new THREE.MeshPhongMaterial({color:0x111827,shininess:60});
  const hubM=new THREE.MeshPhongMaterial({color:0x374151,shininess:80});
  [[.73,-.18,.75],[-.73,-.18,.75],[.73,-.18,-.75],[-.73,-.18,-.75]].forEach(([x,y,z])=>{
    const whl=new THREE.Mesh(wG,wM);whl.rotation.z=Math.PI/2;whl.position.set(x,y,z);
    whl.castShadow=true;car.add(whl);
    const hub=new THREE.Mesh(new THREE.CylinderGeometry(.08,.08,.15,8),hubM);
    hub.rotation.z=Math.PI/2;hub.position.set(x,y,z);car.add(hub);
  });

  // antenna
  const antG=new THREE.CylinderGeometry(.015,.015,.55,6);
  const antM=new THREE.MeshPhongMaterial({color:0x6b7280});
  const ant=new THREE.Mesh(antG,antM);ant.position.set(.3,.8,-.4);car.add(ant);
  const tipM=new THREE.MeshBasicMaterial({color:0xf43f5e});
  const tip=new THREE.Mesh(new THREE.SphereGeometry(.04,8,8),tipM);
  tip.position.set(.3,1.08,-.4);car.add(tip);

  scene.add(car);

  // subtle axes indicator
  scene.add(new THREE.AxesHelper(1.0));

  // state
  let _h=0,_r=0,_p=0;
  window._setCarOrientation=(h,r,p)=>{_h=h;_r=r;_p=p;};

  function resize(){
    const w=canvas.parentElement.clientWidth||300;
    const h=canvas.parentElement.clientHeight-36||120;
    renderer.setSize(w,Math.max(h,80));
    camera.aspect=w/Math.max(h,80);
    camera.updateProjectionMatrix();
  }
  resize();
  new ResizeObserver(resize).observe(canvas.parentElement);

  // soft float animation
  let t=0;
  function animate(){
    requestAnimationFrame(animate);
    t+=.008;
    const yaw=_h*Math.PI/180;
    const rollR=_r*Math.PI/180;
    const pitchR=_p*Math.PI/180;
    car.rotation.set(-pitchR,-yaw,rollR,'YXZ');
    // gentle float
    car.position.y=Math.sin(t)*.03;
    // tip pulse
    tip.material.color.setHSL((t*.5)%1,.9,.6);
    renderer.render(scene,camera);
  }
  animate();
})();

// ══════════════════════════════════════════════════════
// DATA / SSE
// ══════════════════════════════════════════════════════
const mono=v=>v!=null?v:'—';
const fmt=(v,d=5)=>v!=null?Number(v).toFixed(d):'—';
const deg=v=>v!=null?Number(v).toFixed(1)+'°':'—';

function calCls(v){return 'cdot c'+Math.min(3,Math.max(0,v|0));}
function setCalDot(id,v){
  const el=document.getElementById(id);
  el.className=calCls(v);
}

function applyState(s){
  const g=s.gps||{}, im=s.imu||{};

  // clock
  if(s.ts){
    document.getElementById('clock').textContent=
      new Date(s.ts*1000).toLocaleTimeString();
  }

  // GPS pill + live dot
  const gpsPill=document.getElementById('gps-badge');
  const gpsLive=document.getElementById('gps-live-dot');
  if(g.fix&&g.lat){
    gpsPill.className='pill ok';
    gpsPill.innerHTML=`<div class="pill-dot"></div>GPS · ${g.satellites||'?'} sats`;
    gpsLive.style.cssText='background:var(--emerald);box-shadow:0 0 6px var(--emerald);animation:pulse-g 2s infinite';
  } else {
    gpsPill.className='pill bad';
    gpsPill.innerHTML=`<div class="pill-dot"></div>GPS: no fix`;
    gpsLive.style.cssText='background:var(--rose);box-shadow:none;animation:none';
  }

  // GPS bar
  document.getElementById('g-lat').textContent=fmt(g.lat,6);
  document.getElementById('g-lon').textContent=fmt(g.lon,6);
  document.getElementById('g-alt').textContent=g.alt!=null?fmt(g.alt,1)+' m':'—';
  document.getElementById('g-spd').textContent=g.speed!=null?fmt(g.speed,1)+' km/h':'—';
  document.getElementById('g-hdg').textContent=deg(g.heading);
  document.getElementById('g-sat').textContent=mono(g.satellites);
  updateMap(g.lat,g.lon);

  // IMU pill + live dot
  const imuPill=document.getElementById('imu-badge');
  const imuLive=document.getElementById('imu-live-dot');
  const calOk=(im.cal_sys||0)>=1;
  imuPill.className='pill '+(calOk?'ok':'warn');
  imuPill.innerHTML=`<div class="pill-dot"></div>IMU · ${calOk?'calibrated':'calibrating'}`;
  imuLive.style.cssText=calOk
    ?'background:var(--emerald);box-shadow:0 0 6px var(--emerald);animation:pulse-g 2s infinite'
    :'background:var(--amber);box-shadow:0 0 6px var(--amber);animation:pulse-a 2s infinite';

  // compass
  const hdg=im.heading||0;
  document.getElementById('heading-big').textContent=deg(im.heading);
  document.getElementById('heading-dir').textContent=headingToDir(hdg);
  document.getElementById('s-rol').textContent=deg(im.roll);
  document.getElementById('s-pit').textContent=deg(im.pitch);
  drawCompass(hdg);

  // 3D
  window._setCarOrientation&&window._setCarOrientation(im.heading||0,im.roll||0,im.pitch||0);
  document.getElementById('e-hdg').textContent=deg(im.heading);
  document.getElementById('e-rol').textContent=deg(im.roll);
  document.getElementById('e-pit').textContent=deg(im.pitch);

  // calibration
  setCalDot('cal-sys',  im.cal_sys  ||0);
  setCalDot('cal-gyro', im.cal_gyro ||0);
  setCalDot('cal-accel',im.cal_accel||0);
  setCalDot('cal-mag',  im.cal_mag  ||0);
  if(im.temp!=null)
    document.getElementById('imu-temp-val').textContent=im.temp+'°C';

  // motor online/offline
  if(window._joyMotorUpdate) window._joyMotorUpdate(!!s.motor_online);
}

fetch('/api/state').then(r=>r.json()).then(applyState).catch(()=>{});
const es=new EventSource('/api/stream');
es.onmessage=e=>{try{applyState(JSON.parse(e.data));}catch{}};

// camera count
const nCams=document.querySelectorAll('.cam-card').length;
document.getElementById('cam-count').textContent=nCams?nCams+' camera'+(nCams>1?'s':''):'';

// initial draw
drawCompass(0);

// ══════════════════════════════════════════════════════
// PATH PLANNING UI (Draw Path + Coverage Area modes)
// ══════════════════════════════════════════════════════
(function initPathPlanning(){
  const btn       = document.getElementById('path-btn');
  const panel     = document.getElementById('path-panel');
  const tabBtns   = document.querySelectorAll('.path-tab');
  const drawMode  = document.getElementById('draw-mode');
  const covMode   = document.getElementById('coverage-mode');

  // ── Draw Path Mode Elements ──
  const drawFinishBtn = document.getElementById('draw-finish-btn');
  const drawClearBtn  = document.getElementById('draw-clear-btn');
  const drawResult    = document.getElementById('draw-result');
  const waypointsDiv  = document.getElementById('draw-waypoints');

  // ── Coverage Mode Elements ──
  const covDrawBtn = document.getElementById('cov-draw-btn');
  const covGenBtn  = document.getElementById('cov-gen-btn');
  const covClearBtn= document.getElementById('cov-clear-btn');
  const covResult  = document.getElementById('cov-result');
  const widthSlider = document.getElementById('cov-width');
  const angleSlider = document.getElementById('cov-angle');
  const widthVal  = document.getElementById('cov-width-val');
  const angleVal  = document.getElementById('cov-angle-val');

  widthSlider.addEventListener('input',()=>{ widthVal.textContent=Number(widthSlider.value).toFixed(1)+' m'; });
  angleSlider.addEventListener('input',()=>{ angleVal.textContent=angleSlider.value+'°'; });

  // Leaflet feature groups
  const draftGroup    = L.featureGroup().addTo(map);
  const coverageGroup = L.featureGroup().addTo(map);
  const pathGroup     = L.featureGroup().addTo(map);

  // ── Mode switching ──
  let currentMode = 'draw'; // default mode

  tabBtns.forEach(t => {
    t.addEventListener('click', ()=>{
      const mode = t.getAttribute('data-mode');
      if(mode === currentMode) return;
      
      currentMode = mode;
      tabBtns.forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      
      document.querySelectorAll('.path-mode').forEach(m => m.classList.remove('active'));
      if(mode === 'draw') {
        drawMode.classList.add('active');
        stopRectangleDrawing();
      } else {
        covMode.classList.add('active');
        clearDrawPath();
      }
    });
  });

  // ── Panel toggle ──
  btn.addEventListener('click',()=>{
    const open = panel.classList.toggle('open');
    btn.classList.toggle('active', open);
    if(!open) {
      stopRectangleDrawing();
      clearDrawPath();
    }
  });

  // ═══════════════════════════════════════════════════════
  // DRAW PATH MODE (custom waypoint paths)
  // ═══════════════════════════════════════════════════════
  let isDrawingPath = false;
  let waypoints = [];
  let pathPolyline = null;

  function updateWaypointsList(){
    waypointsDiv.innerHTML = waypoints.length === 0 
      ? '<span style="color:var(--muted2)">No waypoints yet</span>'
      : waypoints.map((w, i) => 
          `<div style="padding:3px 0;font-family:var(--mono);font-size:10px">${i+1}. ${w[0].toFixed(4)}, ${w[1].toFixed(4)}</div>`
        ).join('');
    drawFinishBtn.disabled = waypoints.length < 2;
  }

  function redrawPath(){
    if(pathPolyline) map.removeLayer(pathPolyline);
    if(waypoints.length > 0){
      pathPolyline = L.polyline(waypoints, {
        color: '#00e5ff',
        weight: 2.5,
        opacity: 0.9,
        dashArray: waypoints.length < 2 ? '4 4' : ''
      }).addTo(pathGroup);
      
      // Draw waypoint markers
      pathGroup.clearLayers();
      pathPolyline.addTo(pathGroup);
      
      waypoints.forEach((w, i) => {
        const isFirst = i === 0;
        const isLast = i === waypoints.length - 1;
        L.circleMarker(w, {
          radius: 6,
          color: isFirst ? '#10b981' : isLast ? '#f43f5e' : '#00e5ff',
          fillColor: isFirst ? '#10b981' : isLast ? '#f43f5e' : '#00e5ff',
          fillOpacity: 1,
          weight: 2
        }).bindTooltip(`WP ${i+1}`).addTo(pathGroup);
      });
    }
  }

  function clearDrawPath(){
    isDrawingPath = false;
    waypoints = [];
    pathGroup.clearLayers();
    pathPolyline = null;
    map.getContainer().style.cursor = '';
    updateWaypointsList();
    drawResult.classList.remove('show');
    drawResult.innerHTML = '';
  }

  // Click to add waypoints
  map.on('click', (e) => {
    if(currentMode !== 'draw' || !panel.classList.contains('open')) return;
    
    waypoints.push([e.latlng.lat, e.latlng.lng]);
    redrawPath();
    updateWaypointsList();
    
    drawResult.innerHTML = `<strong>${waypoints.length}</strong> waypoints added. Double-click to finish, or continue adding.`;
    drawResult.classList.add('show');
  });

  // Double-click to finish
  map.on('dblclick', (e) => {
    if(currentMode !== 'draw' || !panel.classList.contains('open') || waypoints.length < 2) return;
    L.DomEvent.stop(e);
    // Path is ready
    drawResult.innerHTML = `Path ready with <strong>${waypoints.length}</strong> waypoints. Click <strong>Send to Robot</strong> to start.`;
  });

  drawFinishBtn.addEventListener('click', () => {
    if(waypoints.length < 2) return;
    drawResult.innerHTML = `Sending <strong>${waypoints.length}</strong> waypoints to robot…`;
    // TODO: Implement sending path to robot API
    console.log('Sending path:', waypoints);
  });

  drawClearBtn.addEventListener('click', () => {
    clearDrawPath();
  });

  // ═══════════════════════════════════════════════════════
  // COVERAGE AREA MODE (rectangle-based coverage planning)
  // ═══════════════════════════════════════════════════════
  let isDrawingRect = false;
  let corner1 = null;
  let rectPreview = null;

  function startRectangleDrawing(){
    isDrawingRect = true;
    corner1 = null;
    covDrawBtn.classList.add('drawing');
    covDrawBtn.textContent = '📍 Click top-left corner…';
    map.getContainer().style.cursor = 'crosshair';
  }

  function stopRectangleDrawing(){
    isDrawingRect = false;
    corner1 = null;
    covDrawBtn.classList.remove('drawing');
    covDrawBtn.textContent = '✏️ Draw Rectangle';
    map.getContainer().style.cursor = '';
    if(rectPreview){ map.removeLayer(rectPreview); rectPreview=null; }
  }

  covDrawBtn.addEventListener('click',()=>{
    if(isDrawingRect){ stopRectangleDrawing(); }
    else{ startRectangleDrawing(); }
  });

  // Live preview while moving after first corner
  map.on('mousemove',(e)=>{
    if(!isDrawingRect || !corner1 || currentMode !== 'coverage') return;
    if(rectPreview) map.removeLayer(rectPreview);
    rectPreview = L.rectangle(
      L.latLngBounds(corner1, e.latlng),
      {color:'#7c3aed',weight:2,fillColor:'#7c3aed',fillOpacity:.12,dashArray:'6 4'}
    ).addTo(map);
  });

  map.on('click',(e)=>{
    if(!isDrawingRect || currentMode !== 'coverage') return;
    if(!corner1){
      corner1 = e.latlng;
      covDrawBtn.textContent = '📍 Click bottom-right corner…';
      return;
    }
    // Second click — finalize rectangle
    const bounds = L.latLngBounds(corner1, e.latlng);
    if(rectPreview){ map.removeLayer(rectPreview); rectPreview=null; }
    draftGroup.clearLayers();
    L.rectangle(bounds,{color:'#7c3aed',weight:2,fillColor:'#7c3aed',fillOpacity:.15})
      .addTo(draftGroup);
    stopRectangleDrawing();
    covGenBtn.disabled = false;
    covResult.innerHTML='Rectangle set. Click <strong>Generate Path</strong>.';
    covResult.classList.add('show');
  });

  // Generate coverage path
  covGenBtn.addEventListener('click', async ()=>{
    const layers = draftGroup.getLayers();
    if(!layers.length){ return; }
    const bounds = layers[0].getBounds();
    const sw=bounds.getSouthWest(), ne=bounds.getNorthEast();
    const latlngs=[[sw.lat,sw.lng],[ne.lat,sw.lng],[ne.lat,ne.lng],[sw.lat,ne.lng]];

    covGenBtn.disabled = true;
    covGenBtn.textContent = '⏳ Planning…';
    covResult.innerHTML = 'Generating coverage path…';
    covResult.classList.add('show');

    try{
      const resp = await fetch('/api/coverage',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          polygon: latlngs,
          strip_width_m: parseFloat(widthSlider.value),
          angle_deg: parseFloat(angleSlider.value),
        }),
      });
      const data = await resp.json();
      if(data.error){ throw new Error(data.error); }

      // Draw path
      coverageGroup.clearLayers();
      const strips = data.strips_preview || [];
      const path   = data.path || [];

      if(strips.length){
        strips.forEach(([s, e], i)=>{
          L.polyline([s, e],{
            color: i%2===0 ? '#00e5ff' : '#7c3aed',
            weight: 2.5,
            opacity: 0.85,
          }).addTo(coverageGroup);
        });

        for(let i=0;i<strips.length-1;i++){
          L.polyline([strips[i][1], strips[i+1][0]],{
            color:'#ffffff', weight:1, opacity:0.25, dashArray:'4 6',
          }).addTo(coverageGroup);
        }

        L.circleMarker(strips[0][0],{
          radius:8,color:'#10b981',fillColor:'#10b981',fillOpacity:1,weight:2,
        }).bindTooltip('Start').addTo(coverageGroup);

        L.circleMarker(strips[strips.length-1][1],{
          radius:8,color:'#f43f5e',fillColor:'#f43f5e',fillOpacity:1,weight:2,
        }).bindTooltip('End').addTo(coverageGroup);

        const step = Math.max(1, Math.floor(strips.length/40));
        strips.forEach(([s,e],i)=>{
          if(i%step!==0) return;
          const la=(s[0]+e[0])/2, lo=(s[1]+e[1])/2;
          const ang=Math.atan2(e[1]-s[1], e[0]-s[0])*180/Math.PI;
          L.marker([la,lo],{icon:L.divIcon({
            className:'',
            html:`<div style="transform:rotate(${-ang+90}deg);color:#fff;font-size:10px;opacity:.6;line-height:1">▲</div>`,
            iconSize:[10,10],iconAnchor:[5,5],
          })}).addTo(coverageGroup);
        });

        map.fitBounds(coverageGroup.getBounds(),{padding:[24,24]});
      }

      const areaHa = (data.area_m2/10000).toFixed(3);
      const distM  = data.distance_m.toFixed(1);
      const speed  = 0.5;
      const etaMin = (data.distance_m/speed/60).toFixed(1);
      covResult.innerHTML=
        `<strong>${data.strips}</strong> strips &nbsp;·&nbsp; `+
        `<strong>${areaHa}</strong> ha<br>`+
        `Path: <strong>${distM} m</strong><br>`+
        `ETA @ 0.5 m/s: <strong>~${etaMin} min</strong><br>`+
        `<span style="color:var(--muted);font-size:10px">${(data.notes||[]).join(' · ')}</span>`;
      covResult.classList.add('show');

    } catch(err){
      covResult.innerHTML=`<span style="color:var(--rose)">Error: ${err.message}</span>`;
    } finally{
      covGenBtn.disabled = false;
      covGenBtn.textContent = '⚡ Generate Path';
    }
  });

  // Clear everything
  covClearBtn.addEventListener('click',()=>{
    draftGroup.clearLayers();
    coverageGroup.clearLayers();
    covGenBtn.disabled = true;
    covResult.className='cov-result';
    covResult.innerHTML='';
    stopRectangleDrawing();
  });
})();

// ══════════════════════════════════════════════════════
// JOYSTICK
// ══════════════════════════════════════════════════════
(function initJoystick(){
  const ring   = document.getElementById('joy-ring');
  const thumb  = document.getElementById('joy-thumb');
  const status = document.getElementById('drive-status');

  const ACTION_COLOR = {
    stop:     'var(--muted)',
    forward:  'var(--emerald)',
    backward: 'var(--rose)',
    left:     'var(--cyan)',
    right:    'var(--cyan)',
  };

  let active = false, curAction = 'stop', heartbeat = null;
  let motorOnline = false;

  // Poll motor status from /api/state (already fetched by main SSE handler)
  // We hook into applyState which is called on every SSE update.
  const _origApply = window._joyApplyHook || null;
  window._joyMotorUpdate = function(online){
    motorOnline = online;
    if(!online){
      ring.style.opacity = '0.4';
      ring.style.cursor  = 'not-allowed';
      status.textContent = 'MOTOR OFFLINE';
      status.style.color = 'var(--rose)';
    } else {
      ring.style.opacity = '';
      ring.style.cursor  = 'grab';
      if(status.textContent === 'MOTOR OFFLINE'){
        status.textContent = 'STOPPED';
        status.style.color = ACTION_COLOR.stop;
      }
    }
  };

  function ringGeom(){
    const r = ring.getBoundingClientRect();
    return {cx: r.left+r.width/2, cy: r.top+r.height/2, r: r.width/2};
  }

  function classify(dx, dy, maxR){
    const d = Math.sqrt(dx*dx+dy*dy);
    if(d < maxR*0.2) return 'stop';
    return Math.abs(dy)>=Math.abs(dx) ? (dy<0?'forward':'backward')
                                      : (dx<0?'left':'right');
  }

  function send(action){
    fetch('/api/drive',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action}),
    }).then(r=>{
      if(!r.ok) endDrag();   // server rejected — release joystick
    }).catch(()=>{ endDrag(); });
  }

  function setAction(action){
    if(action===curAction) return;
    curAction = action;
    status.textContent = action.toUpperCase();
    status.style.color = ACTION_COLOR[action]||'var(--muted)';
    send(action);
  }

  function updateThumb(e){
    if(!active) return;
    e.preventDefault();
    const pt = e.touches ? e.touches[0] : e;
    const {cx,cy,r} = ringGeom();
    let dx = pt.clientX-cx, dy = pt.clientY-cy;
    const d = Math.sqrt(dx*dx+dy*dy);
    const lim = r-33;
    if(d>lim){ dx=dx/d*lim; dy=dy/d*lim; }
    thumb.style.transform = `translate(calc(-50% + ${dx}px),calc(-50% + ${dy}px))`;
    setAction(classify(dx,dy,lim));
  }

  function startDrag(e){
    if(!motorOnline) return;   // refuse input when motor driver is offline
    e.preventDefault();
    active = true;
    thumb.classList.add('active');
    updateThumb(e);
    heartbeat = setInterval(()=>{ if(active) send(curAction); }, 50);
  }

  function endDrag(){
    if(!active) return;
    active = false;
    clearInterval(heartbeat);
    thumb.classList.remove('active');
    thumb.style.transform = 'translate(-50%,-50%)';
    curAction = '';
    setAction('stop');
    curAction = 'stop';
  }

  ring.addEventListener('mousedown',  startDrag, {passive:false});
  ring.addEventListener('touchstart', startDrag, {passive:false});
  window.addEventListener('mousemove',   updateThumb, {passive:false});
  window.addEventListener('touchmove',   updateThumb, {passive:false});
  window.addEventListener('mouseup',     endDrag);
  window.addEventListener('touchend',    endDrag);
  window.addEventListener('touchcancel', endDrag);
})();
</script>
</body>
</html>
"""

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[Waste-E] Starting sensor thread…")
    threading.Thread(target=_sensor_loop, daemon=True).start()

    if _HAS_MOTOR:
        def _motor_manager():
            """Initialize motor driver, retrying every 3 s if GPIO is busy."""
            global _active_motor
            while True:
                if _active_motor is None:
                    try:
                        m = MotorDriver()
                        m.open()
                        _active_motor = m
                        print("[Waste-E] Motor driver ready")
                    except Exception as e:
                        print(f"[Waste-E] Motor driver unavailable: {e} — retrying in 3 s")
                        time.sleep(3)
                        continue
                # Watchdog: stop motors when joystick goes silent for > 250 ms (5× heartbeat)
                time.sleep(0.1)
                if time.time() - _motor_last_cmd > 0.25:
                    try: _active_motor.stop()
                    except Exception:
                        _active_motor = None  # lost GPIO — will retry
        threading.Thread(target=_motor_manager, daemon=True).start()

    print(f"[Waste-E] Cameras found: {CAM_DEVS or 'none'}")
    print(f"[Waste-E] Dashboard → http://0.0.0.0:{DASH_PORT}")
    app.run(host="0.0.0.0", port=DASH_PORT, threaded=True, use_reloader=False)
