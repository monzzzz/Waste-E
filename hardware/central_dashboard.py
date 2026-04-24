#!/usr/bin/env python3
"""
Central dashboard server for Waste-E hardware.

Receives registration and periodic data from Orange Pi and Raspberry Pi senders,
stores live device state in memory, and renders a rich operations dashboard.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

_device_lock = threading.Lock()
_devices: dict[str, dict[str, Any]] = {}

DEFAULT_PORT = int(os.getenv("CENTRAL_DASH_PORT", os.getenv("PORT", "9000")))
STATUS_TIMEOUT = float(os.getenv("STATUS_TIMEOUT", "15"))
ORANGE_DASH_PORT = int(os.getenv("ORANGE_DASH_PORT", "8888"))

CAMERA_TARGET_TOTAL = 7
CAMERA_LIMITS = {
    "orangepi": 4,
    "rasppi": 3,
}

DRIVE_ACTIONS = {"forward", "backward", "left", "right", "stop"}

_motor_command_state: dict[str, Any] = {
    "last_action": "stop",
    "last_ok": None,
    "last_error": None,
    "last_response": None,
    "last_ts": None,
}


def _coerce_camera_list(raw: Any, *, device: str | None = None) -> list[str]:
    if not isinstance(raw, list):
        return []

    names = [str(item) for item in raw if item is not None]
    limit = CAMERA_LIMITS.get(device)
    if limit is not None:
        return names[:limit]
    return names


def _safe_state(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return {}


def _make_camera_url(device_info: dict[str, Any], camera_name: str, index: int) -> str:
    ip = device_info.get("ip")
    port = device_info.get("cam_port")
    if not ip or not port:
        return ""

    if device_info.get("device") == "orangepi":
        return f"http://{ip}:{port}/cam/{camera_name}"
    return f"http://{ip}:{port}/video_feed/{index}"


def _make_webrtc_url(device_info: dict[str, Any], camera_name: str) -> str:
    if device_info.get("device") != "orangepi":
        return ""
    ip = device_info.get("ip")
    webrtc_port = device_info.get("webrtc_port")
    if not ip or not webrtc_port:
        return ""
    return f"http://{ip}:{webrtc_port}/{camera_name}/whep"


def _make_drive_url(device_info: dict[str, Any]) -> str:
    urls = _make_drive_urls(device_info)
    return urls[0] if urls else ""


def _make_drive_urls(device_info: dict[str, Any]) -> list[str]:
    if device_info.get("device") != "orangepi":
        return []

    ip = device_info.get("ip")
    if not ip:
        return []

    ports: list[int] = []
    for key in ("drive_port", "dashboard_port", "cam_port"):
        raw_port = device_info.get(key)
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            continue
        if port > 0 and port not in ports:
            ports.append(port)

    if ORANGE_DASH_PORT not in ports:
        ports.append(ORANGE_DASH_PORT)

    return [f"http://{ip}:{port}/api/drive" for port in ports]


def _device_status(device_info: dict[str, Any]) -> str:
    last_register = float(device_info.get("last_register") or 0.0)
    last_data = float(device_info.get("last_data") or 0.0)
    heartbeat = max(last_register, last_data)

    if heartbeat <= 0:
        return "offline"
    if time.time() - heartbeat > STATUS_TIMEOUT:
        return "offline"
    return "online"


def _build_device_view(device_info: dict[str, Any]) -> dict[str, Any]:
    device_name = str(device_info.get("device") or "")
    camera_names = _coerce_camera_list(device_info.get("cameras"), device=device_name)

    cameras = [
        {
            "name": name,
            "url": _make_camera_url(device_info, name, idx),
        }
        for idx, name in enumerate(camera_names)
    ]

    return {
        "device": device_name,
        "ip": device_info.get("ip"),
        "cam_port": device_info.get("cam_port"),
        "status": _device_status(device_info),
        "last_register": device_info.get("last_register"),
        "last_data": device_info.get("last_data"),
        "cameras": cameras,
        "state": _safe_state(device_info.get("state")),
    }


def _collect_cameras(devices: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    camera_cards: list[dict[str, Any]] = []

    for device_key, title in (("orangepi", "OrangePi"), ("rasppi", "Raspberry Pi")):
        info = devices.get(device_key)
        if not info:
            continue

        names = _coerce_camera_list(info.get("cameras"), device=device_key)
        for index, name in enumerate(names):
            camera_cards.append(
                {
                    "id": f"{device_key}-{index}",
                    "device": device_key,
                    "title": title,
                    "name": name,
                    "index": index,
                    "url": _make_camera_url(info, name, index),
                    "webrtc_url": _make_webrtc_url(info, name),
                }
            )

    return camera_cards


def _build_summary_locked() -> dict[str, Any]:
    now = time.time()

    orangepi = _devices.get("orangepi", {})
    rasppi = _devices.get("rasppi", {})

    op_state = _safe_state(orangepi.get("state"))
    rp_state = _safe_state(rasppi.get("state"))

    gps = _safe_state(op_state.get("gps"))
    imu = _safe_state(op_state.get("imu"))
    encoders = _safe_state(op_state.get("encoders"))

    op_view = _build_device_view(orangepi) if orangepi else {}
    rp_view = _build_device_view(rasppi) if rasppi else {}

    motor_online_raw = op_state.get("motor_online")
    if isinstance(motor_online_raw, bool):
        motor_online = motor_online_raw
    else:
        motor_online = False

    drive_urls = _make_drive_urls(orangepi) if orangepi else []
    drive_url = drive_urls[0] if drive_urls else ""

    summary = {
        "now": now,
        "status_timeout_s": STATUS_TIMEOUT,
        "device_count": len(_devices),
        "device_status": {
            "orangepi": op_view.get("status", "offline"),
            "rasppi": rp_view.get("status", "offline"),
        },
        "devices": {
            "orangepi": {
                "ip": op_view.get("ip"),
                "cam_port": op_view.get("cam_port"),
                "status": op_view.get("status", "offline"),
                "last_register": op_view.get("last_register"),
                "last_data": op_view.get("last_data"),
            },
            "rasppi": {
                "ip": rp_view.get("ip"),
                "cam_port": rp_view.get("cam_port"),
                "status": rp_view.get("status", "offline"),
                "last_register": rp_view.get("last_register"),
                "last_data": rp_view.get("last_data"),
            },
        },
        "cameras": _collect_cameras(_devices),
        "camera_target": CAMERA_TARGET_TOTAL,
        "gps": gps,
        "imu": imu,
        "encoders": encoders,
        "motor": {
            "online": motor_online,
            "controllable": bool(drive_urls),
            "target": drive_url,
            "targets": drive_urls,
            "last_action": _motor_command_state.get("last_action"),
            "last_ok": _motor_command_state.get("last_ok"),
            "last_error": _motor_command_state.get("last_error"),
            "last_response": _motor_command_state.get("last_response"),
            "last_ts": _motor_command_state.get("last_ts"),
        },
        "rasppi": {
            "camera_count": rp_state.get("camera_count"),
            "camera_names": rp_state.get("camera_names") or [],
        },
    }

    return summary


def _decode_json_bytes(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        return {"value": data}
    except json.JSONDecodeError:
        return {"raw": text[:400]}


@app.route("/api/register", methods=["POST"])
def api_register() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid json"}), 400

    device = str(payload.get("device") or "").strip().lower()
    if not device:
        return jsonify({"error": "missing device"}), 400

    with _device_lock:
        previous = _devices.get(device, {})

        _devices[device] = {
            "device": device,
            "ip": payload.get("ip") or previous.get("ip"),
            "cam_port": payload.get("cam_port")
            if payload.get("cam_port") is not None
            else previous.get("cam_port"),
            "dashboard_port": payload.get("dashboard_port")
            if payload.get("dashboard_port") is not None
            else previous.get("dashboard_port"),
            "drive_port": payload.get("drive_port")
            if payload.get("drive_port") is not None
            else previous.get("drive_port"),
            "webrtc_port": payload.get("webrtc_port")
            if payload.get("webrtc_port") is not None
            else previous.get("webrtc_port"),
            "cameras": _coerce_camera_list(payload.get("cameras"), device=device)
            if payload.get("cameras") is not None
            else _coerce_camera_list(previous.get("cameras"), device=device),
            "last_register": time.time(),
            "last_data": previous.get("last_data"),
            "state": _safe_state(previous.get("state")),
        }

    return jsonify({"status": "ok"})


@app.route("/api/data", methods=["POST"])
def api_data() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid json"}), 400

    device = str(payload.get("device") or "").strip().lower()
    if not device:
        return jsonify({"error": "missing device"}), 400

    sensors = payload.get("sensors", {})
    sensors = sensors if isinstance(sensors, dict) else {}

    with _device_lock:
        info = _devices.setdefault(
            device,
            {
                "device": device,
                "ip": payload.get("ip"),
                "cam_port": payload.get("cam_port"),
                "dashboard_port": payload.get("dashboard_port"),
                "drive_port": payload.get("drive_port"),
                "cameras": _coerce_camera_list(payload.get("cameras"), device=device),
                "last_register": 0.0,
                "last_data": None,
                "state": {},
            },
        )

        info["state"] = sensors
        info["last_data"] = time.time()

        if payload.get("ip"):
            info["ip"] = payload.get("ip")
        if payload.get("cam_port") is not None:
            info["cam_port"] = payload.get("cam_port")
        if payload.get("dashboard_port") is not None:
            info["dashboard_port"] = payload.get("dashboard_port")
        if payload.get("drive_port") is not None:
            info["drive_port"] = payload.get("drive_port")
        if payload.get("webrtc_port") is not None:
            info["webrtc_port"] = payload.get("webrtc_port")
        if payload.get("cameras") is not None:
            info["cameras"] = _coerce_camera_list(payload.get("cameras"), device=device)

    return jsonify({"status": "ok"})


@app.route("/api/devices")
def api_devices() -> Response:
    with _device_lock:
        response = [_build_device_view(device_info) for device_info in _devices.values()]
    return jsonify({"devices": response, "now": time.time()})


@app.route("/api/summary")
def api_summary() -> Response:
    with _device_lock:
        summary = _build_summary_locked()
    return jsonify(summary)


@app.route("/api/drive", methods=["POST"])
def api_drive() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid json"}), 400

    action = str(payload.get("action") or "stop").strip().lower()
    if action not in DRIVE_ACTIONS:
        return jsonify({"error": f"invalid action: {action}"}), 400

    target_device = str(payload.get("device") or "orangepi").strip().lower()

    with _device_lock:
        info = _devices.get(target_device)
        if not info:
            _motor_command_state.update(
                {
                    "last_action": action,
                    "last_ok": False,
                    "last_error": f"device not registered: {target_device}",
                    "last_response": None,
                    "last_ts": time.time(),
                }
            )
            return jsonify({"error": f"device not registered: {target_device}"}), 404

        drive_urls = _make_drive_urls(info)

    if not drive_urls:
        with _device_lock:
            _motor_command_state.update(
                {
                    "last_action": action,
                    "last_ok": False,
                    "last_error": "drive endpoint unavailable",
                    "last_response": None,
                    "last_ts": time.time(),
                }
            )
        return jsonify({"error": "drive endpoint unavailable"}), 503

    body = json.dumps({"action": action}).encode("utf-8")
    last_error = "all candidate drive endpoints failed"
    attempted: list[str] = []

    for drive_url in drive_urls:
        attempted.append(drive_url)
        req = urllib.request.Request(
            drive_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                raw = resp.read()
                decoded = _decode_json_bytes(raw)
                result = {
                    "ok": True,
                    "status": resp.getcode(),
                    "target": drive_url,
                    "attempted": attempted,
                    "response": decoded,
                }

            with _device_lock:
                _motor_command_state.update(
                    {
                        "last_action": action,
                        "last_ok": True,
                        "last_error": None,
                        "last_response": decoded,
                        "last_ts": time.time(),
                    }
                )

            return jsonify(result)

        except urllib.error.HTTPError as exc:
            decoded = _decode_json_bytes(exc.read())
            # Endpoint exists but not here; try next candidate.
            if exc.code in (404, 405):
                last_error = f"{drive_url} returned HTTP {exc.code}"
                continue

            with _device_lock:
                _motor_command_state.update(
                    {
                        "last_action": action,
                        "last_ok": False,
                        "last_error": f"target HTTP {exc.code}",
                        "last_response": decoded,
                        "last_ts": time.time(),
                    }
                )
            return (
                jsonify(
                    {
                        "error": f"target returned HTTP {exc.code}",
                        "target": drive_url,
                        "attempted": attempted,
                        "response": decoded,
                    }
                ),
                502,
            )

        except Exception as exc:  # network / timeout
            last_error = str(exc)
            continue

    with _device_lock:
        _motor_command_state.update(
            {
                "last_action": action,
                "last_ok": False,
                "last_error": last_error,
                "last_response": {"attempted": attempted},
                "last_ts": time.time(),
            }
        )

    return jsonify({"error": f"drive proxy failed: {last_error}", "targets": attempted}), 502


@app.route("/api/power/<device>", methods=["POST"])
def api_power_proxy(device: str) -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid json"}), 400

    action = str(payload.get("action") or "").strip().lower()
    if action not in ("shutdown", "reboot"):
        return jsonify({"error": "invalid action"}), 400

    with _device_lock:
        info = _devices.get(device)
    if not info:
        return jsonify({"error": f"device not registered: {device}"}), 404

    ip = info.get("ip")
    cam_port = info.get("cam_port")
    if not ip or not cam_port:
        return jsonify({"error": "device has no known address"}), 503

    url = f"http://{ip}:{cam_port}/api/power"
    body = json.dumps({"action": action}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return jsonify(_decode_json_bytes(resp.read()))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/")
def index() -> str:
    return render_template_string(
        HTML_PAGE,
        status_timeout=STATUS_TIMEOUT,
        camera_target=CAMERA_TARGET_TOTAL,
    )


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Waste-E Central Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#060810;
  --surface:#0b0f1b;
  --surface2:#11192b;
  --border:#1f2b46;
  --border-hi:#2f4168;
  --cyan:#00d2ff;
  --violet:#6d28d9;
  --emerald:#22c55e;
  --amber:#f59e0b;
  --rose:#f43f5e;
  --text:#dbe9ff;
  --muted:#7c8da8;
  --mono:'JetBrains Mono',monospace;
  --radius:14px;
  --gap:10px;
}
html,body{
  width:100%;height:100%;
  background:radial-gradient(circle at 20% -20%,#1a1f35 0%,#060810 45%,#03050a 100%);
  color:var(--text);
  font-family:'Inter',sans-serif;
  font-size:13px;
  overflow:hidden;
}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border-hi);border-radius:8px}

#root{
  height:100vh;
  padding:var(--gap);
  display:grid;
  gap:var(--gap);
  grid-template-columns:1.2fr 1fr 340px;
  grid-template-rows:54px 1fr 330px;
  grid-template-areas:
    "hdr hdr hdr"
    "map imu motor"
    "cams cams encoder";
}

.panel{
  background:linear-gradient(180deg,#0d1425 0%, #0a0f1a 100%);
  border:1px solid var(--border);
  border-radius:var(--radius);
  overflow:hidden;
  display:flex;
  flex-direction:column;
  min-height:0;
}
.phdr{
  display:flex;
  align-items:center;
  gap:8px;
  padding:10px 14px;
  background:rgba(17,25,43,.9);
  border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.phdr h2{
  font-size:11px;
  font-weight:700;
  letter-spacing:.08em;
  text-transform:uppercase;
  color:var(--muted);
}
.phdr-icon{
  width:24px;
  height:24px;
  border-radius:7px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:12px;
}
.phdr-icon.cyan{background:#00d2ff22;border:1px solid #00d2ff50}
.phdr-icon.violet{background:#6d28d922;border:1px solid #6d28d950}
.phdr-icon.green{background:#22c55e22;border:1px solid #22c55e50}
.phdr-icon.amber{background:#f59e0b22;border:1px solid #f59e0b50}
.phdr-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.pbody{flex:1;min-height:0;position:relative;overflow:auto}

#hdr{
  grid-area:hdr;
  background:linear-gradient(90deg,#0d1425 0%,#0b1220 60%,#0d1425 100%);
  border:1px solid var(--border);
  border-radius:var(--radius);
  display:flex;
  align-items:center;
  gap:12px;
  padding:0 14px;
  position:relative;
  overflow:hidden;
}
#hdr::before{
  content:'';
  position:absolute;
  top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--violet),var(--cyan),var(--emerald),var(--cyan),var(--violet));
  background-size:300% 100%;
  animation:flow 7s linear infinite;
}
@keyframes flow{to{background-position:300% 0}}

.logo{display:flex;align-items:center;gap:8px}
.logo-badge{
  width:32px;height:32px;border-radius:9px;
  display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,var(--violet),var(--cyan));
  box-shadow:0 0 12px #6d28d980;
}
.logo-text{font-size:16px;font-weight:800;letter-spacing:-.02em}
.logo-text span{
  background:linear-gradient(90deg,var(--cyan),#8b5cf6);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
}

.chip{
  display:flex;
  align-items:center;
  gap:6px;
  padding:4px 10px;
  border-radius:999px;
  font-size:11px;
  font-weight:600;
  border:1px solid transparent;
}
.dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.chip.online{background:#06271a;border-color:#22c55e50;color:var(--emerald)}
.chip.online .dot{background:var(--emerald);box-shadow:0 0 8px var(--emerald);animation:pulseg 2s infinite}
.chip.offline{background:#2c0a13;border-color:#f43f5e50;color:var(--rose)}
.chip.offline .dot{background:var(--rose)}
.chip.warn{background:#2b1a05;border-color:#f59e0b50;color:var(--amber)}
.chip.warn .dot{background:var(--amber);box-shadow:0 0 8px var(--amber)}
@keyframes pulseg{0%,100%{opacity:1}50%{opacity:.45}}

#hdr-right{margin-left:auto;display:flex;align-items:center;gap:10px}
#clock{font-family:var(--mono);font-size:12px;color:var(--muted);min-width:88px;text-align:right}
#last-refresh{font-family:var(--mono);font-size:10px;color:var(--muted)}

#map-panel{grid-area:map}
#map{width:100%;height:100%;background:#0a1020}
.map-placeholder{
  position:absolute;inset:0;
  display:flex;align-items:center;justify-content:center;
  font-size:12px;color:var(--muted);
}
.leaflet-container{background:#0a1020}
.leaflet-tile-pane{filter:invert(1) hue-rotate(180deg) brightness(.82) saturate(.7)}
.leaflet-control-zoom a{
  background:var(--surface2)!important;
  border-color:var(--border)!important;
  color:var(--text)!important;
}
#gps-bar{
  display:grid;
  grid-template-columns:repeat(6,minmax(80px,1fr));
  border-top:1px solid var(--border);
  background:rgba(16,24,41,.95);
}
.gstat{
  padding:8px 10px;
  border-right:1px solid var(--border);
  display:flex;
  flex-direction:column;
  gap:2px;
}
.gstat:last-child{border-right:none}
.glbl{font-size:9px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:700}
.gval{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--text)}
.gval.hi{color:var(--cyan)}

#imu-panel{grid-area:imu}
#imu-body{
  display:grid;
  grid-template-columns:1fr;
  grid-template-rows:1fr 1fr;
  gap:10px;
  padding:10px;
  min-height:0;
}
.imu-block{
  background:rgba(17,25,43,.75);
  border:1px solid var(--border);
  border-radius:10px;
  padding:10px;
  display:flex;
  gap:12px;
  align-items:center;
  min-height:0;
}
#compass-canvas{width:130px;height:130px;flex-shrink:0}
#bird-canvas{width:150px;height:130px;flex-shrink:0}
.imu-meta{display:flex;flex-direction:column;gap:8px;min-width:0}
#heading-big{font-family:var(--mono);font-size:30px;font-weight:700;color:var(--cyan);line-height:1}
#heading-dir{font-size:11px;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;font-weight:700}
.imu-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 10px}
.imu-stat{display:flex;flex-direction:column;gap:1px}
.imu-stat .lbl{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:700}
.imu-stat .val{font-family:var(--mono);font-size:12px;color:var(--text);font-weight:600}
#cal-row{display:flex;align-items:center;gap:6px;padding:0 10px 10px}
.cal-dot{
  min-width:30px;height:22px;border-radius:6px;
  border:1px solid transparent;
  display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:10px;font-weight:700;
}
.cal-dot.c0{background:#131b2f;border-color:#1d2844;color:#3f4e74}
.cal-dot.c1{background:#2c1a05;border-color:#78350f;color:#f59e0b}
.cal-dot.c2{background:#0f2a1a;border-color:#1f7a48;color:#34d399}
.cal-dot.c3{background:#063322;border-color:#22c55e;color:#22c55e;box-shadow:0 0 8px #22c55e70}
#imu-temp{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--muted)}

#motor-panel{grid-area:motor}
#motor-wrap{display:flex;flex-direction:column;gap:10px;padding:10px;height:100%}
#drive-status{
  font-family:var(--mono);
  font-size:12px;
  letter-spacing:.06em;
  color:var(--muted);
  text-transform:uppercase;
}
#drive-grid{
  margin-top:2px;
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:8px;
}
.drive-btn{
  border:1px solid var(--border-hi);
  border-radius:10px;
  background:#101a2d;
  color:var(--text);
  height:52px;
  font-size:12px;
  font-weight:700;
  letter-spacing:.04em;
  cursor:pointer;
  transition:transform .12s, border-color .12s, background .12s;
}
.drive-btn:hover{border-color:var(--cyan)}
.drive-btn:active{transform:translateY(1px)}
.drive-btn.active{background:#00d2ff1d;border-color:#00d2ff;color:var(--cyan)}
.drive-btn.stop{background:#3a121a;border-color:#7f1d2d;color:#fda4af}
.drive-btn:disabled{opacity:.25;cursor:not-allowed;filter:grayscale(.6)}
#motor-offline-banner{
  display:none;
  background:#1e0a0e;
  border:1px solid #7f1d2d;
  border-radius:8px;
  padding:8px 12px;
  font-size:11px;
  font-weight:600;
  color:#fca5a5;
  letter-spacing:.06em;
  text-align:center;
}
#motor-meta{
  margin-top:auto;
  background:rgba(17,25,43,.75);
  border:1px solid var(--border);
  border-radius:10px;
  padding:10px;
  display:grid;
  gap:6px;
}
.meta-row{display:flex;justify-content:space-between;gap:8px}
.meta-row .k{color:var(--muted);font-size:10px;letter-spacing:.08em;text-transform:uppercase;font-weight:700}
.meta-row .v{color:var(--text);font-family:var(--mono);font-size:11px;text-align:right;word-break:break-all}

#cams-panel{grid-area:cams}
#cams-grid{
  height:100%;
  overflow:auto;
  display:grid;
  gap:8px;
  grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
  padding:8px;
}
.cam-card{
  position:relative;
  aspect-ratio:16/9;
  border-radius:10px;
  border:1px solid var(--border);
  overflow:hidden;
  background:#03060d;
}
.cam-card img,.cam-card video{width:100%;height:100%;object-fit:cover;display:block;background:#03060d}
.cam-overlay{
  position:absolute;
  top:0;left:0;right:0;
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:8px;
  padding:6px 10px;
  background:linear-gradient(180deg,rgba(0,0,0,.78) 0%,transparent 100%);
  font-size:10px;
  color:#fff;
}
.cam-overlay .name{font-weight:700;letter-spacing:.03em}
.cam-overlay .device{color:#b4c4e0;font-family:var(--mono)}
.cam-footer{
  position:absolute;
  left:0;right:0;bottom:0;
  padding:5px 10px;
  background:linear-gradient(0deg,rgba(0,0,0,.76) 0%,transparent 100%);
  color:#b8c7e3;
  font-family:var(--mono);
  font-size:9px;
}
.cam-empty{
  display:flex;
  align-items:center;
  justify-content:center;
  color:var(--muted);
  font-size:12px;
  border-style:dashed;
}
.cam-down{
  position:absolute;
  inset:0;
  display:none;
  align-items:center;
  justify-content:center;
  color:#d1d5db;
  font-size:12px;
  background:linear-gradient(135deg,#111827,#0b1220);
}
.cam-card.down .cam-down{display:flex}

#enc-panel{grid-area:encoder}
#enc-wrap{padding:10px;display:grid;grid-template-columns:1fr 1fr;gap:8px}
.enc-card{
  background:rgba(17,25,43,.75);
  border:1px solid var(--border);
  border-radius:10px;
  padding:10px;
  display:flex;
  flex-direction:column;
  gap:2px;
}
.enc-card .lbl{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:700}
.enc-card .val{font-family:var(--mono);font-size:20px;font-weight:700;color:var(--text)}
.enc-card .unit{font-size:10px;color:var(--muted)}
.enc-card.cyan .val{color:var(--cyan)}
.enc-card.green .val{color:var(--emerald)}
#encoder-note{
  grid-column:1/-1;
  margin-top:2px;
  border:1px solid var(--border);
  border-radius:10px;
  padding:9px 12px;
  background:rgba(17,25,43,.55);
  color:var(--muted);
  font-size:11px;
  line-height:1.45;
  display:flex;
  align-items:center;
  gap:8px;
}
.no-fix-badge{
  display:inline-block;
  padding:1px 7px;
  border-radius:999px;
  font-size:9px;
  font-weight:700;
  letter-spacing:.08em;
  background:#2c0a13;
  border:1px solid #f43f5e50;
  color:var(--rose);
  vertical-align:middle;
  margin-left:6px;
}
.fix-badge{
  display:inline-block;
  padding:1px 7px;
  border-radius:999px;
  font-size:9px;
  font-weight:700;
  letter-spacing:.08em;
  background:#06271a;
  border:1px solid #22c55e50;
  color:var(--emerald);
  vertical-align:middle;
  margin-left:6px;
}
.status-val{font-family:var(--mono);font-size:11px;font-weight:700}
.status-val.online{color:var(--emerald)}
.status-val.offline{color:var(--rose)}
.status-val.warn{color:var(--amber)}

.live-dot{
  width:6px;height:6px;border-radius:50%;
  background:var(--emerald);
  box-shadow:0 0 8px var(--emerald);
  animation:pulseg 2s infinite;
}
.pwr-group{display:flex;gap:3px;align-items:center}
.pwr-btn{
  width:24px;height:24px;border-radius:6px;
  background:#0f1928;border:1px solid var(--border-hi);
  color:var(--muted);font-size:13px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:all .12s;padding:0;
}
.pwr-btn:hover{border-color:var(--amber);color:var(--amber)}
.pwr-btn.off:hover{border-color:var(--rose);color:var(--rose)}

@media (max-width:1400px){
  #root{
    grid-template-columns:1fr 1fr;
    grid-template-rows:54px 380px 380px 320px 320px;
    grid-template-areas:
      "hdr hdr"
      "map imu"
      "cams cams"
      "motor encoder"
      "motor encoder";
  }
}
@media (max-width:900px){
  html,body{overflow:auto}
  #root{
    height:auto;
    min-height:100vh;
    grid-template-columns:1fr;
    grid-template-rows:54px 360px 360px 360px 300px 300px;
    grid-template-areas:
      "hdr"
      "map"
      "imu"
      "cams"
      "motor"
      "encoder";
  }
  #gps-bar{grid-template-columns:repeat(3,minmax(80px,1fr))}
}
</style>
</head>
<body>
<div id="root">
  <header id="hdr">
    <div class="logo">
      <div class="logo-badge">♻</div>
      <div class="logo-text">Waste<span>-E</span> Central</div>
    </div>

    <div class="chip offline" id="chip-opi"><div class="dot"></div>OrangePi offline</div>
    <div class="pwr-group">
      <button class="pwr-btn" onclick="doPower('orangepi','reboot')" title="Reboot OrangePi">↺</button>
      <button class="pwr-btn off" onclick="doPower('orangepi','shutdown')" title="Shutdown OrangePi">⏻</button>
    </div>
    <div class="chip offline" id="chip-rpi"><div class="dot"></div>Raspberry Pi offline</div>
    <div class="pwr-group">
      <button class="pwr-btn" onclick="doPower('rasppi','reboot')" title="Reboot Raspberry Pi">↺</button>
      <button class="pwr-btn off" onclick="doPower('rasppi','shutdown')" title="Shutdown Raspberry Pi">⏻</button>
    </div>
    <div class="chip warn" id="chip-cams"><div class="dot"></div>0 / {{ camera_target }} cameras</div>

    <div id="hdr-right">
      <div id="last-refresh">waiting for telemetry...</div>
      <div id="clock">--:--:--</div>
    </div>
  </header>

  <section class="panel" id="map-panel">
    <div class="phdr">
      <div class="phdr-icon cyan">🗺</div>
      <h2>GPS Map</h2>
      <div class="phdr-right"><div class="live-dot" id="gps-live"></div></div>
    </div>
    <div class="pbody">
      <div id="map"></div>
      <div class="map-placeholder" id="map-placeholder" style="display:none">Map unavailable</div>
    </div>
    <div id="gps-bar">
      <div class="gstat"><div class="glbl">Latitude</div><div class="gval hi" id="g-lat">-</div></div>
      <div class="gstat"><div class="glbl">Longitude</div><div class="gval hi" id="g-lon">-</div></div>
      <div class="gstat"><div class="glbl">Altitude</div><div class="gval" id="g-alt">-</div></div>
      <div class="gstat"><div class="glbl">Speed</div><div class="gval" id="g-spd">-</div></div>
      <div class="gstat"><div class="glbl">Heading</div><div class="gval" id="g-hdg">-</div></div>
      <div class="gstat"><div class="glbl">Satellites <span id="g-fix-badge" class="no-fix-badge">NO FIX</span></div><div class="gval" id="g-sat">-</div></div>
    </div>
  </section>

  <section class="panel" id="imu-panel">
    <div class="phdr">
      <div class="phdr-icon violet">🧭</div>
      <h2>IMU + Bird-Eye Orientation</h2>
      <div class="phdr-right"><div class="live-dot" id="imu-live"></div></div>
    </div>
    <div id="imu-body">
      <div class="imu-block">
        <canvas id="compass-canvas" width="130" height="130"></canvas>
        <div class="imu-meta">
          <div id="heading-big">0.0°</div>
          <div id="heading-dir">North</div>
          <div class="imu-grid">
            <div class="imu-stat"><div class="lbl">Roll</div><div class="val" id="i-roll">0.0°</div></div>
            <div class="imu-stat"><div class="lbl">Pitch</div><div class="val" id="i-pitch">0.0°</div></div>
            <div class="imu-stat"><div class="lbl">QW</div><div class="val" id="i-qw">1.000</div></div>
            <div class="imu-stat"><div class="lbl">QX</div><div class="val" id="i-qx">0.000</div></div>
            <div class="imu-stat"><div class="lbl">QY</div><div class="val" id="i-qy">0.000</div></div>
            <div class="imu-stat"><div class="lbl">QZ</div><div class="val" id="i-qz">0.000</div></div>
          </div>
        </div>
      </div>

      <div class="imu-block">
        <canvas id="bird-canvas" width="150" height="130"></canvas>
        <div class="imu-meta">
          <div class="imu-grid">
            <div class="imu-stat"><div class="lbl">Heading</div><div class="val" id="i-heading">0.0°</div></div>
            <div class="imu-stat"><div class="lbl">Direction</div><div class="val" id="i-dir">N</div></div>
            <div class="imu-stat"><div class="lbl">Cal SYS</div><div class="val" id="i-cal-sys">0</div></div>
            <div class="imu-stat"><div class="lbl">Cal GYR</div><div class="val" id="i-cal-gyr">0</div></div>
            <div class="imu-stat"><div class="lbl">Cal ACC</div><div class="val" id="i-cal-acc">0</div></div>
            <div class="imu-stat"><div class="lbl">Cal MAG</div><div class="val" id="i-cal-mag">0</div></div>
          </div>
        </div>
      </div>
    </div>
    <div id="cal-row">
      <span style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:700">Cal</span>
      <div class="cal-dot c0" id="cal-sys">SYS</div>
      <div class="cal-dot c0" id="cal-gyr">GYR</div>
      <div class="cal-dot c0" id="cal-acc">ACC</div>
      <div class="cal-dot c0" id="cal-mag">MAG</div>
      <div id="imu-temp">--</div>
    </div>
  </section>

  <section class="panel" id="motor-panel">
    <div class="phdr">
      <div class="phdr-icon amber">🕹</div>
      <h2>Motor Controller</h2>
      <div class="phdr-right"><div id="drive-status">idle</div></div>
    </div>
    <div id="motor-wrap">
      <div id="motor-offline-banner">⚠ MOTOR DRIVER OFFLINE — check hardware</div>
      <div id="drive-grid">
        <button class="drive-btn" data-action="forward" style="grid-column:2">▲ FORWARD</button>
        <button class="drive-btn" data-action="left" style="grid-column:1;grid-row:2">◀ LEFT</button>
        <button class="drive-btn stop" data-action="stop" style="grid-column:2;grid-row:2">■ STOP</button>
        <button class="drive-btn" data-action="right" style="grid-column:3;grid-row:2">RIGHT ▶</button>
        <button class="drive-btn" data-action="backward" style="grid-column:2;grid-row:3">▼ BACKWARD</button>
      </div>
      <div style="font-size:10px;color:var(--muted);text-align:center;letter-spacing:.06em">WASD / Arrow keys</div>

      <div id="motor-meta">
        <div class="meta-row"><span class="k">Controller</span><span class="v status-val offline" id="m-online">offline</span></div>
        <div class="meta-row"><span class="k">Last Action</span><span class="v" id="m-last-action">stop</span></div>
        <div class="meta-row"><span class="k">Last Result</span><span class="v" id="m-last-result">-</span></div>
        <div class="meta-row"><span class="k">Target</span><span class="v" id="m-target">-</span></div>
        <div class="meta-row"><span class="k">Updated</span><span class="v" id="m-updated">-</span></div>
      </div>
    </div>
  </section>

  <section class="panel" id="cams-panel">
    <div class="phdr">
      <div class="phdr-icon green">📷</div>
      <h2>7-Camera Grid (4 OrangePi + 3 Raspberry Pi)</h2>
      <div class="phdr-right"><span style="font-family:var(--mono);font-size:10px;color:var(--muted)" id="cam-count">0 / {{ camera_target }}</span></div>
    </div>
    <div class="pbody">
      <div id="cams-grid"></div>
    </div>
  </section>

  <section class="panel" id="enc-panel">
    <div class="phdr">
      <div class="phdr-icon cyan">⚙</div>
      <h2>Motor Encoder Data</h2>
    </div>
    <div id="enc-wrap">
      <div class="enc-card cyan"><div class="lbl">Left RPM</div><div class="val" id="e-left-rpm">-</div><div class="unit">rpm</div></div>
      <div class="enc-card cyan"><div class="lbl">Right RPM</div><div class="val" id="e-right-rpm">-</div><div class="unit">rpm</div></div>
      <div class="enc-card green"><div class="lbl">Left Angle</div><div class="val" id="e-left-angle">-</div><div class="unit">deg</div></div>
      <div class="enc-card green"><div class="lbl">Right Angle</div><div class="val" id="e-right-angle">-</div><div class="unit">deg</div></div>
      <div id="encoder-note"><span>⚙</span><span id="encoder-note-text">Waiting for OrangePi encoder telemetry...</span></div>
    </div>
  </section>
</div>

<script>
const STATUS_TIMEOUT = {{ status_timeout|tojson }};
const CAMERA_TARGET = {{ camera_target|tojson }};

const fmt = (v, d=2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? '-' : Number(v).toFixed(d);
const deg = (v) => (v === null || v === undefined || Number.isNaN(Number(v))) ? '-' : Number(v).toFixed(1) + '°';
const mono = (v) => (v === null || v === undefined || v === '') ? '-' : String(v);

const compassCanvas = document.getElementById('compass-canvas');
const compassCtx = compassCanvas.getContext('2d');
const birdCanvas = document.getElementById('bird-canvas');
const birdCtx = birdCanvas.getContext('2d');

let map = null;
let marker = null;
let trail = [];
let trailLine = null;
let mapReady = false;
let cameraSignature = '';
let currentAction = 'stop';
let sendingDrive = false;
const _peerConns = {};

async function _startWhep(camId, whepUrl) {
  if (_peerConns[camId]) { try { _peerConns[camId].close(); } catch(_){} }
  const pc = new RTCPeerConnection({ iceServers: [] });
  _peerConns[camId] = pc;
  pc.addTransceiver('video', { direction: 'recvonly' });
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await new Promise(resolve => {
    if (pc.iceGatheringState === 'complete') { resolve(); return; }
    const h = () => { if (pc.iceGatheringState === 'complete') { pc.removeEventListener('icegatheringstatechange', h); resolve(); } };
    pc.addEventListener('icegatheringstatechange', h);
    setTimeout(resolve, 3000);
  });
  try {
    const r = await fetch(whepUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/sdp' },
      body: pc.localDescription.sdp,
    });
    if (!r.ok) throw new Error('WHEP ' + r.status);
    await pc.setRemoteDescription({ type: 'answer', sdp: await r.text() });
  } catch(err) {
    const card = document.querySelector('[data-camid="' + camId + '"]');
    if (card) card.classList.add('down');
    return;
  }
  pc.ontrack = (e) => {
    const vid = document.getElementById('vid-' + camId);
    if (vid) vid.srcObject = e.streams[0] || new MediaStream([e.track]);
  };
  pc.onconnectionstatechange = () => {
    if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
      const card = document.querySelector('[data-camid="' + camId + '"]');
      if (card) card.classList.add('down');
      setTimeout(() => _startWhep(camId, whepUrl), 3000);
    }
  };
}

const robotIconHtml = `<div style="
  width:34px;height:34px;border-radius:50%;
  background:conic-gradient(from 190deg,#6d28d9,#00d2ff,#6d28d9);
  border:2px solid rgba(255,255,255,.9);
  box-shadow:0 0 0 3px #00d2ff40,0 4px 18px #00d2ff60;
  display:flex;align-items:center;justify-content:center;font-size:16px;
">🤖</div>`;

function headingToDir(h){
  const dirs = ['N','NE','E','SE','S','SW','W','NW'];
  const norm = ((Number(h) || 0) % 360 + 360) % 360;
  return dirs[Math.round(norm / 45) % 8];
}

function setChip(id, status, text){
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'chip ' + (status || 'offline');
  el.innerHTML = `<div class="dot"></div>${text}`;
}

function setLiveDot(id, ok){
  const dot = document.getElementById(id);
  if (!dot) return;
  if (ok){
    dot.style.background = 'var(--emerald)';
    dot.style.boxShadow = '0 0 8px var(--emerald)';
    dot.style.animation = 'pulseg 2s infinite';
  } else {
    dot.style.background = 'var(--rose)';
    dot.style.boxShadow = 'none';
    dot.style.animation = 'none';
  }
}

function clsCal(v){
  const x = Math.max(0, Math.min(3, Number(v) || 0));
  return 'cal-dot c' + x;
}

function initMap(){
  if (typeof L === 'undefined'){
    document.getElementById('map-placeholder').style.display = 'flex';
    return;
  }

  map = L.map('map', {
    center: [13.75, 100.52],
    zoom: 16,
    zoomControl: true,
    attributionControl: false,
  });

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
  }).addTo(map);
}

function updateMap(lat, lon){
  if (!map) return;
  if (lat === null || lat === undefined || lon === null || lon === undefined) return;

  const latNum = Number(lat);
  const lonNum = Number(lon);
  if (Number.isNaN(latNum) || Number.isNaN(lonNum)) return;

  if (!mapReady){
    map.setView([latNum, lonNum], 17);
    mapReady = true;
  }

  if (!marker){
    marker = L.marker([latNum, lonNum], {
      icon: L.divIcon({
        className: '',
        html: robotIconHtml,
        iconSize: [34,34],
        iconAnchor: [17,17],
      }),
    }).addTo(map);
  } else {
    marker.setLatLng([latNum, lonNum]);
  }

  trail.push([latNum, lonNum]);
  if (trail.length > 200) trail.shift();
  if (trailLine) map.removeLayer(trailLine);
  trailLine = L.polyline(trail, {
    color: '#00d2ff',
    weight: 2,
    opacity: .45,
    dashArray: '5 4',
  }).addTo(map);
}

function drawCompass(heading){
  const ctx = compassCtx;
  const canvas = compassCanvas;
  const w = canvas.width;
  const h = canvas.height;
  const r = w / 2;

  ctx.clearRect(0, 0, w, h);

  const glow = ctx.createRadialGradient(r, r, r * 0.35, r, r, r);
  glow.addColorStop(0, '#00d2ff14');
  glow.addColorStop(1, 'transparent');
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(r, r, r, 0, Math.PI * 2);
  ctx.fill();

  ctx.beginPath();
  ctx.arc(r, r, r - 2, 0, Math.PI * 2);
  ctx.strokeStyle = '#233252';
  ctx.lineWidth = 2;
  ctx.stroke();

  for (let i = 0; i < 72; i++){
    const a = (i * 5 - 90) * Math.PI / 180;
    const major = i % 18 === 0;
    const medium = i % 9 === 0;
    const len = major ? 12 : (medium ? 8 : 4);

    ctx.beginPath();
    ctx.moveTo(r + Math.cos(a) * (r - 3), r + Math.sin(a) * (r - 3));
    ctx.lineTo(r + Math.cos(a) * (r - 3 - len), r + Math.sin(a) * (r - 3 - len));
    ctx.strokeStyle = major ? '#00d2ff' : (medium ? '#35507f' : '#1e2c47');
    ctx.lineWidth = major ? 2 : 1;
    ctx.stroke();
  }

  const labels = [
    ['N', -90, '#f43f5e'], ['E', 0, '#9db6e5'], ['S', 90, '#9db6e5'], ['W', 180, '#9db6e5'],
    ['NE', -45, '#4b618f'], ['SE', 45, '#4b618f'], ['SW', 135, '#4b618f'], ['NW', -135, '#4b618f'],
  ];

  labels.forEach(([label, angle, color]) => {
    const rr = label.length === 1 ? r - 24 : r - 28;
    const rad = Number(angle) * Math.PI / 180;
    ctx.fillStyle = String(color);
    ctx.font = `bold ${label.length === 1 ? 10 : 8}px Inter`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(label), r + Math.cos(rad) * rr, r + Math.sin(rad) * rr);
  });

  const hRad = ((Number(heading) || 0) - 90) * Math.PI / 180;
  ctx.save();
  ctx.translate(r, r);
  ctx.rotate(hRad);

  ctx.fillStyle = '#6d28d92a';
  ctx.strokeStyle = '#7c3aed';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.roundRect(-12, -18, 24, 36, 3);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#00d2ff';
  ctx.shadowColor = '#00d2ff';
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(0, -24);
  ctx.lineTo(-6, -16);
  ctx.lineTo(6, -16);
  ctx.closePath();
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.restore();

  ctx.beginPath();
  ctx.arc(r, r, 3.5, 0, Math.PI * 2);
  ctx.fillStyle = '#00d2ff';
  ctx.fill();
}

function drawBirdEye(heading, roll, pitch){
  const ctx = birdCtx;
  const canvas = birdCanvas;
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;

  ctx.clearRect(0, 0, w, h);

  for (let x = 10; x < w; x += 20){
    ctx.strokeStyle = '#1b2740';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 6);
    ctx.lineTo(x, h - 6);
    ctx.stroke();
  }
  for (let y = 10; y < h; y += 20){
    ctx.beginPath();
    ctx.moveTo(6, y);
    ctx.lineTo(w - 6, y);
    ctx.stroke();
  }

  ctx.save();
  const yaw = (Number(heading) || 0) * Math.PI / 180;
  ctx.translate(cx, cy);
  ctx.rotate(yaw);

  ctx.fillStyle = '#0b1c31';
  ctx.strokeStyle = '#00d2ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(-22, -34, 44, 68, 8);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#7c3aed';
  ctx.fillRect(-15, -18, 30, 36);

  ctx.fillStyle = '#00d2ff';
  ctx.beginPath();
  ctx.moveTo(0, -42);
  ctx.lineTo(-8, -31);
  ctx.lineTo(8, -31);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  const rollPct = Math.max(-1, Math.min(1, (Number(roll) || 0) / 45));
  const pitchPct = Math.max(-1, Math.min(1, (Number(pitch) || 0) / 45));

  ctx.fillStyle = '#1a2744';
  ctx.fillRect(10, h - 18, w - 20, 8);
  ctx.fillRect(10, h - 32, w - 20, 8);

  ctx.fillStyle = '#00d2ff';
  ctx.fillRect(w / 2, h - 18, rollPct * (w / 2 - 10), 8);

  ctx.fillStyle = '#22c55e';
  ctx.fillRect(w / 2, h - 32, pitchPct * (w / 2 - 10), 8);

  ctx.fillStyle = '#8da2c8';
  ctx.font = '9px JetBrains Mono';
  ctx.fillText('ROLL', 10, h - 20);
  ctx.fillText('PITCH', 10, h - 34);
}

function renderCameras(cameras){
  const bounded = [...(cameras || [])].slice(0, CAMERA_TARGET);
  const signature = bounded.map(c => `${c.id}|${c.url}|${c.webrtc_url}|${c.name}`).join(';');
  if (signature === cameraSignature) return;
  cameraSignature = signature;

  const grid = document.getElementById('cams-grid');
  const cards = [];

  bounded.forEach((cam) => {
    const title = cam && cam.title ? cam.title : 'Unknown';
    const name = cam && cam.name ? cam.name : '-';
    const url = cam && cam.url ? cam.url : '';
    const webrtcUrl = cam && cam.webrtc_url ? cam.webrtc_url : '';
    const useWebRTC = !!webrtcUrl;
    const slotText = cam && cam.device === 'rasppi'
      ? `Raspberry Pi · slot ${cam.index + 1}`
      : `OrangePi · slot ${cam.index + 1}`;
    const proto = useWebRTC ? 'WebRTC' : 'MJPEG';

    if (useWebRTC) {
      cards.push(`
        <div class="cam-card" data-camid="${cam.id}" data-whep="${webrtcUrl}">
          <video id="vid-${cam.id}" autoplay muted playsinline></video>
          <div class="cam-overlay">
            <span class="name">${name}</span>
            <span class="device">${title}</span>
          </div>
          <div class="cam-footer">${slotText} · ${proto}</div>
          <div class="cam-down">Stream unavailable</div>
        </div>
      `);
    } else {
      cards.push(`
        <div class="cam-card" data-camid="${cam.id}">
          <img src="${url}" alt="${name}" loading="lazy"
               onerror="this.parentElement.classList.add('down');" />
          <div class="cam-overlay">
            <span class="name">${name}</span>
            <span class="device">${title}</span>
          </div>
          <div class="cam-footer">${slotText} · ${proto}</div>
          <div class="cam-down">Stream unavailable</div>
        </div>
      `);
    }
  });

  for (let i = bounded.length; i < CAMERA_TARGET; i++){
    const expected = i < 4 ? 'OrangePi' : 'Raspberry Pi';
    const slot = i < 4 ? i + 1 : i - 3;
    cards.push(`
      <div class="cam-card cam-empty">
        <div>Waiting for ${expected} camera ${slot}</div>
      </div>
    `);
  }

  grid.innerHTML = cards.join('');

  bounded.forEach((cam) => {
    if (cam && cam.webrtc_url) {
      _startWhep(cam.id, cam.webrtc_url);
    }
  });
}

async function sendDrive(action){
  if (sendingDrive) return;
  sendingDrive = true;

  try {
    const resp = await fetch('/api/drive', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ action }),
    });

    const payload = await resp.json().catch(() => ({}));
    if (!resp.ok){
      const msg = payload.error || 'command failed';
      document.getElementById('drive-status').textContent = msg;
      document.getElementById('drive-status').style.color = 'var(--rose)';
      return;
    }

    currentAction = action;
    document.getElementById('drive-status').textContent = action.toUpperCase();
    document.getElementById('drive-status').style.color = action === 'stop' ? 'var(--muted)' : 'var(--emerald)';
  } catch (err) {
    document.getElementById('drive-status').textContent = 'proxy unreachable';
    document.getElementById('drive-status').style.color = 'var(--rose)';
  } finally {
    sendingDrive = false;
  }
}

function setActiveAction(action){
  document.querySelectorAll('.drive-btn').forEach((btn) => {
    if (btn.dataset.action === action){
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

function applySummary(data){
  const now = data.now ? new Date(data.now * 1000) : new Date();
  document.getElementById('clock').textContent = now.toLocaleTimeString();
  document.getElementById('last-refresh').textContent = `last refresh ${now.toLocaleTimeString()}`;

  const op = (data.devices || {}).orangepi || {};
  const rp = (data.devices || {}).rasppi || {};
  setChip('chip-opi', op.status === 'online' ? 'online' : 'offline', `OrangePi ${op.status || 'offline'}`);
  setChip('chip-rpi', rp.status === 'online' ? 'online' : 'offline', `Raspberry Pi ${rp.status || 'offline'}`);

  const cameraCount = (data.cameras || []).length;
  const camStatus = cameraCount >= CAMERA_TARGET ? 'online' : (cameraCount > 0 ? 'warn' : 'offline');
  setChip('chip-cams', camStatus, `${cameraCount} / ${CAMERA_TARGET} cameras`);
  document.getElementById('cam-count').textContent = `${cameraCount} / ${CAMERA_TARGET}`;

  renderCameras(data.cameras || []);

  const gps = data.gps || {};
  const hasFix = !!gps.fix && gps.lat !== null && gps.lon !== null;
  setLiveDot('gps-live', hasFix);

  const fixBadge = document.getElementById('g-fix-badge');
  if (fixBadge) {
    fixBadge.textContent = hasFix ? 'FIX' : 'NO FIX';
    fixBadge.className = hasFix ? 'fix-badge' : 'no-fix-badge';
  }
  const latEl = document.getElementById('g-lat');
  const lonEl = document.getElementById('g-lon');
  latEl.textContent = fmt(gps.lat, 6);
  lonEl.textContent = fmt(gps.lon, 6);
  latEl.style.color = hasFix ? 'var(--cyan)' : 'var(--muted)';
  lonEl.style.color = hasFix ? 'var(--cyan)' : 'var(--muted)';
  document.getElementById('g-alt').textContent = gps.alt != null ? `${fmt(gps.alt, 1)} m` : '-';
  document.getElementById('g-spd').textContent = gps.speed != null ? `${fmt(gps.speed, 1)} km/h` : '-';
  document.getElementById('g-hdg').textContent = deg(gps.heading);
  const satEl = document.getElementById('g-sat');
  satEl.textContent = mono(gps.satellites);
  satEl.style.color = hasFix ? 'var(--text)' : 'var(--rose)';
  updateMap(gps.lat, gps.lon);

  const imu = data.imu || {};
  const hasImu = Object.keys(imu).length > 0;
  const imuConnected = hasImu && (imu.temp !== null && imu.temp !== undefined);
  setLiveDot('imu-live', imuConnected);

  document.getElementById('heading-big').textContent = deg(imu.heading);
  document.getElementById('heading-dir').textContent = headingToDir(imu.heading);
  document.getElementById('i-roll').textContent = deg(imu.roll);
  document.getElementById('i-pitch').textContent = deg(imu.pitch);
  document.getElementById('i-qw').textContent = fmt(imu.qw, 3);
  document.getElementById('i-qx').textContent = fmt(imu.qx, 3);
  document.getElementById('i-qy').textContent = fmt(imu.qy, 3);
  document.getElementById('i-qz').textContent = fmt(imu.qz, 3);
  document.getElementById('i-heading').textContent = deg(imu.heading);
  document.getElementById('i-dir').textContent = headingToDir(imu.heading);

  document.getElementById('i-cal-sys').textContent = mono(imu.cal_sys);
  document.getElementById('i-cal-gyr').textContent = mono(imu.cal_gyro);
  document.getElementById('i-cal-acc').textContent = mono(imu.cal_accel);
  document.getElementById('i-cal-mag').textContent = mono(imu.cal_mag);

  document.getElementById('cal-sys').className = clsCal(imu.cal_sys);
  document.getElementById('cal-gyr').className = clsCal(imu.cal_gyro);
  document.getElementById('cal-acc').className = clsCal(imu.cal_accel);
  document.getElementById('cal-mag').className = clsCal(imu.cal_mag);

  document.getElementById('imu-temp').textContent = imu.temp != null ? `${imu.temp}°C` : '--';

  drawCompass(imu.heading || 0);
  drawBirdEye(imu.heading || 0, imu.roll || 0, imu.pitch || 0);

  const motor = data.motor || {};
  const canDrive = !!motor.controllable;
  document.querySelectorAll('.drive-btn').forEach((btn) => {
    btn.disabled = !canDrive;
  });

  const mOnlineEl = document.getElementById('m-online');
  if (motor.online) {
    mOnlineEl.textContent = 'online';
    mOnlineEl.className = 'v status-val online';
  } else {
    mOnlineEl.textContent = 'offline';
    mOnlineEl.className = 'v status-val offline';
  }
  const offlineBanner = document.getElementById('motor-offline-banner');
  if (offlineBanner) offlineBanner.style.display = (!motor.online && canDrive) ? 'block' : 'none';
  document.getElementById('m-last-action').textContent = mono(motor.last_action || currentAction);
  if (motor.last_ok === true) {
    document.getElementById('m-last-result').textContent = 'ok';
  } else if (motor.last_ok === false) {
    document.getElementById('m-last-result').textContent = mono(motor.last_error || 'error');
  } else {
    document.getElementById('m-last-result').textContent = '-';
  }
  document.getElementById('m-target').textContent = mono(motor.target);
  document.getElementById('m-updated').textContent = motor.last_ts ? new Date(motor.last_ts * 1000).toLocaleTimeString() : '-';

  const enc = data.encoders || {};
  document.getElementById('e-left-rpm').textContent = fmt(enc.left_rpm, 2);
  document.getElementById('e-right-rpm').textContent = fmt(enc.right_rpm, 2);
  document.getElementById('e-left-angle').textContent = fmt(enc.left_angle, 2);
  document.getElementById('e-right-angle').textContent = fmt(enc.right_angle, 2);

  const hasEncoder = ['left_rpm', 'right_rpm', 'left_angle', 'right_angle']
    .some((k) => enc[k] !== null && enc[k] !== undefined);
  const encNoteText = document.getElementById('encoder-note-text');
  if (hasEncoder){
    const lr = Number(enc.left_rpm || 0);
    const rr = Number(enc.right_rpm || 0);
    const drift = (lr - rr).toFixed(2);
    if (encNoteText) encNoteText.textContent = `L/R drift: ${drift} rpm  ·  updated ${now.toLocaleTimeString()}`;
  } else {
    if (encNoteText) encNoteText.textContent = 'Waiting for encoder data from OrangePi...';
  }

  if (canDrive) {
    document.getElementById('drive-status').textContent = (motor.last_action || currentAction || 'idle').toUpperCase();
    document.getElementById('drive-status').style.color = motor.last_action === 'stop' ? 'var(--muted)' : 'var(--emerald)';
  } else {
    document.getElementById('drive-status').textContent = 'controller unavailable';
    document.getElementById('drive-status').style.color = 'var(--rose)';
  }

  setActiveAction((motor.last_action || currentAction || 'stop'));
}

async function refresh(){
  try {
    const resp = await fetch('/api/summary', { cache: 'no-store' });
    if (!resp.ok) throw new Error('summary fetch failed');
    const data = await resp.json();
    applySummary(data);
  } catch (err) {
    document.getElementById('last-refresh').textContent = `telemetry error: ${err.message}`;
    setChip('chip-opi', 'offline', 'OrangePi offline');
    setChip('chip-rpi', 'offline', 'Raspberry Pi offline');
  }
}

async function doPower(device, action) {
  const label = device === 'orangepi' ? 'OrangePi' : 'Raspberry Pi';
  const verb = action === 'shutdown' ? 'Shutdown' : 'Reboot';
  if (!confirm(`${verb} ${label}?\n\nThe device will go offline.`)) return;
  try {
    const resp = await fetch(`/api/power/${device}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ action }),
    });
    const data = await resp.json().catch(() => ({}));
    if (resp.ok) {
      alert(`${verb} command sent to ${label}.`);
    } else {
      alert(`Error: ${data.error || resp.statusText}`);
    }
  } catch(err) {
    alert(`Could not reach dashboard: ${err.message}`);
  }
}

function wireControls(){
  document.querySelectorAll('.drive-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.action;
      if (!action) return;
      sendDrive(action);
      setActiveAction(action);
    });
  });

  window.addEventListener('keydown', (event) => {
    const key = event.key.toLowerCase();
    if (key === 'arrowup' || key === 'w') { sendDrive('forward'); setActiveAction('forward'); }
    if (key === 'arrowdown' || key === 's') { sendDrive('backward'); setActiveAction('backward'); }
    if (key === 'arrowleft' || key === 'a') { sendDrive('left'); setActiveAction('left'); }
    if (key === 'arrowright' || key === 'd') { sendDrive('right'); setActiveAction('right'); }
    if (key === ' ' || key === 'x') { sendDrive('stop'); setActiveAction('stop'); }
  });
}

initMap();
wireControls();
drawCompass(0);
drawBirdEye(0, 0, 0);
refresh();
setInterval(refresh, 500);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(f"Starting Waste-E central dashboard on http://0.0.0.0:{DEFAULT_PORT}")
    app.run(host="0.0.0.0", port=DEFAULT_PORT, threaded=True, debug=False)
