#!/usr/bin/env python3
"""
Central dashboard server for Waste-E hardware.

Receives registration and periodic data from Orange Pi and Raspberry Pi senders,
stores live device state in memory, and displays a simple dashboard website.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

_device_lock = threading.Lock()
_devices: dict[str, dict[str, Any]] = {}

DEFAULT_PORT = 9000
STATUS_TIMEOUT = 15.0


def _make_camera_url(device_info: dict[str, Any], camera_name: str, index: int) -> str:
    ip = device_info.get("ip")
    port = device_info.get("cam_port")
    if not ip or not port:
        return ""

    if device_info.get("device") == "orangepi":
        return f"http://{ip}:{port}/cam/{camera_name}"
    return f"http://{ip}:{port}/video_feed/{index}"


def _device_status(device_info: dict[str, Any]) -> str:
    last_register = device_info.get("last_register", 0.0)
    if time.time() - last_register > STATUS_TIMEOUT:
        return "offline"
    return "online"


def _build_device_view(device_info: dict[str, Any]) -> dict[str, Any]:
    camera_names = device_info.get("cameras", []) or []
    urls = [
        {
            "name": name,
            "url": _make_camera_url(device_info, name, idx),
        }
        for idx, name in enumerate(camera_names)
    ]
    return {
        "device": device_info.get("device"),
        "ip": device_info.get("ip"),
        "status": _device_status(device_info),
        "last_register": device_info.get("last_register"),
        "last_data": device_info.get("last_data"),
        "cameras": urls,
        "state": device_info.get("state", {}),
    }


@app.route("/api/register", methods=["POST"])
def api_register() -> Response:
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "invalid json"}), 400

    device = payload.get("device")
    if not device:
        return jsonify({"error": "missing device"}), 400

    with _device_lock:
        _devices[device] = {
            "device": device,
            "ip": payload.get("ip"),
            "cam_port": payload.get("cam_port"),
            "cameras": payload.get("cameras", []),
            "last_register": time.time(),
            "last_data": _devices.get(device, {}).get("last_data"),
            "state": _devices.get(device, {}).get("state", {}),
        }

    return jsonify({"status": "ok"})


@app.route("/api/data", methods=["POST"])
def api_data() -> Response:
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "invalid json"}), 400

    device = payload.get("device")
    if not device:
        return jsonify({"error": "missing device"}), 400

    sensors = payload.get("sensors", {})
    with _device_lock:
        device_info = _devices.setdefault(device, {
            "device": device,
            "ip": payload.get("ip"),
            "cam_port": payload.get("cam_port"),
            "cameras": payload.get("cameras", []),
            "last_register": 0.0,
            "last_data": None,
            "state": {},
        })
        device_info["state"] = sensors
        device_info["last_data"] = time.time()
        if payload.get("ip"):
            device_info["ip"] = payload.get("ip")
        if payload.get("cam_port"):
            device_info["cam_port"] = payload.get("cam_port")
        if payload.get("cameras") is not None:
            device_info["cameras"] = payload.get("cameras")

    return jsonify({"status": "ok"})


@app.route("/api/devices")
def api_devices() -> Response:
    with _device_lock:
        response = [_build_device_view(device_info) for device_info in _devices.values()]
    return jsonify({"devices": response, "now": time.time()})


@app.route("/")
def index() -> str:
    page = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Waste-E Hardware Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 16px; background: #f4f5f7; }
            header { margin-bottom: 20px; }
            .device { background: #fff; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.08); margin-bottom: 18px; padding: 18px; }
            .device h2 { margin: 0 0 10px; }
            .status { font-weight: bold; }
            .status.online { color: #107c10; }
            .status.offline { color: #a80000; }
            .camera-row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; }
            .camera-card { width: calc(33% - 12px); min-width: 240px; background: #222; color: #fff; border-radius: 8px; overflow: hidden; }
            .camera-card img { width: 100%; display: block; }
            .camera-card .label { padding: 8px; font-size: 0.95rem; }
            pre { white-space: pre-wrap; word-break: break-word; background: #111; color: #eee; padding: 12px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <header>
            <h1>Waste-E Central Dashboard</h1>
            <p>Devices register here and send live state to the dashboard server.</p>
        </header>
        <div id="devices"></div>
        <script>
            const api = '/api/devices';
            async function loadDevices() {
                const resp = await fetch(api);
                const data = await resp.json();
                const root = document.getElementById('devices');
                root.innerHTML = '';

                if (!data.devices || data.devices.length === 0) {
                    root.innerHTML = '<p>No devices have registered yet.</p>';
                    return;
                }

                for (const device of data.devices) {
                    const statusClass = device.status === 'online' ? 'online' : 'offline';
                    const deviceDiv = document.createElement('div');
                    deviceDiv.className = 'device';
                    deviceDiv.innerHTML = `
                        <h2>${device.device}</h2>
                        <div><strong>IP:</strong> ${device.ip || 'unknown'}</div>
                        <div><strong>Camera port:</strong> ${device.cam_port || 'unknown'}</div>
                        <div class="status ${statusClass}">Status: ${device.status}</div>
                        <div><strong>Last register:</strong> ${device.last_register ? new Date(device.last_register * 1000).toLocaleTimeString() : 'never'}</div>
                        <div><strong>Last data:</strong> ${device.last_data ? new Date(device.last_data * 1000).toLocaleTimeString() : 'never'}</div>
                    `;

                    if (device.cameras && device.cameras.length) {
                        const cameraRow = document.createElement('div');
                        cameraRow.className = 'camera-row';
                        for (const cam of device.cameras) {
                            const card = document.createElement('div');
                            card.className = 'camera-card';
                            card.innerHTML = `
                                <div class="label">${cam.name}</div>
                                <img src="${cam.url}" alt="${cam.name}" onerror="this.src='data:image/svg+xml;charset=UTF-8,<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'320\' height=\'240\'><rect width=\'100%\' height=\'100%\' fill=\'#333\'/><text x=\'50%\' y=\'50%\' fill=\'#fff\' font-size=\'20\' text-anchor=\'middle\' dominant-baseline=\'middle\'>Stream unavailable</text></svg>'" />
                            `;
                            cameraRow.appendChild(card);
                        }
                        deviceDiv.appendChild(cameraRow);
                    }

                    const stateBlock = document.createElement('pre');
                    stateBlock.textContent = JSON.stringify(device.state, null, 2);
                    deviceDiv.appendChild(stateBlock);
                    root.appendChild(deviceDiv);
                }
            }

            loadDevices();
            setInterval(loadDevices, 5000);
        </script>
    </body>
    </html>
    """
    return render_template_string(page)


if __name__ == "__main__":
    print(f"Starting Waste-E central dashboard on http://0.0.0.0:{DEFAULT_PORT}")
    app.run(host="0.0.0.0", port=DEFAULT_PORT, threaded=True, debug=False)
