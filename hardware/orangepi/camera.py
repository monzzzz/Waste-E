from __future__ import annotations

import glob
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, Response, abort, render_template_string

# V4L2 device pattern — primary capture nodes only
_VIDEO_RE = re.compile(r"^video\d+$")

# Cameras mounted upside-down on the robot chassis
ROTATE_180_DEVS: set[str] = {"/dev/video0", "/dev/video2", "/dev/video4", "/dev/video6"}

# Default stream parameters
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 10
DEFAULT_QUALITY = 2  # MJPEG quality: 2 (best) – 31 (worst)


def _video_sort_key(path: str) -> tuple[int, str]:
    m = re.search(r"(\d+)$", path)
    return (int(m.group(1)), path) if m else (10**9, path)


def _is_primary_node(device_path: Path) -> bool:
    index_file = Path("/sys/class/video4linux") / device_path.name / "index"
    try:
        return index_file.read_text(encoding="utf-8").strip() == "0"
    except OSError:
        return False


def discover_cameras() -> list[str]:
    """Return sorted list of primary V4L2 capture device paths."""
    devices: list[str] = []
    for raw in sorted(glob.glob("/dev/video*"), key=_video_sort_key):
        p = Path(raw)
        if _VIDEO_RE.match(p.name) and _is_primary_node(p):
            devices.append(str(p))
    return devices


class CameraStreamer:
    """
    MJPEG streaming server for all USB cameras connected to the Orange Pi.

    Uses ffmpeg under the hood; exposes a Flask app on a background thread.

    Usage::

        cam = CameraStreamer(port=8080)
        cam.start()          # non-blocking, starts Flask in a daemon thread
        ...
        cam.stop()

    Physical connections: USB cameras plug into any USB-A port.
    No GPIO wiring required.
    """

    def __init__(
        self,
        port: int = 8080,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        quality: int = DEFAULT_QUALITY,
    ) -> None:
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality

        self._devs: list[str] = discover_cameras()
        self._procs: dict[str, subprocess.Popen] = {}
        self._app = self._build_app()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Flask streaming server in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._app.run,
            kwargs={"host": "0.0.0.0", "port": self.port, "threaded": True},
            daemon=True,
        )
        self._thread.start()
        print(f"Camera stream started on http://0.0.0.0:{self.port}")
        print(f"Cameras detected: {self._devs or 'none'}")

    def stop(self) -> None:
        """Terminate all ffmpeg child processes."""
        for proc in self._procs.values():
            if proc.poll() is None:
                proc.terminate()
        self._procs.clear()
        print("Camera streamer stopped")

    def get_camera_list(self) -> list[str]:
        return list(self._devs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_ffmpeg(self, dev: str) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-f", "v4l2",
            "-input_format", "mjpeg",
            "-framerate", str(self.fps),
            "-video_size", f"{self.width}x{self.height}",
            "-i", dev,
        ]
        if dev in ROTATE_180_DEVS:
            cmd.extend(["-vf", "hflip,vflip"])
        cmd.extend(["-f", "mjpeg", "-q:v", str(self.quality), "pipe:1"])
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def _get_proc(self, dev: str) -> subprocess.Popen:
        p = self._procs.get(dev)
        if p is None or p.poll() is not None:
            self._procs[dev] = self._start_ffmpeg(dev)
        return self._procs[dev]

    def _build_app(self) -> Flask:
        app = Flask(__name__)
        devs = self._devs  # closure

        @app.route("/")
        def index():
            cams = [{"dev": d, "name": os.path.basename(d)} for d in devs if os.path.exists(d)]
            if not cams:
                return "No cameras found.", 404
            html = """
            <!doctype html><html>
            <head>
              <meta charset="utf-8"/>
              <meta name="viewport" content="width=device-width, initial-scale=1"/>
              <title>Waste-E Cameras</title>
              <style>
                body{margin:0;font-family:system-ui,sans-serif;background:#111;color:#eee}
                header{padding:12px 16px;position:sticky;top:0;background:#111;border-bottom:1px solid #333}
                .grid{display:grid;gap:10px;padding:10px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
                .card{background:#1a1a1a;border:1px solid #333;border-radius:10px;overflow:hidden}
                .label{padding:8px 10px;font-size:14px;border-bottom:1px solid #333}
                img{width:100%;height:auto;display:block;background:#000}
              </style>
            </head>
            <body>
              <header><b>Waste-E Camera Feeds</b></header>
              <div class="grid">
                {% for cam in cams %}
                  <div class="card">
                    <div class="label">{{ cam.name }} ({{ cam.dev }})</div>
                    <img src="/cam/{{ cam.name }}" />
                  </div>
                {% endfor %}
              </div>
            </body></html>
            """
            return render_template_string(html, cams=cams)

        @app.route("/cam/<name>")
        def cam_stream(name):
            dev = f"/dev/{name}"
            if dev not in devs or not os.path.exists(dev):
                abort(404)
            p = self._get_proc(dev)

            def generate():
                buf = b""
                while True:
                    chunk = p.stdout.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        soi = buf.find(b"\xff\xd8")
                        if soi == -1:
                            break
                        eoi = buf.find(b"\xff\xd9", soi + 2)
                        if eoi == -1:
                            break
                        frame = buf[soi:eoi + 2]
                        buf = buf[eoi + 2:]
                        yield b"--frame\r\n"
                        yield b"Content-Type: image/jpeg\r\n"
                        yield f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        yield frame
                        yield b"\r\n"

            return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/health")
        def health():
            return {"cams": [d for d in devs if os.path.exists(d)], "running": True}

        return app


def main():
    """Run the camera streamer standalone."""
    streamer = CameraStreamer(port=8080)
    streamer.start()

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        streamer.stop()


if __name__ == "__main__":
    main()
