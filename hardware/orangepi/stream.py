from __future__ import annotations

import glob
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Generator, Optional

MEDIAMTX_BIN         = os.getenv("MEDIAMTX_BIN", str(Path(__file__).parent / "mediamtx"))
MEDIAMTX_RTSP_PORT   = int(os.getenv("MEDIAMTX_RTSP_PORT", "8554"))
MEDIAMTX_WEBRTC_PORT = int(os.getenv("MEDIAMTX_WEBRTC_PORT", "8889"))
H264_ENCODER         = os.getenv("H264_ENCODER", "libx264")

# Cameras mounted upside-down — flip their stream
ROTATE_180: set[str] = set()

_VIDEO_RE = re.compile(r"^video\d+$")


class StreamManager:
    def __init__(
        self,
        cam_fps: int   = 10,
        cam_w: int     = 640,
        cam_h: int     = 480,
        cam_q: int     = 2,
    ) -> None:
        self.cam_fps      = cam_fps
        self.cam_w        = cam_w
        self.cam_h        = cam_h
        self.cam_q        = cam_q
        self.webrtc_mode  = os.path.isfile(MEDIAMTX_BIN) and os.access(MEDIAMTX_BIN, os.X_OK)
        self.rtsp_port    = MEDIAMTX_RTSP_PORT
        self.webrtc_port  = MEDIAMTX_WEBRTC_PORT
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock        = threading.Lock()

    def discover(self) -> list[str]:
        return [
            str(p)
            for p in sorted(
                (Path(x) for x in glob.glob("/dev/video*")),
                key=lambda p: _video_sort_key(str(p)),
            )
            if _VIDEO_RE.match(p.name) and _is_primary(p)
        ]

    def start_mediamtx(self) -> Optional[subprocess.Popen]:
        if not self.webrtc_mode:
            return None
        cfg = Path(MEDIAMTX_BIN).parent / "mediamtx.yml"
        cmd = [MEDIAMTX_BIN] + ([str(cfg)] if cfg.exists() else [])
        print(f"[MediaMTX] starting ({MEDIAMTX_BIN})")
        return subprocess.Popen(
            cmd, cwd=str(Path(MEDIAMTX_BIN).parent),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def get_or_start_proc(self, dev: str) -> subprocess.Popen:
        with self._lock:
            p = self._procs.get(dev)
            if p is None or p.poll() is not None:
                self._procs[dev] = self._start_ffmpeg(dev)
            return self._procs[dev]

    def mjpeg_gen(self, dev: str) -> Generator[bytes, None, None]:
        p   = self.get_or_start_proc(dev)
        buf = b""
        while True:
            chunk = p.stdout.read(65536)
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
                frame = buf[s: e + 2]
                buf   = buf[e + 2:]
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    + frame
                    + b"\r\n"
                )

    def start_recording_ffmpeg(self, dev: str, dest: Path) -> subprocess.Popen:
        cam_name = os.path.basename(dev)
        if self.webrtc_mode:
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-rtsp_transport", "tcp",
                "-i", f"rtsp://127.0.0.1:{self.rtsp_port}/{cam_name}",
                "-map", "0:v:0", "-c:v", "copy", "-an", "-f", "matroska", str(dest),
            ]
        else:
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "v4l2", "-input_format", "mjpeg",
                "-framerate", str(self.cam_fps),
                "-video_size", f"{self.cam_w}x{self.cam_h}",
                "-i", dev,
            ]
            if dev in ROTATE_180:
                cmd += ["-vf", "hflip,vflip"]
                if H264_ENCODER == "libx264":
                    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency"]
                else:
                    cmd += ["-c:v", H264_ENCODER]
            else:
                cmd += ["-c:v", "copy"]
            cmd += ["-an", "-f", "matroska", str(dest)]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0)

    # ── private ───────────────────────────────────────────────────────────────

    def _start_ffmpeg(self, dev: str) -> subprocess.Popen:
        cam_name = os.path.basename(dev)
        if self.webrtc_mode:
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "v4l2", "-input_format", "mjpeg",
                "-framerate", str(self.cam_fps),
                "-video_size", f"{self.cam_w}x{self.cam_h}",
                "-i", dev,
            ]
            if dev in ROTATE_180:
                cmd += ["-vf", "hflip,vflip"]
            if H264_ENCODER == "libx264":
                cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency"]
            else:
                cmd += ["-c:v", H264_ENCODER]
            cmd += [
                "-b:v", "800k", "-g", str(self.cam_fps * 2),
                "-f", "rtsp", "-rtsp_transport", "tcp",
                f"rtsp://127.0.0.1:{self.rtsp_port}/{cam_name}",
            ]
            return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0)

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "v4l2", "-input_format", "mjpeg",
            "-framerate", str(self.cam_fps),
            "-video_size", f"{self.cam_w}x{self.cam_h}",
            "-i", dev,
        ]
        if dev in ROTATE_180:
            cmd += ["-vf", "hflip,vflip"]
        cmd += ["-f", "mjpeg", "-q:v", str(self.cam_q), "pipe:1"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)


def _video_sort_key(p: str) -> tuple[int, str]:
    m = re.search(r"(\d+)$", p)
    return (int(m.group(1)), p) if m else (10**9, p)


def _is_primary(dev: Path) -> bool:
    f = Path("/sys/class/video4linux") / dev.name / "index"
    try:
        return f.read_text().strip() == "0"
    except OSError:
        return False
