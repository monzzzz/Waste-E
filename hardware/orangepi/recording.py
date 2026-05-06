from __future__ import annotations

import http.client
import os
import signal
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from stream import StreamManager

LOCAL_RECORDINGS_DIR = Path(
    os.getenv("LOCAL_RECORDINGS_DIR", str(Path(__file__).parent / "recordings"))
)
_DEVICE_NAME = "orangepi"


class RecordingManager:
    def __init__(
        self,
        streams: StreamManager,
        get_central_url: Callable[[], str],
    ) -> None:
        self._streams        = streams
        self._get_central_url = get_central_url
        self._lock           = threading.Lock()
        self._session: dict | None = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._session is not None

    def start(self, session_id: str, upload_url: str, cam_devs: list[str]) -> dict:
        with self._lock:
            if self._session is not None:
                return {"ok": False, "error": "already recording", "session": self._session["id"]}

            session_dir = LOCAL_RECORDINGS_DIR / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            active_devs = [d for d in cam_devs if os.path.exists(d)]
            procs:  dict[str, subprocess.Popen] = {}
            files:  dict[str, str]              = {}
            errors: dict[str, str]              = {}

            for dev in active_devs:
                name = os.path.basename(dev)
                if self._streams.webrtc_mode:
                    self._streams.get_or_start_proc(dev)
                dest = session_dir / f"{_safe_name(name)}.mkv"
                try:
                    procs[name] = self._streams.start_recording_ffmpeg(dev, dest)
                    files[name] = str(dest)
                except Exception as exc:
                    errors[name] = str(exc)

            self._session = {
                "id":         session_id,
                "dir":        str(session_dir),
                "upload_url": upload_url,
                "started_at": time.time(),
                "procs":      procs,
                "files":      files,
            }

        print(f"[OrangePi recording] started {session_id}: {list(files)}")
        return {
            "ok":      True,
            "session": session_id,
            "dir":     str(session_dir),
            "cameras": list(files),
            "errors":  errors,
        }

    def stop(self, expected_session: str = "") -> dict:
        with self._lock:
            if self._session is None:
                return {"ok": False, "error": "not recording"}
            sess = self._session
            if expected_session and expected_session != sess["id"]:
                return {"ok": False, "error": "session mismatch", "session": sess["id"]}
            self._session = None

        for proc in sess["procs"].values():
            _stop_proc(proc)

        uploads: list[dict] = []
        for name, raw_path in sess["files"].items():
            fpath = Path(raw_path)
            if not fpath.exists() or fpath.stat().st_size <= 0:
                uploads.append({
                    "camera": name, "ok": False,
                    "error": "empty or missing recording", "file": str(fpath),
                })
                continue
            try:
                result           = _upload(sess["upload_url"], self._get_central_url(), sess["id"], name, fpath)
                result["camera"] = name
                uploads.append(result)
            except Exception as exc:
                uploads.append({"camera": name, "ok": False, "error": str(exc), "file": str(fpath)})

        print(f"[OrangePi recording] stopped {sess['id']}: {uploads}")
        return {"ok": True, "session": sess["id"], "dir": sess["dir"], "uploads": uploads}


def _safe_name(raw: str) -> str:
    cleaned = "".join(c for c in raw if c.isalnum() or c in "-_")
    return cleaned or "unnamed"


def _stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    for sig, timeout in [(signal.SIGINT, 8), (signal.SIGTERM, 3)]:
        try:
            proc.send_signal(sig)
            proc.wait(timeout=timeout)
            return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def _upload(upload_url: str, central_url: str, session_id: str, camera_name: str, fpath: Path) -> dict:
    if upload_url:
        parsed_upload = urllib.parse.urlsplit(upload_url)
        if parsed_upload.hostname in {"localhost", "127.0.0.1", "::1"}:
            upload_url = ""
    if not upload_url:
        upload_url = urllib.parse.urljoin(central_url.rstrip("/") + "/", "api/recording/video")

    parsed = urllib.parse.urlsplit(upload_url)
    query  = urllib.parse.urlencode({
        "device": _DEVICE_NAME, "cam": camera_name,
        "session": session_id,  "filename": fpath.name,
    })
    path = (parsed.path or "/") + f"?{parsed.query + '&' if parsed.query else ''}{query}"

    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn     = conn_cls(parsed.netloc, timeout=120)
    try:
        with open(fpath, "rb") as fh:
            conn.request("POST", path, body=fh, headers={
                "Content-Type":   "video/x-matroska",
                "Content-Length": str(fpath.stat().st_size),
            })
            resp = conn.getresponse()
            raw  = resp.read().decode("utf-8", errors="replace")
            if resp.status >= 400:
                raise RuntimeError(f"dashboard upload http {resp.status}: {raw[:300]}")
            return {"ok": True, "status": resp.status, "file": str(fpath), "response": raw[:300]}
    finally:
        conn.close()
