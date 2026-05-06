from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

try:
    from motor_driver import MotorDriver
    _HAS_MOTOR = True
except Exception:
    _HAS_MOTOR = False

DRIVE_ACTIONS = {
    "forward", "backward", "left", "right", "stop",
    "forward_left", "forward_right", "backward_left", "backward_right",
}


class MotorManager:
    def __init__(self, dashboard_drive_url: Optional[str] = None) -> None:
        self._dashboard_url = dashboard_drive_url
        self._lock          = threading.Lock()
        self._driver        = None
        self._last_error: Optional[str] = None

    @property
    def is_online(self) -> bool:
        with self._lock:
            return self._driver is not None

    def start(self) -> None:
        """Start the motor reconnect loop. No-op in dashboard passthrough mode."""
        if self._dashboard_url:
            return
        if not _HAS_MOTOR:
            print("[motor] MotorDriver import failed — /api/drive will return 503")
            return
        threading.Thread(target=self._manager_loop, daemon=True).start()

    def apply_action(self, action: str) -> tuple[str, int]:
        """Send a drive action. Returns (json_body, http_status)."""
        if action not in DRIVE_ACTIONS:
            return json.dumps({"error": f"invalid action: {action}"}), 400

        if self._dashboard_url:
            return self._proxy_post(self._dashboard_url, {"action": action})

        with self._lock:
            driver = self._driver
        if driver is None:
            err = self._last_error or "not initialized"
            return json.dumps({"error": f"motor driver unavailable: {err}"}), 503
        try:
            driver.apply_action(action)
            return json.dumps({"ok": True, "action": action}), 200
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            return json.dumps({"error": str(exc)}), 500

    def set_speed(self, speed: float) -> tuple[str, int]:
        """Set motor speed (0.0–1.0). Returns (json_body, http_status)."""
        if self._dashboard_url:
            speed_url = self._dashboard_url.replace("/api/drive", "/api/speed")
            return self._proxy_post(speed_url, {"speed": speed})

        with self._lock:
            driver = self._driver
        if driver is None:
            return json.dumps({"error": "motor driver unavailable"}), 503
        driver.set_speed(speed)
        return json.dumps({"ok": True, "speed": speed}), 200

    # ── internals ─────────────────────────────────────────────────────────────

    def _manager_loop(self) -> None:
        while True:
            with self._lock:
                already_open = self._driver is not None
            if already_open:
                time.sleep(1.0)
                continue
            try:
                m = MotorDriver()
                m.open()
                with self._lock:
                    self._driver     = m
                    self._last_error = None
                print("[motor] local motor driver ready")
            except Exception as e:
                with self._lock:
                    self._driver     = None
                    self._last_error = str(e)
                print(f"[motor] unavailable: {e} — retrying in 3s")
                time.sleep(3.0)

    @staticmethod
    def _proxy_post(url: str, payload: dict) -> tuple[str, int]:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return raw or json.dumps({"ok": True}), resp.getcode()
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            return raw or json.dumps({"error": f"dashboard http {exc.code}"}), exc.code
        except Exception as exc:
            return json.dumps({"error": f"proxy failed: {exc}"}), 502
