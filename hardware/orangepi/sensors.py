from __future__ import annotations

import copy
import http.client
import json
import math
import os
import threading
import time
import urllib.parse
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from motor_manager import MotorManager

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

GPS_PORT = os.getenv("GPS_PORT", "/dev/ttyS0")
IMU_PORT = os.getenv("IMU_PORT", "/dev/ttyS6")

ENCODER_GEAR_RATIO  = float(os.getenv("ENCODER_GEAR_RATIO", "51.0"))
ENC_HZ              = 100
ENC_MEDIAN_N        = 5
ENC_WINDOW_SECS     = 0.5
ENC_LEFT_SIGN       = int(os.getenv("ENC_LEFT_SIGN",  "-1"))
ENC_RIGHT_SIGN      = int(os.getenv("ENC_RIGHT_SIGN", "1"))
ENC_MAX_MOTOR_RPM   = float(os.getenv("ENC_MAX_MOTOR_RPM", "8000"))
ENC_LOG             = os.getenv("ENC_LOG", "0") == "1"
ENC_LOG_DIR         = os.getenv("ENC_LOG_DIR", str(Path(__file__).parent))
WHEEL_DIAMETER_M    = float(os.getenv("WHEEL_DIAMETER_M", "0.11"))

_INITIAL_STATE: dict = {
    "gps": {
        "lat": None, "lon": None, "alt": None,
        "speed": None, "heading": None, "satellites": None, "fix": False,
    },
    "imu": {
        "heading": 0.0, "roll": 0.0, "pitch": 0.0,
        "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
        "temp": None, "cal_sys": 0, "cal_gyro": 0, "cal_accel": 0, "cal_mag": 0,
    },
    "encoders": {
        "left_rpm": None, "right_rpm": None,
        "left_angle": None, "right_angle": None,
        "left_dist_m": 0.0, "right_dist_m": 0.0,
    },
    "motor_online": False,
    "motor_source": "none",
    "ts": 0.0,
}


class SensorManager:
    def __init__(
        self,
        send_hz: float,
        dashboard_url: Optional[str] = None,
        motor_manager: Optional[MotorManager] = None,
    ) -> None:
        self._send_hz       = send_hz
        self._dashboard_url = dashboard_url
        self._motors        = motor_manager
        self._lock          = threading.Lock()
        self._state         = copy.deepcopy(_INITIAL_STATE)
        self._enc_lock      = threading.Lock()
        self._enc_out: dict = {
            "left_rpm": None, "right_rpm": None,
            "left_angle": None, "right_angle": None,
            "left_dist_m": 0.0, "right_dist_m": 0.0,
        }

    def start(self) -> None:
        if not self._dashboard_url:
            self._start_encoders()
        threading.Thread(target=self._sensor_loop, daemon=True).start()

    def get_state(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._state)

    def update_motor_speed(self, speed: float) -> None:
        with self._lock:
            self._state.setdefault("motor", {})["speed"] = speed

    # ── encoders ──────────────────────────────────────────────────────────────

    def _start_encoders(self) -> None:
        for bus, side, sign, a_key, r_key in [
            (4, "left",  ENC_LEFT_SIGN,  "left_angle",  "left_rpm"),
            (5, "right", ENC_RIGHT_SIGN, "right_angle", "right_rpm"),
        ]:
            threading.Thread(
                target=self._encoder_thread,
                args=(bus, 0x36, side, sign, a_key, r_key),
                daemon=True,
            ).start()

    def _encoder_thread(
        self,
        bus_id: int,
        address: int,
        side: str,
        sign: int,
        angle_key: str,
        rpm_key: str,
    ) -> None:
        enc = None
        if _HAS_ENC:
            try:
                enc = AS5600Encoder(bus_id=bus_id, address=address)
                enc.open()
                print(f"[ENC] {side} encoder opened (bus {bus_id}, addr {address:#04x})")
            except Exception as e:
                print(f"[ENC] {side} encoder: {e}")

        cum        = 0.0
        prev_angle = None
        win: deque = deque()
        interval   = 1.0 / ENC_HZ

        log_file = None
        if ENC_LOG:
            log_path = Path(ENC_LOG_DIR) / f"enc_{side}.csv"
            log_file = open(log_path, "w", buffering=1)
            log_file.write(
                "mono_s,angle_deg,delta_deg,cumulative_deg,"
                "dt_ms,implied_motor_rpm,accepted,window_rpm\n"
            )
            print(f"[ENC] {side} logging to {log_path}")

        while True:
            t0 = time.monotonic()
            if enc:
                try:
                    reads  = sorted(enc.read_angle_deg() for _ in range(ENC_MEDIAN_N))
                    angle  = reads[ENC_MEDIAN_N // 2]
                    result = {angle_key: round(angle, 2)}

                    delta = dt_sample = implied = 0.0
                    accepted   = False
                    window_rpm = None

                    if prev_angle is not None:
                        delta = angle - prev_angle
                        if delta > 180:    delta -= 360
                        elif delta < -180: delta += 360
                        dt_sample = t0 - (win[-1][0] if win else t0)
                        implied   = abs(delta / 360.0 / dt_sample * 60.0) if dt_sample > 0 else 0.0
                        if implied <= ENC_MAX_MOTOR_RPM:
                            cum += delta
                            win.append((t0, cum))
                            accepted = True
                    else:
                        win.append((t0, cum))
                        accepted = True
                    prev_angle = angle

                    while win and t0 - win[0][0] > ENC_WINDOW_SECS:
                        win.popleft()
                    if len(win) >= 2:
                        dt_w = win[-1][0] - win[0][0]
                        da_w = win[-1][1] - win[0][1]
                        if dt_w > 0:
                            window_rpm = round(
                                sign * (da_w / 360.0) / dt_w * 60.0 / ENCODER_GEAR_RATIO, 2
                            )
                            result[rpm_key] = window_rpm

                    result[f"{side}_dist_m"] = round(
                        sign * cum / 360.0 / ENCODER_GEAR_RATIO * math.pi * WHEEL_DIAMETER_M, 4
                    )
                    with self._enc_lock:
                        self._enc_out.update(result)

                    if log_file:
                        log_file.write(
                            f"{t0:.6f},{angle:.3f},{delta:.3f},{cum:.3f},"
                            f"{dt_sample*1000:.3f},{implied:.1f},"
                            f"{'Y' if accepted else 'N'},"
                            f"{f'{window_rpm:.2f}' if window_rpm is not None else ''}\n"
                        )
                except Exception:
                    if log_file:
                        log_file.write(f"{t0:.6f},READ_ERROR,,,,,N,\n")

            rem = interval - (time.monotonic() - t0)
            if rem > 0:
                time.sleep(rem)

    # ── sensor loop ───────────────────────────────────────────────────────────

    def _sensor_loop(self) -> None:
        reader = _PersistentReader(timeout=2.0)
        gps: Optional[object] = None
        imu: Optional[object] = None

        if not self._dashboard_url:
            gps = self._init_gps()
            imu = self._init_imu()

        while True:
            with self._lock:
                state = copy.deepcopy(self._state)

            if self._dashboard_url:
                self._read_dashboard(state, reader)
            else:
                imu = self._read_local(state, gps, imu)

            state["ts"] = time.time()
            with self._lock:
                self._state.update(state)

            time.sleep(1.0 / self._send_hz)

    def _read_dashboard(self, state: dict, reader: _PersistentReader) -> None:
        try:
            dash = reader.get_json(f"{self._dashboard_url.rstrip('/')}/api/state")
            g    = dash.get("gps", {})
            state["gps"].update({
                "lat": g.get("lat"),     "lon": g.get("lon"),
                "alt": g.get("alt"),     "speed": g.get("speed"),
                "heading": g.get("heading"), "satellites": g.get("satellites"),
                "fix": bool(g.get("fix")),
            })
            im = dash.get("imu", {})
            state["imu"].update({
                "heading": im.get("heading", 0.0), "roll":  im.get("roll", 0.0),
                "pitch":   im.get("pitch",   0.0), "qw":    im.get("qw",  1.0),
                "qx": im.get("qx", 0.0), "qy": im.get("qy", 0.0), "qz": im.get("qz", 0.0),
                "temp": im.get("temp"),
                "cal_sys":   im.get("cal_sys",   0), "cal_gyro":  im.get("cal_gyro",  0),
                "cal_accel": im.get("cal_accel", 0), "cal_mag":   im.get("cal_mag",   0),
            })
            enc = dash.get("encoders", {})
            if isinstance(enc, dict):
                state["encoders"].update({
                    "left_rpm":    enc.get("left_rpm"),    "right_rpm":   enc.get("right_rpm"),
                    "left_angle":  enc.get("left_angle"),  "right_angle": enc.get("right_angle"),
                    "left_dist_m": enc.get("left_dist_m", 0.0),
                    "right_dist_m": enc.get("right_dist_m", 0.0),
                })
            state["motor_online"] = bool(dash.get("motor_online"))
            state["motor_source"] = "dashboard"
        except Exception:
            state["motor_online"] = False
            state["motor_source"] = "dashboard"

    def _read_local(self, state: dict, gps, imu):
        if gps:
            try:
                d = gps.get_gps_data()
                if d:
                    state["gps"].update({
                        "lat": d.get("latitude"),  "lon": d.get("longitude"),
                        "alt": d.get("altitude"),  "speed": d.get("speed"),
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
                        "qw": quat.get("w") or 1.0, "qx": quat.get("x") or 0.0,
                        "qy": quat.get("y") or 0.0, "qz": quat.get("z") or 0.0,
                        "temp": d.get("temperature"),
                        "cal_sys":   cal.get("system", 0), "cal_gyro":  cal.get("gyro",  0),
                        "cal_accel": cal.get("accel",  0), "cal_mag":   cal.get("mag",   0),
                    })
            except Exception as e:
                print(f"[IMU] read error: {e}")
                if "Bad file descriptor" in str(e) or "NoneType" in str(e):
                    imu = self._reconnect_imu(imu)

        state["motor_online"] = self._motors.is_online if self._motors else False
        state["motor_source"] = "local"

        with self._enc_lock:
            state["encoders"].update(self._enc_out)

        return imu

    # ── init helpers ──────────────────────────────────────────────────────────

    def _init_gps(self):
        if not _HAS_GPS:
            return None
        try:
            gps = NEO8M_GPS(port=GPS_PORT)
            if gps.connect():
                print(f"[GPS] connected on {GPS_PORT}")
                return gps
        except Exception as e:
            print(f"[GPS] error: {e}")
        return None

    def _init_imu(self):
        if not _HAS_IMU:
            return None
        try:
            imu = BNO055_IMU(uart_port=IMU_PORT)
            if imu.connect():
                print(f"[IMU] connected on {IMU_PORT}")
                return imu
        except Exception as e:
            print(f"[IMU] error: {e}")
        return None

    def _reconnect_imu(self, imu):
        try:
            imu.disconnect()
            time.sleep(2)
            if imu.connect():
                print("[IMU] reconnected")
                return imu
        except Exception:
            pass
        return None


class _PersistentReader:
    def __init__(self, timeout: float = 2.5) -> None:
        self._timeout = timeout
        self._host: str = ""
        self._conn: http.client.HTTPConnection | None = None

    def get_json(self, url: str) -> dict:
        parsed = urllib.parse.urlsplit(url)
        host   = parsed.netloc
        path   = parsed.path or "/"
        if host != self._host:
            self._conn = None
            self._host = host
        for attempt in range(2):
            try:
                if self._conn is None:
                    self._conn = http.client.HTTPConnection(host, timeout=self._timeout)
                self._conn.request("GET", path, headers={"Connection": "keep-alive"})
                resp = self._conn.getresponse()
                raw  = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw) if raw else {}
                return data if isinstance(data, dict) else {}
            except Exception:
                self._conn = None
                if attempt == 0:
                    continue
                return {}
        return {}
