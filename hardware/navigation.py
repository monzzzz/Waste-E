"""
Waypoint navigation controller — adapted from
self-driving-pipeline/alpamayo_pipeline/robot/autonomy.py

This is pure hardware/control logic with no AI inference.
The navigator follows a list of geographic (lat/lon) waypoints by computing
the heading error to the next target and issuing forward/turn motor pulses.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from ego_state import EgoState
from geo_utils import haversine_distance_m, latlon_to_local_m


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MotorCalibration:
    """Physical calibration constants for the robot's drivetrain."""
    max_speed_mps: float = 0.5
    left_turn_deg_per_ms: float = 0.1
    right_turn_deg_per_ms: float = 0.1


@dataclass
class NavConfig:
    """Tuning parameters for the waypoint navigator."""
    heading_deadband_deg: float = 10.0
    waypoint_reached_distance_m: float = 1.5
    lookahead_distance_m: float = 2.0
    min_forward_pulse_ms: float = 100.0
    max_forward_pulse_ms: float = 1000.0
    min_turn_pulse_ms: float = 50.0
    max_turn_pulse_ms: float = 500.0
    control_hz: float = 10.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _project_geographic_waypoint(
    lat_deg: float,
    lon_deg: float,
    ego: EgoState,
) -> tuple[float, float] | None:
    """
    Returns (forward_m, lateral_m) of the waypoint in the robot's local frame.
    forward_m > 0 means the waypoint is ahead; lateral_m > 0 is to the left.
    Returns None if ego has no GPS fix.
    """
    if ego.lat_deg is None or ego.lon_deg is None:
        return None
    dx_m, dy_m = latlon_to_local_m(ego.lat_deg, ego.lon_deg, lat_deg, lon_deg)
    cos_yaw = math.cos(ego.yaw_rad)
    sin_yaw = math.sin(ego.yaw_rad)
    forward_m = cos_yaw * dx_m + sin_yaw * dy_m
    lateral_m = -sin_yaw * dx_m + cos_yaw * dy_m
    return forward_m, lateral_m


# ---------------------------------------------------------------------------
# Control decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NavDecision:
    action: str
    duration_s: float
    summary: str


def _compute_decision(
    ego: EgoState,
    waypoint_lat: float,
    waypoint_lon: float,
    config: NavConfig,
    calibration: MotorCalibration,
) -> NavDecision:
    """Compute a single motor action to move toward the given waypoint."""
    projected = _project_geographic_waypoint(waypoint_lat, waypoint_lon, ego)
    if projected is None:
        return NavDecision("stop", 0.0, "No GPS fix — waiting.")

    forward_m, lateral_m = projected
    distance_m = math.hypot(forward_m, lateral_m)
    heading_error_deg = math.degrees(math.atan2(lateral_m, forward_m))

    if distance_m <= config.waypoint_reached_distance_m:
        return NavDecision("stop", 0.0, f"Waypoint reached ({distance_m:.1f} m).")

    if forward_m <= 0.0:
        return NavDecision(
            "stop", 0.0,
            f"Waypoint ({waypoint_lat:.6f}, {waypoint_lon:.6f}) is behind the robot."
        )

    if abs(heading_error_deg) > config.heading_deadband_deg:
        if heading_error_deg > 0.0:
            deg_per_ms = calibration.left_turn_deg_per_ms
            action = "left"
        else:
            deg_per_ms = calibration.right_turn_deg_per_ms
            action = "right"

        if deg_per_ms <= 0.0:
            return NavDecision("stop", 0.0, f"Missing {action}-turn calibration.")

        pulse_ms = _clamp(
            abs(heading_error_deg) / deg_per_ms,
            config.min_turn_pulse_ms,
            config.max_turn_pulse_ms,
        )
        return NavDecision(
            action, pulse_ms / 1000.0,
            f"Turn {action} {abs(heading_error_deg):.1f}° toward "
            f"({waypoint_lat:.6f}, {waypoint_lon:.6f})."
        )

    if calibration.max_speed_mps <= 0.0:
        return NavDecision("stop", 0.0, "Missing forward-speed calibration.")

    pulse_ms = _clamp(
        (forward_m / calibration.max_speed_mps) * 1000.0,
        config.min_forward_pulse_ms,
        config.max_forward_pulse_ms,
    )
    return NavDecision(
        "forward", pulse_ms / 1000.0,
        f"Forward {forward_m:.2f} m to ({waypoint_lat:.6f}, {waypoint_lon:.6f})."
    )


# ---------------------------------------------------------------------------
# WaypointNavigator
# ---------------------------------------------------------------------------

class WaypointNavigator:
    """
    Autonomous waypoint follower that drives toward a sequence of GPS
    coordinates using a MotorDriver.

    Usage::

        navigator = WaypointNavigator(NavConfig(), MotorCalibration())
        navigator.set_waypoints([(lat1, lon1), (lat2, lon2)])
        navigator.start(motor_driver, get_ego_state=gps_provider.read)
        ...
        navigator.stop()
    """

    def __init__(self, config: NavConfig, calibration: MotorCalibration) -> None:
        self._config = config
        self._calibration = calibration
        self._waypoints: list[tuple[float, float]] = []
        self._waypoint_index = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._recent_events: deque[dict] = deque(maxlen=32)

    def set_waypoints(self, waypoints: list[tuple[float, float]]) -> None:
        """Set the list of (lat_deg, lon_deg) targets to follow."""
        with self._lock:
            self._waypoints = list(waypoints)
            self._waypoint_index = 0
        self._emit(f"Loaded {len(waypoints)} waypoint(s).")

    def start(
        self,
        motor_driver,
        get_ego_state: Callable[[], EgoState | None],
        on_waypoint_reached: Callable[[int, float, float], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        """Start the navigation loop in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(motor_driver, get_ego_state, on_waypoint_reached, on_finished),
            name="waypoint-navigator",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the navigation loop and wait for the thread to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    @property
    def current_waypoint_index(self) -> int:
        with self._lock:
            return self._waypoint_index

    @property
    def recent_events(self) -> list[dict]:
        return list(self._recent_events)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(
        self,
        motor_driver,
        get_ego_state: Callable[[], EgoState | None],
        on_waypoint_reached,
        on_finished,
    ) -> None:
        loop_period_s = max(1.0 / self._config.control_hz, 0.05)
        last_summary = ""
        self._emit("Navigation started.")

        try:
            while not self._stop_event.is_set():
                with self._lock:
                    if self._waypoint_index >= len(self._waypoints):
                        motor_driver.stop()
                        self._emit("All waypoints reached.")
                        if on_finished:
                            on_finished()
                        break

                    lat, lon = self._waypoints[self._waypoint_index]

                ego = get_ego_state()
                if ego is None:
                    motor_driver.stop()
                    self._sleep(loop_period_s)
                    continue

                decision = _compute_decision(ego, lat, lon, self._config, self._calibration)

                if decision.summary != last_summary:
                    self._emit(decision.summary)
                    last_summary = decision.summary

                if decision.action == "stop" or decision.duration_s <= 0.0:
                    # Check if we actually reached the waypoint
                    if "reached" in decision.summary:
                        with self._lock:
                            idx = self._waypoint_index
                            self._waypoint_index += 1
                        if on_waypoint_reached:
                            on_waypoint_reached(idx, lat, lon)
                    else:
                        motor_driver.stop()
                    self._sleep(loop_period_s)
                    continue

                motor_driver.apply_action(decision.action)
                if self._sleep(decision.duration_s):
                    motor_driver.stop()
                    continue
                motor_driver.stop()

        except Exception as exc:
            self._emit(f"Navigation error: {exc}")
            try:
                motor_driver.stop()
            except Exception:
                pass

    def _sleep(self, duration_s: float) -> bool:
        """Sleep interruptibly; returns True if stopped early."""
        deadline = time.monotonic() + max(duration_s, 0.0)
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if self._stop_event.wait(timeout=min(0.02, remaining)):
                return True
        return self._stop_event.is_set()

    def _emit(self, message: str) -> None:
        self._recent_events.append({"timestamp_s": time.time(), "message": message})
        print(f"[navigator] {message}")
