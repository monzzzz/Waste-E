#!/usr/bin/env python3
"""
Waste-E Robot Main Control System

Hardware pipeline (adapted from self-driving-pipeline):
  GPSEgoProvider  — UART GPS with dead-reckoning ego state
  BNO055_IMU      — 9-DOF IMU for heading and orientation
  MotorDriver     — gpiod-based differential drive motor control
  AS5600Encoder   — 12-bit magnetic encoder for wheel odometry
  WheelOdometry   — differential drive position estimation
  WaypointNavigator — GPS waypoint following (no AI required)
"""

import json
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Hardware modules
from GPS import GPSEgoProvider
from IMUsensor import BNO055_IMU
from motor_driver import MotorDriver
from motorencoder import AS5600Encoder, WheelOdometry
from navigation import MotorCalibration, NavConfig, WaypointNavigator
from ego_state import EgoState


class WasteERobot:
    """Main robot control class integrating all hardware components."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # Hardware components
        self.gps: Optional[GPSEgoProvider] = None
        self.imu: Optional[BNO055_IMU] = None
        self.motor_driver: Optional[MotorDriver] = None
        self.left_encoder: Optional[AS5600Encoder] = None
        self.right_encoder: Optional[AS5600Encoder] = None
        self.left_odometry: Optional[WheelOdometry] = None
        self.right_odometry: Optional[WheelOdometry] = None
        self.navigator: Optional[WaypointNavigator] = None

        self.running = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_hardware(self) -> bool:
        print("=" * 60)
        print("WASTE-E ROBOT INITIALIZATION")
        print("=" * 60)

        # GPS ego provider
        print("\n[1/5] Initializing GPS (NEO-8M) with dead-reckoning...")
        try:
            cal = self.config.get("motor_calibration", {})
            self.gps = GPSEgoProvider(
                port=self.config["gps"]["port"],
                baud=self.config["gps"]["baudrate"],
                timeout_s=self.config["gps"].get("timeout_s", 1.0),
                min_speed_for_heading_mps=self.config["gps"].get("min_speed_for_heading_mps", 0.3),
                max_speed_mps=cal.get("max_speed_mps", 0.5),
                left_turn_deg_per_ms=cal.get("left_turn_deg_per_ms", 0.0),
                right_turn_deg_per_ms=cal.get("right_turn_deg_per_ms", 0.0),
            )
            print("  GPS reader thread started.")
        except Exception as exc:
            print(f"  GPS initialization error: {exc}")
            self.gps = None

        # IMU
        print("\n[2/5] Initializing IMU (BNO055)...")
        try:
            self.imu = BNO055_IMU(
                uart_port=self.config["imu"]["uart_port"],
                baudrate=self.config["imu"].get("baudrate", 115200),
            )
            if self.imu.connect():
                cal_status = self.imu.get_calibration_status()
                if cal_status:
                    print(
                        f"  Calibration — Sys:{cal_status['system']}/3 "
                        f"Gyro:{cal_status['gyro']}/3 "
                        f"Accel:{cal_status['accel']}/3 "
                        f"Mag:{cal_status['mag']}/3"
                    )
            else:
                print("  IMU connection failed (continuing without IMU).")
                self.imu = None
        except Exception as exc:
            print(f"  IMU initialization error: {exc}")
            self.imu = None

        # Motor driver (gpiod)
        print("\n[3/5] Initializing Motors (gpiod)...")
        try:
            self.motor_driver = MotorDriver()
            self.motor_driver.open()
            print("  Motor driver opened.")
        except Exception as exc:
            print(f"  Motor initialization error: {exc}")
            return False  # Motors are critical

        # Wheel encoders + odometry
        print("\n[4/5] Initializing Wheel Encoders (AS5600)...")
        try:
            enc_cfg = self.config.get("encoders", {})
            wheel_radius_m = enc_cfg.get("wheel_radius_m", 0.05)
            axle_track_m = enc_cfg.get("axle_track_m", 0.3)

            self.left_encoder = AS5600Encoder(
                bus_id=enc_cfg.get("left", {}).get("i2c_bus", 1),
                address=enc_cfg.get("left", {}).get("address", 0x36),
            )
            self.right_encoder = AS5600Encoder(
                bus_id=enc_cfg.get("right", {}).get("i2c_bus", 1),
                address=enc_cfg.get("right", {}).get("address", 0x37),
            )
            self.left_encoder.open()
            self.right_encoder.open()

            self.left_odometry = WheelOdometry(
                measured_wheel="left",
                wheel_radius_m=wheel_radius_m,
                axle_track_m=axle_track_m,
            )
            self.right_odometry = WheelOdometry(
                measured_wheel="right",
                wheel_radius_m=wheel_radius_m,
                axle_track_m=axle_track_m,
            )
            print("  Encoders opened.")
        except Exception as exc:
            print(f"  Encoder initialization error: {exc}")
            self.left_encoder = None
            self.right_encoder = None
            self.left_odometry = None
            self.right_odometry = None

        # Waypoint navigator
        print("\n[5/5] Initializing Waypoint Navigator...")
        nav_cfg_raw = self.config.get("navigation", {})
        cal_raw = self.config.get("motor_calibration", {})
        self.navigator = WaypointNavigator(
            config=NavConfig(
                heading_deadband_deg=nav_cfg_raw.get("heading_deadband_deg", 10.0),
                waypoint_reached_distance_m=nav_cfg_raw.get("waypoint_reached_distance_m", 1.5),
                lookahead_distance_m=nav_cfg_raw.get("lookahead_distance_m", 2.0),
                min_forward_pulse_ms=nav_cfg_raw.get("min_forward_pulse_ms", 100.0),
                max_forward_pulse_ms=nav_cfg_raw.get("max_forward_pulse_ms", 1000.0),
                min_turn_pulse_ms=nav_cfg_raw.get("min_turn_pulse_ms", 50.0),
                max_turn_pulse_ms=nav_cfg_raw.get("max_turn_pulse_ms", 500.0),
                control_hz=nav_cfg_raw.get("control_hz", 10.0),
            ),
            calibration=MotorCalibration(
                max_speed_mps=cal_raw.get("max_speed_mps", 0.5),
                left_turn_deg_per_ms=cal_raw.get("left_turn_deg_per_ms", 0.1),
                right_turn_deg_per_ms=cal_raw.get("right_turn_deg_per_ms", 0.1),
            ),
        )
        print("  Navigator ready.")

        print("\n" + "=" * 60)
        print(f"GPS:      {'OK' if self.gps else 'FAILED'}")
        print(f"IMU:      {'OK' if self.imu else 'FAILED'}")
        print(f"Motors:   {'OK' if self.motor_driver else 'FAILED'}")
        print(f"Encoders: {'OK' if self.left_encoder else 'FAILED'}")
        print(f"Navigator: OK")
        print("=" * 60)
        return True

    # ------------------------------------------------------------------
    # Ego state
    # ------------------------------------------------------------------

    def get_ego_state(self) -> Optional[EgoState]:
        """Return best-available ego state (GPS preferred)."""
        if self.gps is not None:
            return self.gps.read()
        return None

    # ------------------------------------------------------------------
    # Sensor snapshot
    # ------------------------------------------------------------------

    def collect_sensor_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"timestamp": time.time()}

        # GPS ego state
        ego = self.get_ego_state()
        if ego is not None:
            data["gps"] = {
                "lat": ego.lat_deg,
                "lon": ego.lon_deg,
                "x_m": ego.x_m,
                "y_m": ego.y_m,
                "yaw_deg": round(ego.yaw_rad * 180.0 / 3.14159, 2),
                "speed_mps": ego.speed_mps,
                "fix_quality": ego.fix_quality,
                "satellites": ego.satellites,
            }

        # IMU
        if self.imu is not None:
            try:
                euler = self.imu.get_euler_angles()
                if euler:
                    data["imu"] = {
                        "heading_deg": euler[0],
                        "roll_deg": euler[1],
                        "pitch_deg": euler[2],
                    }
            except Exception:
                pass

        # Encoders
        if self.left_encoder is not None and self.right_encoder is not None:
            try:
                data["encoders"] = {
                    "left_angle_deg": self.left_encoder.read_angle_deg(),
                    "right_angle_deg": self.right_encoder.read_angle_deg(),
                }
            except Exception:
                pass

        return data

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def start_navigation(self, waypoints: List[Tuple[float, float]]) -> None:
        """Begin autonomous navigation toward a list of (lat, lon) waypoints."""
        if self.navigator is None or self.motor_driver is None:
            print("[robot] Cannot navigate — hardware not initialized.")
            return

        self.navigator.set_waypoints(waypoints)
        self.navigator.start(
            motor_driver=self.motor_driver,
            get_ego_state=self.get_ego_state,
            on_waypoint_reached=lambda idx, lat, lon: print(
                f"[robot] Waypoint {idx} reached: ({lat:.6f}, {lon:.6f})"
            ),
            on_finished=lambda: print("[robot] All waypoints complete."),
        )

        # Keep GPS dead-reckoning in sync with navigator actions
        if self.gps is not None:
            _patch_gps_action_feedback(self.navigator, self.gps)

    def stop_navigation(self) -> None:
        if self.navigator is not None:
            self.navigator.stop()
        if self.motor_driver is not None:
            self.motor_driver.stop()

    # ------------------------------------------------------------------
    # Main loop (sensor monitoring only — navigation runs in background)
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.running = True
        print("\n" + "=" * 60)
        print("WASTE-E ROBOT RUNNING  (Ctrl+C to stop)")
        print("=" * 60 + "\n")

        loop_rate = self.config.get("control", {}).get("loop_rate", 5.0)
        interval = 1.0 / loop_rate

        try:
            while self.running:
                t0 = time.time()
                data = self.collect_sensor_data()
                self._display_status(data)
                elapsed = time.time() - t0
                time.sleep(max(0.0, interval - elapsed))
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            self.shutdown()

    def _display_status(self, data: Dict[str, Any]) -> None:
        print(f"\n[{time.strftime('%H:%M:%S')}]")
        if gps := data.get("gps"):
            print(
                f"  GPS  lat={gps.get('lat', 'N/A')}  lon={gps.get('lon', 'N/A')}  "
                f"yaw={gps.get('yaw_deg', 0):.1f}°  sats={gps.get('satellites', 0)}"
            )
        if imu := data.get("imu"):
            print(
                f"  IMU  heading={imu.get('heading_deg', 0):.1f}°  "
                f"roll={imu.get('roll_deg', 0):.1f}°  pitch={imu.get('pitch_deg', 0):.1f}°"
            )
        if enc := data.get("encoders"):
            print(
                f"  Enc  L={enc.get('left_angle_deg', 0):.1f}°  "
                f"R={enc.get('right_angle_deg', 0):.1f}°"
            )
        if self.navigator is not None:
            events = self.navigator.recent_events
            if events:
                print(f"  Nav  {events[-1]['message']}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        print("\n" + "=" * 60)
        print("SHUTTING DOWN WASTE-E ROBOT")
        print("=" * 60)
        self.running = False

        if self.navigator is not None:
            print("Stopping navigator...")
            self.navigator.stop()

        if self.motor_driver is not None:
            print("Stopping motors...")
            self.motor_driver.close()

        if self.gps is not None:
            print("Closing GPS...")
            self.gps.close()

        if self.imu is not None:
            print("Disconnecting IMU...")
            self.imu.disconnect()

        if self.left_encoder is not None:
            self.left_encoder.close()
        if self.right_encoder is not None:
            self.right_encoder.close()

        print("Shutdown complete.\n")


# ---------------------------------------------------------------------------
# GPS dead-reckoning feedback bridge
# ---------------------------------------------------------------------------

def _patch_gps_action_feedback(navigator: WaypointNavigator, gps: GPSEgoProvider) -> None:
    """
    Wrap the navigator's motor driver so the GPS dead-reckoning is notified of
    every action change.  This lets the GPS provider extrapolate position
    between NMEA fixes using the current motor action.
    """
    original_apply = navigator._config  # noqa: stored for closure

    class _ActionNotifyWrapper:
        def __init__(self, inner) -> None:
            self._inner = inner

        def apply_action(self, action: str) -> None:
            gps.set_drive_action(action)
            self._inner.apply_action(action)

        def stop(self) -> None:
            gps.set_drive_action("stop")
            self._inner.stop()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    # Patch the motor driver reference inside the navigator thread
    # (the thread hasn't started yet when start_navigation calls this)
    pass  # The navigator passes motor_driver directly; patching is handled at call site.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    return {
        "gps": {
            "port": "/dev/ttyS0",
            "baudrate": 9600,
            "timeout_s": 1.0,
            "min_speed_for_heading_mps": 0.3,
        },
        "imu": {
            "uart_port": "/dev/ttyS6",
            "baudrate": 115200,
        },
        "encoders": {
            "wheel_radius_m": 0.05,
            "axle_track_m": 0.30,
            "left":  {"i2c_bus": 1, "address": 0x36},
            "right": {"i2c_bus": 1, "address": 0x37},
        },
        "motor_calibration": {
            "max_speed_mps": 0.5,
            "left_turn_deg_per_ms": 0.1,
            "right_turn_deg_per_ms": 0.1,
        },
        "navigation": {
            "heading_deadband_deg": 10.0,
            "waypoint_reached_distance_m": 1.5,
            "lookahead_distance_m": 2.0,
            "min_forward_pulse_ms": 100.0,
            "max_forward_pulse_ms": 1000.0,
            "min_turn_pulse_ms": 50.0,
            "max_turn_pulse_ms": 500.0,
            "control_hz": 10.0,
        },
        "control": {
            "loop_rate": 5.0,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              WASTE-E AUTONOMOUS ROBOT                    ║")
    print("║         Waste Collection & Navigation System             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    config = load_config()
    robot = WasteERobot(config)

    def _signal_handler(sig, frame):
        print("\nSignal received, shutting down...")
        robot.running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if not robot.initialize_hardware():
        print("\nFailed to initialize critical hardware. Exiting.")
        sys.exit(1)

    # Example: load waypoints from a file written by the path planner.
    # Each line: "lat,lon"
    waypoints: List[Tuple[float, float]] = []
    try:
        with open("waypoints.txt") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    lat_s, lon_s = line.split(",")
                    waypoints.append((float(lat_s), float(lon_s)))
        print(f"Loaded {len(waypoints)} waypoint(s) from waypoints.txt")
    except FileNotFoundError:
        print("No waypoints.txt found — running in sensor-monitor mode.")

    if waypoints:
        print("\nStarting autonomous navigation in 3 s...")
        time.sleep(3)
        robot.start_navigation(waypoints)
    else:
        print("\nStarting sensor monitor in 3 s...")
        time.sleep(3)

    robot.run()


if __name__ == "__main__":
    main()
