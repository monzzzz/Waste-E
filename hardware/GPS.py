from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import serial

try:
    import pynmea2
except ImportError:
    pynmea2 = None  # type: ignore

from ego_state import EgoState
from geo_utils import haversine_distance_m, latlon_to_local_m


class NEO8M_GPS:
    """
    GPS module for NEO-8M on Orange Pi
    
    Physical GPIO Connections:
    - VCC  -> Pin 1 or 2 (3.3V or 5V)
    - GND  -> Pin 6, 9, 14, 20, 25, 30, 34, or 39 (Ground)
    - TX   -> Pin 10 (GPIO 0, UART1_RX) - GPS transmits to Orange Pi
    - RX   -> Pin 8  (GPIO 1, UART1_TX) - Orange Pi transmits to GPS
    
    For UART2:
    - TX   -> Pin 16 (GPIO 4, UART2_RX)
    - RX   -> Pin 18 (GPIO 5, UART2_TX)
    
    Software Configuration:
    - UART1: /dev/ttyS0 (default)
    - UART2: /dev/ttyS2
    - Baud rate: 9600 (default for NEO-8M)
    
    Note: Make sure UART is enabled in Orange Pi configuration
    """
    
    def __init__(self, port: str = '/dev/ttyS0', baudrate: int = 9600, timeout: float = 1.0):
        """
        Initialize NEO-8M GPS module
        
        Args:
            port: Serial port (e.g., '/dev/ttyS0', '/dev/ttyS2')
            baudrate: Baud rate (default: 9600)
            timeout: Serial timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        
    def connect(self) -> bool:
        """Establish serial connection to GPS module"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            return True
        except serial.SerialException as e:
            print(f"Error connecting to GPS: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
    
    def get_gps_data(self, max_lines: int = 20) -> Optional[Dict[str, Any]]:
        """
        Get GPS data from NEO-8M module.
        Scans up to max_lines NMEA sentences to find a GGA or RMC sentence,
        since the module outputs many sentence types (GSA, GSV, VTG, etc.).

        Returns:
            Dictionary containing GPS data or None if no valid data available.
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("GPS not connected. Call connect() first.")
            return None

        for _ in range(max_lines):
            try:
                line = self.serial_conn.readline().decode('ascii', errors='replace').strip()
                if not line:
                    continue

                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    msg = pynmea2.parse(line)
                    if msg.gps_qual == 0:
                        continue
                    return {
                        'latitude': msg.latitude if msg.latitude else None,
                        'longitude': msg.longitude if msg.longitude else None,
                        'altitude': msg.altitude if msg.altitude else None,
                        'satellites': msg.num_sats if msg.num_sats else 0,
                        'fix_quality': msg.gps_qual,
                        'timestamp': msg.timestamp if msg.timestamp else None,
                        'horizontal_dilution': msg.horizontal_dil if msg.horizontal_dil else None,
                    }

                elif line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                    msg = pynmea2.parse(line)
                    if msg.status != 'A':
                        continue
                    return {
                        'latitude': msg.latitude if msg.latitude else None,
                        'longitude': msg.longitude if msg.longitude else None,
                        'speed': msg.spd_over_grnd * 1.852 if msg.spd_over_grnd else None,
                        'timestamp': msg.timestamp if msg.timestamp else None,
                        'date': msg.datestamp if msg.datestamp else None,
                        'magnetic_variation': msg.mag_variation if msg.mag_variation else None,
                    }

            except pynmea2.ParseError:
                continue
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading GPS data: {e}")
                break

        return None
    
    def get_position(self) -> Optional[tuple]:
        """
        Get current GPS position (simplified)
        
        Returns:
            Tuple of (latitude, longitude) or None if unavailable
        """
        data = self.get_gps_data()
        if data and data.get('latitude') and data.get('longitude'):
            return (data['latitude'], data['longitude'])
        return None
    
    def read_continuous(self, callback=None, max_attempts: int = 50):
        """
        Continuously read GPS data until valid fix is obtained
        
        Args:
            callback: Optional callback function to process GPS data
            max_attempts: Maximum number of read attempts
        """
        attempts = 0
        while attempts < max_attempts:
            data = self.get_gps_data()
            if data:
                if callback:
                    callback(data)
                else:
                    print(f"GPS Data: {data}")
                return data
            attempts += 1
        
        print("No valid GPS data received")
        return None


def main():
    """
    Example usage
    
    Hardware Setup:
    1. Connect NEO-8M to Orange Pi GPIO pins:
       - NEO-8M VCC -> Orange Pi Pin 2 (5V) or Pin 1 (3.3V)
       - NEO-8M GND -> Orange Pi Pin 6 (GND)
       - NEO-8M TX  -> Orange Pi Pin 10 (UART1_RX/GPIO0)
       - NEO-8M RX  -> Orange Pi Pin 8  (UART1_TX/GPIO1)
    
    2. Enable UART on Orange Pi (if not already enabled):
       sudo orangepi-config -> System -> Hardware -> uart1
    """
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyS0'

    print(f"Opening {port} at 9600 baud...")
    try:
        ser = serial.Serial(port=port, baudrate=9600, timeout=1.0)
        ser.reset_input_buffer()
    except serial.SerialException as exc:
        print(f"Cannot open port: {exc}")
        return

    print("Reading raw NMEA lines (Ctrl+C to stop):\n")
    try:
        while True:
            raw = ser.readline()
            if not raw:
                print("  (no data — check wiring and baud rate)")
                continue
            line = raw.decode("ascii", errors="replace").strip()
            print(f"  {line}")

            # Parse and show fix info when available
            if line.startswith(("$GPGGA", "$GNGGA")):
                fields = line.split(",")
                fix_q = fields[6] if len(fields) > 6 else "?"
                sats   = fields[7] if len(fields) > 7 else "?"
                lat_raw = fields[2] if len(fields) > 3 else ""
                lat_hem = fields[3] if len(fields) > 3 else ""
                lon_raw = fields[4] if len(fields) > 5 else ""
                lon_hem = fields[5] if len(fields) > 5 else ""
                fix_label = {"0": "NO FIX", "1": "GPS", "2": "DGPS"}.get(fix_q, fix_q)
                print(f"    → fix={fix_label}  sats={sats}  lat={lat_raw}{lat_hem}  lon={lon_raw}{lon_hem}")
                if fix_q == "0":
                    print("    ⚠ No satellite fix yet — move outdoors or wait.")

            elif line.startswith(("$GPRMC", "$GNRMC")):
                fields = line.split(",")
                status = fields[2] if len(fields) > 2 else "?"
                status_label = "ACTIVE (valid)" if status == "A" else "VOID (no fix)"
                print(f"    → status={status_label}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


def test_ego_provider(port: str = "/dev/ttyS0", baud: int = 9600) -> None:
    """Test GPSEgoProvider — prints ego state (x_m, y_m, yaw, lat, lon) continuously."""
    print(f"GPSEgoProvider test on {port} @ {baud} baud")
    print("Waiting for first GPS fix... (Ctrl+C to stop)\n")
    provider = GPSEgoProvider(port=port, baud=baud)
    try:
        while True:
            ego = provider.read()
            print(
                f"  x={ego.x_m:8.2f} m  y={ego.y_m:8.2f} m  "
                f"yaw={ego.yaw_rad * 57.296:6.1f}°  "
                f"lat={ego.lat_deg}  lon={ego.lon_deg}  "
                f"sats={ego.satellites}  fix={ego.fix_quality}"
            )
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        provider.close()


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "raw"
    port = sys.argv[2] if len(sys.argv) > 2 else "/dev/ttyS0"
    if mode == "ego":
        test_ego_provider(port=port)
    else:
        main()


# ---------------------------------------------------------------------------
# GPSEgoProvider — adapted from self-driving-pipeline/alpamayo_pipeline/robot/gps_ego.py
# Reads NMEA sentences in a background thread and maintains a dead-reckoning
# ego state (x_m, y_m, yaw_rad) fused with GPS fixes.
# ---------------------------------------------------------------------------

@dataclass
class _GPSFix:
    timestamp_s: float
    lat_deg: float
    lon_deg: float
    speed_mps: float = 0.0
    course_deg: float | None = None
    fix_quality: int = 0
    satellites: int = 0
    hdop: float | None = None


class GPSEgoProvider:
    """
    UART GPS reader that produces an EgoState with dead reckoning.

    Position origin is set on the first valid fix.  Between fixes the
    position is extrapolated from the last known motor action using the
    provided calibration values.

    Args:
        port: Serial device (e.g. '/dev/ttyS0').
        baud: Baud rate (9600 for NEO-8M).
        timeout_s: Serial read timeout.
        min_speed_for_heading_mps: Use GPS course for yaw only when speed
            exceeds this threshold.
        max_speed_mps: Robot top speed used for dead-reckoning forward motion.
        left_turn_deg_per_ms: Degrees/ms for left-turn dead reckoning.
        right_turn_deg_per_ms: Degrees/ms for right-turn dead reckoning.
    """

    def __init__(
        self,
        port: str = "/dev/ttyS0",
        baud: int = 9600,
        timeout_s: float = 1.0,
        min_speed_for_heading_mps: float = 0.3,
        max_speed_mps: float = 0.5,
        left_turn_deg_per_ms: float = 0.0,
        right_turn_deg_per_ms: float = 0.0,
    ) -> None:
        self._serial = serial.Serial(port=port, baudrate=baud, timeout=max(timeout_s, 0.1))
        self._serial.reset_input_buffer()
        self._min_speed_for_heading_mps = min_speed_for_heading_mps
        self._max_speed_mps = max_speed_mps
        self._left_turn_deg_per_ms = left_turn_deg_per_ms
        self._right_turn_deg_per_ms = right_turn_deg_per_ms

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._origin_lat: float | None = None
        self._origin_lon: float | None = None
        self._latest_fix: _GPSFix | None = None
        self._last_position_fix: _GPSFix | None = None
        self._estimated_x_m = 0.0
        self._estimated_y_m = 0.0
        self._estimated_yaw_rad = 0.0
        self._last_pose_update_s = time.time()
        self._current_action = "stop"

        self._thread = threading.Thread(target=self._reader_loop, name="gps-uart-reader", daemon=True)
        self._thread.start()

    def read(self, timestamp_s: float | None = None) -> EgoState:
        if timestamp_s is None:
            timestamp_s = time.time()
        with self._lock:
            self._advance_dead_reckoning(timestamp_s)
            fix = self._latest_fix
            return EgoState(
                timestamp_s=timestamp_s,
                x_m=self._estimated_x_m,
                y_m=self._estimated_y_m,
                yaw_rad=self._estimated_yaw_rad,
                speed_mps=fix.speed_mps if fix else 0.0,
                lat_deg=fix.lat_deg if fix else None,
                lon_deg=fix.lon_deg if fix else None,
                fix_quality=fix.fix_quality if fix else 0,
                satellites=fix.satellites if fix else 0,
                hdop=fix.hdop if fix else None,
                source="gps_uart",
            )

    def set_drive_action(self, action: str) -> None:
        with self._lock:
            self._advance_dead_reckoning(time.time())
            self._current_action = action

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._serial.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw = self._serial.readline()
            except serial.SerialException:
                break
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if not line.startswith("$") or not _gps_checksum_ok(line):
                continue
            try:
                fix = self._parse_line(line)
            except ValueError:
                continue
            if fix is None:
                continue
            with self._lock:
                self._apply_fix(fix)

    def _parse_line(self, line: str) -> _GPSFix | None:
        fields = line.split(",")
        sentence_type = fields[0][-3:]
        if sentence_type == "GGA" and len(fields) >= 10:
            lat = _nmea_to_decimal(fields[2], fields[3])
            lon = _nmea_to_decimal(fields[4], fields[5])
            if lat is None or lon is None:
                return None
            return _GPSFix(
                timestamp_s=time.time(),
                lat_deg=lat,
                lon_deg=lon,
                fix_quality=int(fields[6] or "0"),
                satellites=int(fields[7] or "0"),
                hdop=float(fields[8]) if fields[8] else None,
            )
        if sentence_type == "RMC" and len(fields) >= 9 and fields[2] == "A":
            lat = _nmea_to_decimal(fields[3], fields[4])
            lon = _nmea_to_decimal(fields[5], fields[6])
            if lat is None or lon is None:
                return None
            speed_knots = float(fields[7]) if fields[7] else 0.0
            course_deg = float(fields[8]) if fields[8] else None
            return _GPSFix(
                timestamp_s=time.time(),
                lat_deg=lat,
                lon_deg=lon,
                speed_mps=speed_knots * 0.514444,
                course_deg=course_deg,
                fix_quality=self._latest_fix.fix_quality if self._latest_fix else 1,
                satellites=self._latest_fix.satellites if self._latest_fix else 0,
                hdop=self._latest_fix.hdop if self._latest_fix else None,
            )
        return None

    def _apply_fix(self, fix: _GPSFix) -> None:
        if self._origin_lat is None or self._origin_lon is None:
            self._origin_lat = fix.lat_deg
            self._origin_lon = fix.lon_deg
        self._advance_dead_reckoning(fix.timestamp_s)
        self._estimated_x_m, self._estimated_y_m = latlon_to_local_m(
            self._origin_lat, self._origin_lon, fix.lat_deg, fix.lon_deg
        )
        if fix.course_deg is not None and fix.speed_mps >= self._min_speed_for_heading_mps:
            self._estimated_yaw_rad = _course_to_yaw_rad(fix.course_deg)
        elif self._last_position_fix is not None:
            dist = haversine_distance_m(
                self._last_position_fix.lat_deg, self._last_position_fix.lon_deg,
                fix.lat_deg, fix.lon_deg,
            )
            if dist >= 0.75:
                dx, dy = latlon_to_local_m(
                    self._last_position_fix.lat_deg, self._last_position_fix.lon_deg,
                    fix.lat_deg, fix.lon_deg,
                )
                self._estimated_yaw_rad = math.atan2(dy, dx)
        self._last_position_fix = fix
        if self._latest_fix is None:
            self._latest_fix = fix
            return
        self._latest_fix = _GPSFix(
            timestamp_s=fix.timestamp_s,
            lat_deg=fix.lat_deg,
            lon_deg=fix.lon_deg,
            speed_mps=fix.speed_mps if fix.speed_mps > 0.0 else self._latest_fix.speed_mps,
            course_deg=fix.course_deg if fix.course_deg is not None else self._latest_fix.course_deg,
            fix_quality=fix.fix_quality,
            satellites=fix.satellites,
            hdop=fix.hdop if fix.hdop is not None else self._latest_fix.hdop,
        )

    def _advance_dead_reckoning(self, now_s: float) -> None:
        dt_s = max(now_s - self._last_pose_update_s, 0.0)
        if dt_s <= 0.0:
            return
        action = self._current_action
        if action == "left" and self._left_turn_deg_per_ms > 0.0:
            self._estimated_yaw_rad += math.radians(self._left_turn_deg_per_ms * dt_s * 1000.0)
        elif action == "right" and self._right_turn_deg_per_ms > 0.0:
            self._estimated_yaw_rad -= math.radians(self._right_turn_deg_per_ms * dt_s * 1000.0)
        elif action == "forward" and self._max_speed_mps > 0.0:
            distance_m = self._max_speed_mps * dt_s
            self._estimated_x_m += math.cos(self._estimated_yaw_rad) * distance_m
            self._estimated_y_m += math.sin(self._estimated_yaw_rad) * distance_m
        self._estimated_yaw_rad = math.atan2(
            math.sin(self._estimated_yaw_rad), math.cos(self._estimated_yaw_rad)
        )
        self._last_pose_update_s = now_s


def _gps_checksum_ok(line: str) -> bool:
    if "*" not in line:
        return False
    body, checksum = line[1:].split("*", 1)
    if len(checksum) < 2:
        return False
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(checksum[:2], 16)
    except ValueError:
        return False


def _nmea_to_decimal(raw_value: str, hemisphere: str) -> float | None:
    if not raw_value:
        return None
    value = float(raw_value)
    degrees = int(value / 100)
    minutes = value - (degrees * 100)
    decimal = degrees + (minutes / 60.0)
    if hemisphere in {"S", "W"}:
        decimal *= -1
    return decimal


def _course_to_yaw_rad(course_deg: float) -> float:
    return math.radians(90.0 - course_deg)
