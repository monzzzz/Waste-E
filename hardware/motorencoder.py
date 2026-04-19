from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import smbus2

from ego_state import EgoState


class AS5048A_Encoder:
    """
    AS5048A 14-bit Magnetic Rotary Position Sensor (Motor Encoder)
    
    Physical I2C GPIO Connections:
    - VDD  -> Pin 1 or 2 (3.3V or 5V)
    - GND  -> Pin 6, 9, 14, 20, 25, 30, 34, or 39 (Ground)
    - SDA  -> Pin 3 (GPIO 12, I2C0_SDA)
    - SCL  -> Pin 5 (GPIO 11, I2C0_SCL)
    
    Alternative I2C1:
    - SDA  -> Pin 27 (GPIO 19, I2C1_SDA)
    - SCL  -> Pin 28 (GPIO 18, I2C1_SCL)
    
    Features:
    - 14-bit resolution (16384 positions per revolution)
    - 360° absolute angle measurement
    - I2C interface (default address: 0x40)
    - Speed calculation from position changes
    - Direction detection
    
    I2C Addresses: 0x40 (default), can be changed via A1/A2 pins
    """
    
    # Register addresses
    REG_ANGLE_MSB = 0xFE
    REG_ANGLE_LSB = 0xFF
    REG_MAGNITUDE_MSB = 0xFC
    REG_MAGNITUDE_LSB = 0xFD
    REG_DIAGNOSTICS = 0xFB
    REG_AGC = 0xFA
    
    # Constants
    RESOLUTION = 16384  # 14-bit resolution (2^14)
    MAX_ANGLE = 360.0
    
    def __init__(self, i2c_bus: int = 0, address: int = 0x40):
        """
        Initialize AS5048A encoder
        
        Args:
            i2c_bus: I2C bus number (0 for I2C0, 1 for I2C1)
            address: I2C address (0x40, 0x41, 0x42, 0x43 depending on A1/A2 pins)
        """
        self.i2c_bus = i2c_bus
        self.address = address
        self.bus = None
        
        # Position tracking
        self.last_angle = 0.0
        self.last_raw_position = 0
        self.total_rotations = 0
        self.total_angle = 0.0
        
        # Speed calculation
        self.last_time = 0.0
        self.current_speed = 0.0  # RPM
        
    def connect(self) -> bool:
        """Initialize I2C connection"""
        try:
            self.bus = smbus2.SMBus(self.i2c_bus)
            
            # Test connection by reading angle
            self.read_raw_angle()
            
            # Initialize tracking
            self.last_angle = self.read_angle()
            self.last_raw_position = self.read_raw_angle()
            self.last_time = time.time()
            
            return True
        except Exception as e:
            print(f"Error connecting to AS5048A: {e}")
            return False
    
    def disconnect(self):
        """Close I2C connection"""
        if self.bus:
            self.bus.close()
            self.bus = None
    
    def read_raw_angle(self) -> int:
        """
        Read raw angle value (0-16383)
        
        Returns:
            14-bit raw angle value
        """
        if not self.bus:
            print("Encoder not connected. Call connect() first.")
            return 0
        
        try:
            # Read MSB and LSB
            msb = self.bus.read_byte_data(self.address, self.REG_ANGLE_MSB)
            lsb = self.bus.read_byte_data(self.address, self.REG_ANGLE_LSB)
            
            # Combine to 14-bit value (bits 13-0)
            raw_angle = ((msb & 0x3F) << 8) | lsb
            
            return raw_angle
        except Exception as e:
            print(f"Error reading raw angle: {e}")
            return 0
    
    def read_angle(self) -> float:
        """
        Read angle in degrees (0-360)
        
        Returns:
            Angle in degrees
        """
        raw = self.read_raw_angle()
        angle = (raw * self.MAX_ANGLE) / self.RESOLUTION
        return angle
    
    def read_radians(self) -> float:
        """
        Read angle in radians (0-2π)
        
        Returns:
            Angle in radians
        """
        angle_deg = self.read_angle()
        return math.radians(angle_deg)
    
    def read_magnitude(self) -> int:
        """
        Read magnetic field magnitude
        Used to check if magnet is properly positioned
        
        Returns:
            Magnitude value (higher is better, typical range 100-3000)
        """
        if not self.bus:
            print("Encoder not connected. Call connect() first.")
            return 0
        
        try:
            msb = self.bus.read_byte_data(self.address, self.REG_MAGNITUDE_MSB)
            lsb = self.bus.read_byte_data(self.address, self.REG_MAGNITUDE_LSB)
            
            magnitude = ((msb & 0x3F) << 8) | lsb
            return magnitude
        except Exception as e:
            print(f"Error reading magnitude: {e}")
            return 0
    
    def read_diagnostics(self) -> dict:
        """
        Read diagnostic information
        
        Returns:
            Dictionary with diagnostic flags
        """
        if not self.bus:
            print("Encoder not connected. Call connect() first.")
            return {}
        
        try:
            diag = self.bus.read_byte_data(self.address, self.REG_DIAGNOSTICS)
            
            return {
                'comp_high': bool(diag & 0x04),  # CORDIC overflow
                'comp_low': bool(diag & 0x02),   # Magnet too weak
                'cof': bool(diag & 0x01)         # CORDIC overflow flag
            }
        except Exception as e:
            print(f"Error reading diagnostics: {e}")
            return {}
    
    def read_agc(self) -> int:
        """
        Read Automatic Gain Control value
        
        Returns:
            AGC value (0-255)
        """
        if not self.bus:
            print("Encoder not connected. Call connect() first.")
            return 0
        
        try:
            agc = self.bus.read_byte_data(self.address, self.REG_AGC)
            return agc
        except Exception as e:
            print(f"Error reading AGC: {e}")
            return 0
    
    def update_position(self) -> Tuple[float, int]:
        """
        Update position tracking and calculate total rotations
        
        Returns:
            Tuple of (current_angle, total_rotations)
        """
        current_angle = self.read_angle()
        current_raw = self.read_raw_angle()
        
        # Detect rotation crossing 0/360 boundary
        angle_diff = current_angle - self.last_angle
        
        # Crossed from 360 to 0 (forward)
        if angle_diff < -180:
            self.total_rotations += 1
        # Crossed from 0 to 360 (backward)
        elif angle_diff > 180:
            self.total_rotations -= 1
        
        # Update total angle
        self.total_angle = (self.total_rotations * 360.0) + current_angle
        
        # Update tracking
        self.last_angle = current_angle
        self.last_raw_position = current_raw
        
        return current_angle, self.total_rotations
    
    def get_total_angle(self) -> float:
        """
        Get total angle including full rotations
        
        Returns:
            Total angle in degrees (can be > 360 or < 0)
        """
        self.update_position()
        return self.total_angle
    
    def get_rotations(self) -> int:
        """
        Get number of complete rotations
        
        Returns:
            Number of rotations (positive = forward, negative = backward)
        """
        self.update_position()
        return self.total_rotations
    
    def calculate_speed(self) -> float:
        """
        Calculate rotational speed in RPM
        
        Returns:
            Speed in RPM (positive = forward, negative = backward)
        """
        current_time = time.time()
        current_angle = self.read_angle()
        
        # Calculate time difference
        time_diff = current_time - self.last_time
        
        if time_diff < 0.001:  # Avoid division by zero
            return self.current_speed
        
        # Calculate angle difference
        angle_diff = current_angle - self.last_angle
        
        # Handle wrap-around
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        # Calculate speed in degrees per second
        speed_deg_per_sec = angle_diff / time_diff
        
        # Convert to RPM
        rpm = (speed_deg_per_sec / 360.0) * 60.0
        
        # Update tracking
        self.last_angle = current_angle
        self.last_time = current_time
        self.current_speed = rpm
        
        return rpm
    
    def get_speed(self) -> float:
        """
        Get last calculated speed in RPM
        
        Returns:
            Speed in RPM
        """
        return self.current_speed
    
    def reset_position(self):
        """Reset position tracking to zero"""
        self.total_rotations = 0
        self.total_angle = 0.0
        self.last_angle = self.read_angle()
        self.last_raw_position = self.read_raw_angle()
        self.last_time = time.time()
    
    def check_magnet_status(self) -> dict:
        """
        Check if magnet is properly positioned
        
        Returns:
            Dictionary with status information
        """
        magnitude = self.read_magnitude()
        diagnostics = self.read_diagnostics()
        agc = self.read_agc()
        
        # Typical good magnitude range: 200-2000
        magnet_ok = 200 <= magnitude <= 3000
        
        return {
            'magnitude': magnitude,
            'magnet_ok': magnet_ok,
            'too_weak': diagnostics.get('comp_low', False),
            'too_strong': diagnostics.get('comp_high', False),
            'agc': agc,
            'diagnostics': diagnostics
        }


class EncoderMonitor:
    """Monitor for tracking motor position and speed with AS5048A"""
    
    def __init__(self, encoder: AS5048A_Encoder, update_rate: float = 10.0):
        """
        Initialize encoder monitor
        
        Args:
            encoder: AS5048A_Encoder instance
            update_rate: Update rate in Hz
        """
        self.encoder = encoder
        self.update_rate = update_rate
        self.running = False
    
    def start_monitoring(self, duration: Optional[float] = None):
        """
        Start monitoring encoder
        
        Args:
            duration: Monitoring duration in seconds (None for infinite)
        """
        self.running = True
        start_time = time.time()
        interval = 1.0 / self.update_rate
        
        print(f"Monitoring encoder at {self.update_rate} Hz...")
        print("Angle\tRaw\tRotations\tTotal Angle\tSpeed (RPM)")
        print("-" * 70)
        
        try:
            while self.running:
                # Update position
                angle, rotations = self.encoder.update_position()
                total_angle = self.encoder.get_total_angle()
                raw = self.encoder.read_raw_angle()
                
                # Calculate speed
                speed = self.encoder.calculate_speed()
                
                # Display
                print(f"{angle:.2f}°\t{raw}\t{rotations}\t\t{total_angle:.2f}°\t\t{speed:.2f}")
                
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nStopped monitoring")
        finally:
            self.running = False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False


def main():
    """
    Example usage for AS5048A encoder
    
    Hardware Setup:
    1. Connect AS5048A to Orange Pi GPIO pins (I2C0):
       - AS5048A VDD -> Orange Pi Pin 1 (3.3V) or Pin 2 (5V)
       - AS5048A GND -> Orange Pi Pin 6 (GND)
       - AS5048A SDA -> Orange Pi Pin 3 (I2C0_SDA/GPIO12)
       - AS5048A SCL -> Orange Pi Pin 5 (I2C0_SCL/GPIO11)
    
    2. Mount magnet on motor shaft (centered, 0.5-3mm from sensor)
    
    3. Enable I2C on Orange Pi:
       sudo orangepi-config -> System -> Hardware -> i2c0
    
    4. Verify I2C device:
       sudo i2cdetect -y 0
       (Should show device at address 0x40)
    """
    
    # Initialize encoder
    print("Initializing AS5048A encoder...")
    encoder = AS5048A_Encoder(i2c_bus=0, address=0x40)
    
    if encoder.connect():
        print("Encoder connected successfully\n")
        
        try:
            # Check magnet positioning
            print("Checking magnet status...")
            status = encoder.check_magnet_status()
            print(f"Magnitude: {status['magnitude']}")
            print(f"Magnet OK: {status['magnet_ok']}")
            print(f"AGC: {status['agc']}")
            print(f"Diagnostics: {status['diagnostics']}\n")
            
            if not status['magnet_ok']:
                print("WARNING: Magnet may not be properly positioned!")
                print("Adjust distance (0.5-3mm) or alignment\n")
            
            # Reset position
            encoder.reset_position()
            print("Position reset to zero\n")
            
            # Read basic position
            print("Reading position...")
            angle = encoder.read_angle()
            raw = encoder.read_raw_angle()
            print(f"Angle: {angle:.2f}° (raw: {raw})\n")
            
            # Monitor encoder
            print("Starting continuous monitoring...")
            print("Rotate the motor shaft to see position and speed")
            print("Press Ctrl+C to stop\n")
            
            monitor = EncoderMonitor(encoder, update_rate=10.0)
            monitor.start_monitoring()
            
        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            encoder.disconnect()
            print("Encoder disconnected")
    else:
        print("Failed to connect to encoder")
        print("\nTroubleshooting:")
        print("1. Check I2C is enabled: sudo orangepi-config")
        print("2. Check wiring connections")
        print("3. Verify device: sudo i2cdetect -y 0")
        print("4. Check I2C address (default: 0x40)")
        print("5. Install smbus2: pip install smbus2")


# ---------------------------------------------------------------------------
# AS5600Encoder — adapted from self-driving-pipeline/encoder_ego.py
# 12-bit magnetic rotary encoder (4096 counts/rev), I2C interface.
# Reads raw angle via a single 2-byte block read on register 0x0C.
# ---------------------------------------------------------------------------

_AS5600_COUNTS_PER_TURN = 4096
_AS5600_RAW_ANGLE_REGISTER = 0x0C


class AS5600Encoder:
    """
    AS5600 12-bit magnetic encoder over I2C.

    Args:
        bus_id: I2C bus number.
        address: I2C device address (default 0x36).
    """

    def __init__(self, bus_id: int = 1, address: int = 0x36) -> None:
        self._bus_id = int(bus_id)
        self._address = int(address)
        self._bus: smbus2.SMBus | None = None

    def open(self) -> None:
        if self._bus is None:
            self._bus = smbus2.SMBus(self._bus_id)

    def close(self) -> None:
        if self._bus is not None:
            self._bus.close()
            self._bus = None

    def read_raw_angle(self) -> int:
        if self._bus is None:
            raise RuntimeError("AS5600 bus is not open — call open() first")
        high, low = self._bus.read_i2c_block_data(self._address, _AS5600_RAW_ANGLE_REGISTER, 2)
        return ((high << 8) | low) & 0x0FFF

    def read_angle_deg(self) -> float:
        return self.read_raw_angle() * 360.0 / _AS5600_COUNTS_PER_TURN


# ---------------------------------------------------------------------------
# WheelOdometry — adapted from self-driving-pipeline/encoder_ego.py
# Differential drive odometry from a single measured wheel (encoder) and an
# assumed paired-wheel speed ratio.  Produces an EgoState.
# ---------------------------------------------------------------------------

class WheelOdometry:
    """
    Differential drive odometry using one encoder wheel.

    Args:
        measured_wheel: "left" or "right" — which wheel has the encoder.
        wheel_radius_m: Wheel radius in metres.
        axle_track_m: Distance between wheel contact patches in metres.
        paired_wheel_speed_scale: Speed of the un-measured wheel relative to
            the measured wheel (use 1.0 if both wheels move at the same speed).
        speed_deadband_mps: Velocities below this are treated as zero.
    """

    def __init__(
        self,
        measured_wheel: str = "left",
        wheel_radius_m: float = 0.05,
        axle_track_m: float = 0.3,
        paired_wheel_speed_scale: float = 1.0,
        speed_deadband_mps: float = 0.01,
    ) -> None:
        measured_wheel = measured_wheel.strip().lower()
        if measured_wheel not in {"left", "right"}:
            raise ValueError(f"measured_wheel must be 'left' or 'right', got {measured_wheel!r}")
        self._measured_wheel = measured_wheel
        self._wheel_radius_m = wheel_radius_m
        self._axle_track_m = axle_track_m
        self._paired_scale = paired_wheel_speed_scale
        self._deadband = speed_deadband_mps
        self._x_m = 0.0
        self._y_m = 0.0
        self._yaw_rad = 0.0
        self._distance_m = 0.0
        self._last_ts: float | None = None

    def update(
        self,
        *,
        timestamp_s: float,
        angular_velocity_rad_s: float,
        action: str,
    ) -> EgoState:
        linear_mps = angular_velocity_rad_s * self._wheel_radius_m
        left_mps, right_mps = self._wheel_velocities(linear_mps, action)

        if self._last_ts is None:
            self._last_ts = timestamp_s
            return self._ego(timestamp_s, 0.0, 0.0)

        dt = max(timestamp_s - self._last_ts, 0.0)
        self._last_ts = timestamp_s
        v = (left_mps + right_mps) / 2.0
        omega = (right_mps - left_mps) / self._axle_track_m
        yaw_mid = self._yaw_rad + 0.5 * omega * dt
        self._x_m += v * math.cos(yaw_mid) * dt
        self._y_m += v * math.sin(yaw_mid) * dt
        self._yaw_rad = math.atan2(math.sin(self._yaw_rad + omega * dt),
                                   math.cos(self._yaw_rad + omega * dt))
        self._distance_m += abs(v) * dt
        return self._ego(timestamp_s, v, omega)

    def _ego(self, ts: float, speed: float, omega: float) -> EgoState:
        return EgoState(
            timestamp_s=ts,
            x_m=self._x_m,
            y_m=self._y_m,
            yaw_rad=self._yaw_rad,
            speed_mps=speed,
            distance_traveled_m=self._distance_m,
            angular_velocity_rad_s=omega,
            source="encoder_odo",
        )

    def _wheel_velocities(self, measured_mps: float, action: str) -> tuple[float, float]:
        if abs(measured_mps) < self._deadband:
            measured_mps = 0.0
        mag = abs(measured_mps)
        paired = mag * self._paired_scale
        if self._measured_wheel == "left":
            if action == "forward":   return mag, paired
            if action == "backward":  return -mag, -paired
            if action == "left":      return -mag, paired
            if action == "right":     return mag, -paired
            return measured_mps, measured_mps * self._paired_scale
        else:
            if action == "forward":   return paired, mag
            if action == "backward":  return -paired, -mag
            if action == "left":      return -paired, mag
            if action == "right":     return paired, -mag
            return measured_mps * self._paired_scale, measured_mps


if __name__ == "__main__":
    main()
