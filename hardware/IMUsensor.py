import board
import busio
import adafruit_bno055
from typing import Optional, Dict, Any, Tuple
import time


class BNO055_IMU:
    """
    BNO055 9-DOF Absolute Orientation IMU Module for Orange Pi
    
    Physical I2C GPIO Connections:
    - VIN  -> Pin 1 or 2 (3.3V or 5V)
    - GND  -> Pin 6, 9, 14, 20, 25, 30, 34, or 39 (Ground)
    - SDA  -> Pin 3 (GPIO 12, I2C0_SDA)
    - SCL  -> Pin 5 (GPIO 11, I2C0_SCL)
    
    Alternative I2C1:
    - SDA  -> Pin 27 (GPIO 19, I2C1_SDA)
    - SCL  -> Pin 28 (GPIO 18, I2C1_SCL)
    
    Features:
    - 9-axis absolute orientation (quaternion, euler angles)
    - 3-axis accelerometer
    - 3-axis gyroscope
    - 3-axis magnetometer
    - Temperature sensor
    - Built-in sensor fusion algorithm
    
    I2C Address: 0x28 (default) or 0x29
    """
    
    def __init__(self, i2c_bus: int = 0, address: int = 0x28):
        """
        Initialize BNO055 IMU sensor
        
        Args:
            i2c_bus: I2C bus number (0 for I2C0, 1 for I2C1)
            address: I2C address (0x28 or 0x29)
        """
        self.i2c_bus = i2c_bus
        self.address = address
        self.sensor = None
        self.i2c = None
        
    def connect(self) -> bool:
        """Initialize I2C connection and sensor"""
        try:
            # Initialize I2C
            if self.i2c_bus == 0:
                self.i2c = busio.I2C(board.SCL, board.SDA)
            else:
                # For I2C1, you may need to use specific pins
                self.i2c = busio.I2C(board.SCL, board.SDA)
            
            # Initialize BNO055 sensor
            self.sensor = adafruit_bno055.BNO055_I2C(self.i2c, address=self.address)
            
            # Give sensor time to initialize
            time.sleep(0.5)
            
            return True
        except Exception as e:
            print(f"Error connecting to BNO055: {e}")
            return False
    
    def disconnect(self):
        """Close I2C connection"""
        if self.i2c:
            self.i2c.deinit()
            self.i2c = None
            self.sensor = None
    
    def get_calibration_status(self) -> Optional[Dict[str, int]]:
        """
        Get calibration status for all sensors
        
        Returns:
            Dictionary with calibration status (0-3, where 3 is fully calibrated):
            - system: Overall system calibration
            - gyro: Gyroscope calibration
            - accel: Accelerometer calibration
            - mag: Magnetometer calibration
        """
        if not self.sensor:
            print("Sensor not connected. Call connect() first.")
            return None
        
        try:
            status = self.sensor.calibration_status
            return {
                'system': status[0],
                'gyro': status[1],
                'accel': status[2],
                'mag': status[3]
            }
        except Exception as e:
            print(f"Error reading calibration status: {e}")
            return None
    
    def is_calibrated(self) -> bool:
        """Check if sensor is fully calibrated"""
        status = self.get_calibration_status()
        if status:
            return all(v == 3 for v in status.values())
        return False
    
    def get_euler_angles(self) -> Optional[Tuple[float, float, float]]:
        """
        Get orientation as Euler angles
        
        Returns:
            Tuple of (heading, roll, pitch) in degrees
            - heading: 0-360° (yaw/compass heading)
            - roll: -180 to 180°
            - pitch: -90 to 90°
        """
        if not self.sensor:
            print("Sensor not connected. Call connect() first.")
            return None
        
        try:
            return self.sensor.euler
        except Exception as e:
            print(f"Error reading Euler angles: {e}")
            return None
    
    def get_quaternion(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Get orientation as quaternion
        
        Returns:
            Tuple of (w, x, y, z) quaternion values
        """
        if not self.sensor:
            print("Sensor not connected. Call connect() first.")
            return None
        
        try:
            return self.sensor.quaternion
        except Exception as e:
            print(f"Error reading quaternion: {e}")
            return None
    
    def get_accelerometer(self) -> Optional[Tuple[float, float, float]]:
        """
        Get linear acceleration (without gravity)
        
        Returns:
            Tuple of (x, y, z) acceleration in m/s²
        """
        if not self.sensor:
            print("Sensor not connected. Call connect() first.")
            return None
        
        try:
            return self.sensor.linear_acceleration
        except Exception as e:
            print(f"Error reading accelerometer: {e}")
            return None
    
    def get_gyroscope(self) -> Optional[Tuple[float, float, float]]:
        """
        Get angular velocity
        
        Returns:
            Tuple of (x, y, z) angular velocity in rad/s
        """
        if not self.sensor:
            print("Sensor not connected. Call connect() first.")
            return None
        
        try:
            return self.sensor.gyro
        except Exception as e:
            print(f"Error reading gyroscope: {e}")
            return None
    
    def get_magnetometer(self) -> Optional[Tuple[float, float, float]]:
        """
        Get magnetic field strength
        
        Returns:
            Tuple of (x, y, z) magnetic field in microteslas
        """
        if not self.sensor:
            print("Sensor not connected. Call connect() first.")
            return None
        
        try:
            return self.sensor.magnetic
        except Exception as e:
            print(f"Error reading magnetometer: {e}")
            return None
    
    def get_gravity(self) -> Optional[Tuple[float, float, float]]:
        """
        Get gravity vector
        
        Returns:
            Tuple of (x, y, z) gravity in m/s²
        """
        if not self.sensor:
            print("Sensor not connected. Call connect() first.")
            return None
        
        try:
            return self.sensor.gravity
        except Exception as e:
            print(f"Error reading gravity: {e}")
            return None
    
    def get_temperature(self) -> Optional[int]:
        """
        Get sensor temperature
        
        Returns:
            Temperature in Celsius
        """
        if not self.sensor:
            print("Sensor not connected. Call connect() first.")
            return None
        
        try:
            return self.sensor.temperature
        except Exception as e:
            print(f"Error reading temperature: {e}")
            return None
    
    def get_all_data(self) -> Optional[Dict[str, Any]]:
        """
        Get all sensor data at once
        
        Returns:
            Dictionary containing all sensor readings
        """
        if not self.sensor:
            print("Sensor not connected. Call connect() first.")
            return None
        
        try:
            euler = self.get_euler_angles()
            quat = self.get_quaternion()
            accel = self.get_accelerometer()
            gyro = self.get_gyroscope()
            mag = self.get_magnetometer()
            gravity = self.get_gravity()
            temp = self.get_temperature()
            cal_status = self.get_calibration_status()
            
            return {
                'euler': {
                    'heading': euler[0] if euler else None,
                    'roll': euler[1] if euler else None,
                    'pitch': euler[2] if euler else None
                },
                'quaternion': {
                    'w': quat[0] if quat else None,
                    'x': quat[1] if quat else None,
                    'y': quat[2] if quat else None,
                    'z': quat[3] if quat else None
                },
                'accelerometer': {
                    'x': accel[0] if accel else None,
                    'y': accel[1] if accel else None,
                    'z': accel[2] if accel else None
                },
                'gyroscope': {
                    'x': gyro[0] if gyro else None,
                    'y': gyro[1] if gyro else None,
                    'z': gyro[2] if gyro else None
                },
                'magnetometer': {
                    'x': mag[0] if mag else None,
                    'y': mag[1] if mag else None,
                    'z': mag[2] if mag else None
                },
                'gravity': {
                    'x': gravity[0] if gravity else None,
                    'y': gravity[1] if gravity else None,
                    'z': gravity[2] if gravity else None
                },
                'temperature': temp,
                'calibration': cal_status
            }
        except Exception as e:
            print(f"Error reading all data: {e}")
            return None
    
    def wait_for_calibration(self, timeout: int = 60):
        """
        Wait for sensor to be fully calibrated
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        print("Calibrating BNO055... Move the sensor in a figure-8 pattern")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_calibration_status()
            if status:
                print(f"Calibration: Sys={status['system']}, Gyro={status['gyro']}, "
                      f"Accel={status['accel']}, Mag={status['mag']}")
                
                if self.is_calibrated():
                    print("Sensor fully calibrated!")
                    return True
            
            time.sleep(1)
        
        print("Calibration timeout reached")
        return False


def main():
    """
    Example usage
    
    Hardware Setup:
    1. Connect BNO055 to Orange Pi GPIO pins (I2C0):
       - BNO055 VIN -> Orange Pi Pin 2 (5V) or Pin 1 (3.3V)
       - BNO055 GND -> Orange Pi Pin 6 (GND)
       - BNO055 SDA -> Orange Pi Pin 3 (I2C0_SDA/GPIO12)
       - BNO055 SCL -> Orange Pi Pin 5 (I2C0_SCL/GPIO11)
    
    2. Enable I2C on Orange Pi (if not already enabled):
       sudo orangepi-config -> System -> Hardware -> i2c0
    
    3. Verify I2C device is detected:
       sudo i2cdetect -y 0
       (Should show device at address 0x28)
    """
    # Initialize IMU sensor on I2C0
    imu = BNO055_IMU(i2c_bus=0, address=0x28)
    
    # Connect to sensor
    if imu.connect():
        print("BNO055 connected successfully\n")
        
        try:
            # Check calibration status
            print("Checking calibration status...")
            status = imu.get_calibration_status()
            if status:
                print(f"System: {status['system']}/3, Gyro: {status['gyro']}/3, "
                      f"Accel: {status['accel']}/3, Mag: {status['mag']}/3\n")
            
            # Read sensor data continuously
            print("Reading IMU data... (Press Ctrl+C to stop)\n")
            while True:
                # Get Euler angles
                euler = imu.get_euler_angles()
                if euler:
                    print(f"Orientation - Heading: {euler[0]:.2f}°, "
                          f"Roll: {euler[1]:.2f}°, Pitch: {euler[2]:.2f}°")
                
                # Get accelerometer
                accel = imu.get_accelerometer()
                if accel:
                    print(f"Acceleration - X: {accel[0]:.2f}, "
                          f"Y: {accel[1]:.2f}, Z: {accel[2]:.2f} m/s²")
                
                # Get gyroscope
                gyro = imu.get_gyroscope()
                if gyro:
                    print(f"Gyroscope - X: {gyro[0]:.2f}, "
                          f"Y: {gyro[1]:.2f}, Z: {gyro[2]:.2f} rad/s")
                
                # Get temperature
                temp = imu.get_temperature()
                if temp:
                    print(f"Temperature: {temp}°C")
                
                print("-" * 50)
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping IMU reading")
        finally:
            imu.disconnect()
            print("IMU disconnected")
    else:
        print("Failed to connect to BNO055")
        print("\nTroubleshooting:")
        print("1. Check I2C is enabled: sudo orangepi-config")
        print("2. Check wiring connections")
        print("3. Verify device: sudo i2cdetect -y 0")
        print("4. Install required packages: pip install adafruit-circuitpython-bno055")


if __name__ == "__main__":
    main()
