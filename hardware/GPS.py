import serial
import pynmea2
from typing import Optional, Dict, Any


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
    - UART1: /dev/ttyS1 (default)
    - UART2: /dev/ttyS2
    - Baud rate: 9600 (default for NEO-8M)
    
    Note: Make sure UART is enabled in Orange Pi configuration
    """
    
    def __init__(self, port: str = '/dev/ttyS1', baudrate: int = 9600, timeout: float = 1.0):
        """
        Initialize NEO-8M GPS module
        
        Args:
            port: Serial port (e.g., '/dev/ttyS1', '/dev/ttyS2')
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
    
    def get_gps_data(self) -> Optional[Dict[str, Any]]:
        """
        Get GPS data from NEO-8M module
        
        Returns:
            Dictionary containing GPS data:
            - latitude: Latitude in decimal degrees
            - longitude: Longitude in decimal degrees
            - altitude: Altitude in meters
            - speed: Speed over ground in km/h
            - timestamp: UTC timestamp
            - satellites: Number of satellites in use
            - fix_quality: GPS fix quality (0=invalid, 1=GPS fix, 2=DGPS fix)
            Returns None if no valid data available
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("GPS not connected. Call connect() first.")
            return None
        
        try:
            # Read line from GPS
            line = self.serial_conn.readline().decode('ascii', errors='replace').strip()
            
            # Parse NMEA sentence
            if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                msg = pynmea2.parse(line)
                
                # Check if we have a valid fix
                if msg.gps_qual == 0:
                    return None
                
                return {
                    'latitude': msg.latitude if msg.latitude else None,
                    'longitude': msg.longitude if msg.longitude else None,
                    'altitude': msg.altitude if msg.altitude else None,
                    'satellites': msg.num_sats if msg.num_sats else 0,
                    'fix_quality': msg.gps_qual,
                    'timestamp': msg.timestamp if msg.timestamp else None,
                    'horizontal_dilution': msg.horizontal_dil if msg.horizontal_dil else None
                }
            
            elif line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                msg = pynmea2.parse(line)
                
                # Check if data is valid
                if msg.status != 'A':
                    return None
                
                return {
                    'latitude': msg.latitude if msg.latitude else None,
                    'longitude': msg.longitude if msg.longitude else None,
                    'speed': msg.spd_over_grnd * 1.852 if msg.spd_over_grnd else None,  # Convert knots to km/h
                    'timestamp': msg.timestamp if msg.timestamp else None,
                    'date': msg.datestamp if msg.datestamp else None,
                    'magnetic_variation': msg.mag_variation if msg.mag_variation else None
                }
                
        except pynmea2.ParseError as e:
            # Ignore parse errors and try next sentence
            pass
        except UnicodeDecodeError:
            # Ignore decode errors
            pass
        except Exception as e:
            print(f"Error reading GPS data: {e}")
        
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
    # Initialize GPS on Orange Pi UART1 (GPIO pins 8 and 10)
    gps = NEO8M_GPS(port='/dev/ttyS1', baudrate=9600)
    
    # Connect to GPS module
    if gps.connect():
        print("GPS connected successfully")
        
        try:
            # Read GPS data continuously
            print("Reading GPS data... (Press Ctrl+C to stop)")
            while True:
                data = gps.get_gps_data()
                if data:
                    print(f"\nLatitude: {data.get('latitude')}")
                    print(f"Longitude: {data.get('longitude')}")
                    print(f"Altitude: {data.get('altitude')} m")
                    print(f"Satellites: {data.get('satellites')}")
                    print(f"Fix Quality: {data.get('fix_quality')}")
                
        except KeyboardInterrupt:
            print("\nStopping GPS reading")
        finally:
            gps.disconnect()
            print("GPS disconnected")
    else:
        print("Failed to connect to GPS")


if __name__ == "__main__":
    main()
