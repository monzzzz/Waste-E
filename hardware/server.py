import socket
import json
import threading
import time
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Callable
from GPS import NEO8M_GPS
from IMUsensor import BNO055_IMU
from motor import DifferentialDrive, DCMotor


class HardwareServer:
    """
    Hardware data server for Orange Pi
    Collects data from GPS, IMU, and motor controllers
    Sends data to remote server via TCP/UDP

    When dashboard.py is running on the same device, set dashboard_url so
    this server fetches sensor state from the dashboard API instead of
    opening the serial ports directly.  Both programs sharing the same serial
    port splits the NMEA byte-stream, causing the dashboard to miss sentences
    and never update the map.

    Example:
        server = HardwareServer(..., dashboard_url="http://localhost:8888")
    """

    def __init__(self, server_host: str, server_port: int,
                 protocol: str = 'tcp', update_rate: float = 1.0,
                 dashboard_url: Optional[str] = None):
        """
        Args:
            server_host: Remote server IP address
            server_port: Remote server port
            protocol: 'tcp' or 'udp'
            update_rate: Data update rate in Hz
            dashboard_url: Base URL of a running dashboard.py instance
                           (e.g. "http://localhost:8888").  When set, GPS and
                           IMU data are fetched from /api/state instead of
                           opening the serial ports directly.
        """
        self.server_host = server_host
        self.server_port = server_port
        self.dashboard_url = dashboard_url.rstrip("/") if dashboard_url else None
        self.protocol = protocol.lower()
        self.update_rate = update_rate
        self.socket = None
        self.connected = False
        self.running = False
        
        # Hardware components (initialized separately)
        self.gps = None
        self.imu = None
        self.motors = None
        
        # Data thread
        self.data_thread = None
    
    def connect(self) -> bool:
        """Connect to remote server"""
        try:
            if self.protocol == 'tcp':
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.server_host, self.server_port))
                self.socket.settimeout(5.0)
            elif self.protocol == 'udp':
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                # UDP doesn't need explicit connect
            else:
                print(f"Unknown protocol: {self.protocol}")
                return False
            
            self.connected = True
            print(f"Connected to {self.server_host}:{self.server_port} via {self.protocol.upper()}")
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from server"""
        self.running = False
        if self.data_thread:
            self.data_thread.join(timeout=2.0)
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        self.connected = False
        print("Disconnected from server")
    
    def send_data(self, data: Dict[str, Any]) -> bool:
        """
        Send data to server
        
        Args:
            data: Dictionary of data to send
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.socket:
            return False
        
        try:
            # Convert to JSON
            json_data = json.dumps(data)
            message = json_data.encode('utf-8')
            
            if self.protocol == 'tcp':
                # Add message length header for TCP
                length = len(message)
                header = length.to_bytes(4, byteorder='big')
                self.socket.sendall(header + message)
            else:  # UDP
                self.socket.sendto(message, (self.server_host, self.server_port))
            
            return True
            
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False
            return False
    
    def receive_data(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Receive data from server
        
        Args:
            timeout: Receive timeout in seconds
        
        Returns:
            Dictionary of received data or None
        """
        if not self.connected or not self.socket:
            return None
        
        try:
            old_timeout = self.socket.gettimeout()
            if timeout is not None:
                self.socket.settimeout(timeout)
            
            if self.protocol == 'tcp':
                # Read length header
                header = self.socket.recv(4)
                if not header:
                    return None
                
                length = int.from_bytes(header, byteorder='big')
                
                # Read message
                message = b''
                while len(message) < length:
                    chunk = self.socket.recv(length - len(message))
                    if not chunk:
                        return None
                    message += chunk
            else:  # UDP
                message, _ = self.socket.recvfrom(4096)
            
            # Restore timeout
            if timeout is not None:
                self.socket.settimeout(old_timeout)
            
            # Parse JSON
            data = json.loads(message.decode('utf-8'))
            return data
            
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Receive error: {e}")
            return None
    
    def initialize_hardware(self, gps_port: str = '/dev/ttyS1',
                          imu_bus: int = 0,
                          motor_pins: Optional[Dict] = None):
        """
        Initialize hardware components
        
        Args:
            gps_port: GPS serial port
            imu_bus: I2C bus for IMU
            motor_pins: Dictionary with motor pin configuration
        """
        # Initialize GPS
        try:
            self.gps = NEO8M_GPS(port=gps_port)
            if self.gps.connect():
                print("GPS initialized")
        except Exception as e:
            print(f"GPS initialization failed: {e}")
        
        # Initialize IMU
        try:
            self.imu = BNO055_IMU()
            if self.imu.connect():
                print("IMU initialized")
        except Exception as e:
            print(f"IMU initialization failed: {e}")
        
        # Initialize Motors
        if motor_pins:
            try:
                import OPi.GPIO as GPIO
                left_motor = DCMotor(
                    enable_pin=motor_pins['left']['enable'],
                    in1_pin=motor_pins['left']['in1'],
                    in2_pin=motor_pins['left']['in2'],
                    board_mode=GPIO.BOARD
                )
                right_motor = DCMotor(
                    enable_pin=motor_pins['right']['enable'],
                    in1_pin=motor_pins['right']['in1'],
                    in2_pin=motor_pins['right']['in2'],
                    board_mode=GPIO.BOARD
                )
                self.motors = DifferentialDrive(left_motor, right_motor)
                print("Motors initialized")
            except Exception as e:
                print(f"Motor initialization failed: {e}")
    
    def _fetch_dashboard_state(self) -> Optional[Dict[str, Any]]:
        """Fetch current sensor state from a running dashboard.py instance."""
        try:
            url = f"{self.dashboard_url}/api/state"
            with urllib.request.urlopen(url, timeout=2) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            print(f"Dashboard fetch error: {e}")
            return None

    def collect_sensor_data(self) -> Dict[str, Any]:
        """
        Collect data from all sensors.

        When dashboard_url is configured, GPS and IMU are read from the
        dashboard API so this process does not compete for the serial ports.
        Motor data is always read locally (motors are not exposed by the
        dashboard).
        """
        data = {
            'timestamp': time.time(),
            'gps': None,
            'imu': None,
            'motor': None,
        }

        if self.dashboard_url:
            # Pull GPS + IMU from the dashboard instead of the serial ports.
            state = self._fetch_dashboard_state()
            if state:
                g = state.get('gps', {})
                if g.get('lat') is not None and g.get('lon') is not None:
                    data['gps'] = {
                        'latitude':   g.get('lat'),
                        'longitude':  g.get('lon'),
                        'altitude':   g.get('alt'),
                        'speed':      g.get('speed'),
                        'heading':    g.get('heading'),
                        'satellites': g.get('satellites'),
                        'fix_quality': 1 if g.get('fix') else 0,
                    }
                im = state.get('imu', {})
                if im:
                    data['imu'] = {
                        'heading': im.get('heading'),
                        'roll':    im.get('roll'),
                        'pitch':   im.get('pitch'),
                        'temp':    im.get('temp'),
                    }
        else:
            # Direct serial-port access (only safe when dashboard.py is NOT running).
            if self.gps:
                try:
                    gps_data = self.gps.get_gps_data()
                    if gps_data:
                        data['gps'] = gps_data
                except Exception as e:
                    print(f"GPS read error: {e}")

            if self.imu:
                try:
                    imu_data = self.imu.get_all_data()
                    if imu_data:
                        data['imu'] = imu_data
                except Exception as e:
                    print(f"IMU read error: {e}")

        # Motor status is always read locally.
        if self.motors:
            try:
                data['motor'] = {
                    'left_speed':  self.motors.left_motor.get_speed(),
                    'right_speed': self.motors.right_motor.get_speed(),
                }
            except Exception as e:
                print(f"Motor read error: {e}")

        return data
    
    def start_streaming(self):
        """Start streaming sensor data to server"""
        if not self.connected:
            print("Not connected to server")
            return
        
        self.running = True
        self.data_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.data_thread.start()
        print(f"Started streaming data at {self.update_rate} Hz")
    
    def stop_streaming(self):
        """Stop streaming sensor data"""
        self.running = False
        if self.data_thread:
            self.data_thread.join(timeout=2.0)
        print("Stopped streaming data")
    
    def _stream_loop(self):
        """Internal streaming loop"""
        interval = 1.0 / self.update_rate
        
        while self.running and self.connected:
            try:
                # Collect sensor data
                data = self.collect_sensor_data()
                
                # Send to server
                if not self.send_data(data):
                    print("Failed to send data, stopping stream")
                    break
                
                # Wait for next update
                time.sleep(interval)
                
            except Exception as e:
                print(f"Stream error: {e}")
                break
        
        self.running = False
    
    def handle_commands(self, callback: Optional[Callable] = None):
        """
        Listen for commands from server
        
        Args:
            callback: Optional callback function(command_dict) to handle commands
        """
        while self.running and self.connected:
            try:
                command = self.receive_data(timeout=1.0)
                if command:
                    print(f"Received command: {command}")
                    
                    if callback:
                        callback(command)
                    else:
                        # Default command handling
                        self._handle_default_command(command)
                        
            except Exception as e:
                print(f"Command handling error: {e}")
    
    def _handle_default_command(self, command: Dict[str, Any]):
        """Default command handler"""
        cmd_type = command.get('type')
        
        if cmd_type == 'motor' and self.motors:
            # Motor control command
            action = command.get('action')
            speed = command.get('speed', 50)
            
            if action == 'forward':
                self.motors.forward(speed)
            elif action == 'backward':
                self.motors.backward(speed)
            elif action == 'left':
                self.motors.turn_left(speed)
            elif action == 'right':
                self.motors.turn_right(speed)
            elif action == 'stop':
                self.motors.stop()
            
            # Send acknowledgment
            self.send_data({'status': 'ok', 'command': cmd_type})


def main():
    """
    Example usage
    
    Setup:
    1. Make sure all hardware is initialized (GPS, IMU, Motors)
    2. Configure server IP and port
    3. Run the server
    """
    
    # Server configuration
    SERVER_HOST = '192.168.1.100'  # Change to your server IP
    SERVER_PORT = 5000
    
    # Set dashboard_url if dashboard.py is running on the same device.
    # This prevents both programs from opening the same GPS/IMU serial ports,
    # which would split the byte-stream and break GPS on the dashboard map.
    DASHBOARD_URL = 'http://localhost:8888'   # set to None to use serial directly

    # Create server instance
    server = HardwareServer(
        server_host=SERVER_HOST,
        server_port=SERVER_PORT,
        protocol='tcp',
        update_rate=10.0,
        dashboard_url=DASHBOARD_URL,
    )

    try:
        # Initialize hardware
        print("Initializing hardware...")
        server.initialize_hardware(
            gps_port='/dev/ttyS0',
            imu_bus=0,
            motor_pins={
                'left': {'enable': 32, 'in1': 29, 'in2': 31},
                'right': {'enable': 12, 'in1': 11, 'in2': 13}
            }
        )
        
        # Connect to server
        print(f"Connecting to server at {SERVER_HOST}:{SERVER_PORT}...")
        if server.connect():
            # Start streaming data
            server.start_streaming()
            
            # Handle commands (blocking)
            print("Listening for commands... (Press Ctrl+C to stop)")
            server.handle_commands()
        else:
            print("Failed to connect to server")
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        server.stop_streaming()
        server.disconnect()
        
        if server.motors:
            server.motors.cleanup()
        if server.gps:
            server.gps.disconnect()
        if server.imu:
            server.imu.disconnect()
        
        print("Server stopped")


if __name__ == "__main__":
    main()
