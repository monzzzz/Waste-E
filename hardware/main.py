#!/usr/bin/env python3
"""
Waste-E Robot Main Control System
Integrates GPS, IMU, Motor Encoders, and Motors
Communicates with AI server for autonomous control
"""

import time
import json
import sys
import signal
from typing import Dict, Any, Optional
import OPi.GPIO as GPIO

# Import hardware modules
from GPS import NEO8M_GPS
from IMUsensor import BNO055_IMU
from motor import DCMotor, DifferentialDrive
from motorencoder import AS5048A_Encoder
from server import HardwareServer


class WasteERobot:
    """Main robot control class integrating all hardware components"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize robot with configuration
        
        Args:
            config: Configuration dictionary with hardware settings
        """
        self.config = config
        
        # Hardware components
        self.gps = None
        self.imu = None
        self.left_motor = None
        self.right_motor = None
        self.drive = None
        self.left_encoder = None
        self.right_encoder = None
        self.server = None
        
        # State tracking
        self.running = False
        self.autonomous_mode = False
        
    def initialize_hardware(self) -> bool:
        """Initialize all hardware components"""
        print("=" * 60)
        print("WASTE-E ROBOT INITIALIZATION")
        print("=" * 60)
        
        # Initialize GPS
        print("\n[1/6] Initializing GPS (NEO-8M)...")
        try:
            self.gps = NEO8M_GPS(
                port=self.config['gps']['port'],
                baudrate=self.config['gps']['baudrate']
            )
            if self.gps.connect():
                print("✓ GPS connected successfully")
            else:
                print("✗ GPS connection failed (will continue without GPS)")
                self.gps = None
        except Exception as e:
            print(f"✗ GPS initialization error: {e}")
            self.gps = None
        
        # Initialize IMU
        print("\n[2/6] Initializing IMU (BNO055)...")
        try:
            self.imu = BNO055_IMU(
                i2c_bus=self.config['imu']['i2c_bus'],
                address=self.config['imu']['address']
            )
            if self.imu.connect():
                print("✓ IMU connected successfully")
                
                # Check calibration
                cal_status = self.imu.get_calibration_status()
                if cal_status:
                    print(f"  Calibration - Sys:{cal_status['system']}/3 "
                          f"Gyro:{cal_status['gyro']}/3 "
                          f"Accel:{cal_status['accel']}/3 "
                          f"Mag:{cal_status['mag']}/3")
            else:
                print("✗ IMU connection failed (will continue without IMU)")
                self.imu = None
        except Exception as e:
            print(f"✗ IMU initialization error: {e}")
            self.imu = None
        
        # Initialize Motors
        print("\n[3/6] Initializing Motors (12V DC Motors)...")
        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            
            self.left_motor = DCMotor(
                enable_pin=self.config['motors']['left']['enable'],
                in1_pin=self.config['motors']['left']['in1'],
                in2_pin=self.config['motors']['left']['in2'],
                board_mode=GPIO.BOARD
            )
            
            self.right_motor = DCMotor(
                enable_pin=self.config['motors']['right']['enable'],
                in1_pin=self.config['motors']['right']['in1'],
                in2_pin=self.config['motors']['right']['in2'],
                board_mode=GPIO.BOARD
            )
            
            self.drive = DifferentialDrive(self.left_motor, self.right_motor)
            print("✓ Motors initialized successfully")
        except Exception as e:
            print(f"✗ Motor initialization error: {e}")
            return False
        
        # Initialize Motor Encoders
        print("\n[4/6] Initializing Motor Encoders (AS5048A)...")
        try:
            self.left_encoder = AS5048A_Encoder(
                i2c_bus=self.config['encoders']['left']['i2c_bus'],
                address=self.config['encoders']['left']['address']
            )
            
            self.right_encoder = AS5048A_Encoder(
                i2c_bus=self.config['encoders']['right']['i2c_bus'],
                address=self.config['encoders']['right']['address']
            )
            
            if self.left_encoder.connect() and self.right_encoder.connect():
                print("✓ Encoders connected successfully")
                
                # Check magnet status
                left_status = self.left_encoder.check_magnet_status()
                right_status = self.right_encoder.check_magnet_status()
                
                print(f"  Left encoder - Magnitude: {left_status['magnitude']} "
                      f"({'OK' if left_status['magnet_ok'] else 'CHECK MAGNET'})")
                print(f"  Right encoder - Magnitude: {right_status['magnitude']} "
                      f"({'OK' if right_status['magnet_ok'] else 'CHECK MAGNET'})")
                
                # Reset encoder positions
                self.left_encoder.reset_position()
                self.right_encoder.reset_position()
            else:
                print("✗ Encoder connection failed (will continue without encoders)")
                self.left_encoder = None
                self.right_encoder = None
        except Exception as e:
            print(f"✗ Encoder initialization error: {e}")
            self.left_encoder = None
            self.right_encoder = None
        
        # Initialize Server Connection
        print("\n[5/6] Connecting to AI Server...")
        try:
            self.server = HardwareServer(
                server_host=self.config['server']['host'],
                server_port=self.config['server']['port'],
                protocol=self.config['server']['protocol'],
                update_rate=self.config['server']['update_rate']
            )
            
            if self.server.connect():
                print(f"✓ Connected to {self.config['server']['host']}:"
                      f"{self.config['server']['port']}")
            else:
                print("✗ Server connection failed (running in standalone mode)")
                self.server = None
        except Exception as e:
            print(f"✗ Server initialization error: {e}")
            self.server = None
        
        # Final Status
        print("\n[6/6] Initialization Complete")
        print("=" * 60)
        print(f"GPS:      {'✓' if self.gps else '✗'}")
        print(f"IMU:      {'✓' if self.imu else '✗'}")
        print(f"Motors:   {'✓' if self.drive else '✗'}")
        print(f"Encoders: {'✓' if self.left_encoder and self.right_encoder else '✗'}")
        print(f"Server:   {'✓' if self.server else '✗'}")
        print("=" * 60)
        
        return True
    
    def collect_sensor_data(self) -> Dict[str, Any]:
        """Collect data from all sensors"""
        data = {
            'timestamp': time.time(),
            'gps': None,
            'imu': None,
            'encoders': None,
            'motors': None
        }
        
        # Get GPS data
        if self.gps:
            try:
                gps_data = self.gps.get_gps_data()
                if gps_data:
                    data['gps'] = gps_data
            except Exception as e:
                print(f"GPS read error: {e}")
        
        # Get IMU data
        if self.imu:
            try:
                imu_data = self.imu.get_all_data()
                if imu_data:
                    data['imu'] = imu_data
            except Exception as e:
                print(f"IMU read error: {e}")
        
        # Get encoder data
        if self.left_encoder and self.right_encoder:
            try:
                # Update positions
                left_angle, left_rotations = self.left_encoder.update_position()
                right_angle, right_rotations = self.right_encoder.update_position()
                
                # Calculate speeds
                left_rpm = self.left_encoder.calculate_speed()
                right_rpm = self.right_encoder.calculate_speed()
                
                data['encoders'] = {
                    'left': {
                        'angle': left_angle,
                        'rotations': left_rotations,
                        'total_angle': self.left_encoder.get_total_angle(),
                        'speed_rpm': left_rpm
                    },
                    'right': {
                        'angle': right_angle,
                        'rotations': right_rotations,
                        'total_angle': self.right_encoder.get_total_angle(),
                        'speed_rpm': right_rpm
                    }
                }
            except Exception as e:
                print(f"Encoder read error: {e}")
        
        # Get motor status
        if self.drive:
            try:
                data['motors'] = {
                    'left_speed': self.left_motor.get_speed(),
                    'right_speed': self.right_motor.get_speed()
                }
            except Exception as e:
                print(f"Motor read error: {e}")
        
        return data
    
    def send_to_server(self, data: Dict[str, Any]) -> bool:
        """Send sensor data to AI server for processing"""
        if not self.server or not self.server.connected:
            return False
        
        try:
            return self.server.send_data(data)
        except Exception as e:
            print(f"Server send error: {e}")
            return False
    
    def receive_ai_commands(self) -> Optional[Dict[str, Any]]:
        """Receive control commands from AI server"""
        if not self.server or not self.server.connected:
            return None
        
        try:
            return self.server.receive_data(timeout=0.1)
        except Exception as e:
            print(f"Server receive error: {e}")
            return None
    
    def control_motors(self, command: Dict[str, Any]):
        """Execute motor control commands"""
        if not self.drive:
            return
        
        try:
            cmd_type = command.get('type')
            
            if cmd_type == 'motor':
                action = command.get('action')
                speed = command.get('speed', 50)
                
                if action == 'forward':
                    self.drive.forward(speed)
                    print(f"→ Moving forward at {speed}%")
                elif action == 'backward':
                    self.drive.backward(speed)
                    print(f"← Moving backward at {speed}%")
                elif action == 'left':
                    self.drive.turn_left(speed)
                    print(f"↺ Turning left at {speed}%")
                elif action == 'right':
                    self.drive.turn_right(speed)
                    print(f"↻ Turning right at {speed}%")
                elif action == 'stop':
                    self.drive.stop()
                    print("■ Motors stopped")
                elif action == 'arcade':
                    throttle = command.get('throttle', 0.0)
                    turn = command.get('turn', 0.0)
                    self.drive.arcade_drive(throttle, turn)
                    print(f"⊕ Arcade drive - Throttle: {throttle:.2f}, Turn: {turn:.2f}")
                
            elif cmd_type == 'set_motors':
                left_speed = command.get('left_speed', 0)
                right_speed = command.get('right_speed', 0)
                self.drive.set_motors(left_speed, right_speed)
                print(f"⚙ Motors set - Left: {left_speed}%, Right: {right_speed}%")
            
            elif cmd_type == 'mode':
                mode = command.get('mode')
                if mode == 'autonomous':
                    self.autonomous_mode = True
                    print("🤖 Autonomous mode enabled")
                elif mode == 'manual':
                    self.autonomous_mode = False
                    print("👤 Manual mode enabled")
        
        except Exception as e:
            print(f"Motor control error: {e}")
    
    def run(self):
        """Main robot control loop"""
        self.running = True
        
        print("\n" + "=" * 60)
        print("WASTE-E ROBOT RUNNING")
        print("=" * 60)
        print("Press Ctrl+C to stop\n")
        
        update_interval = 1.0 / self.config['control']['loop_rate']
        display_counter = 0
        display_interval = int(self.config['control']['loop_rate'] / 2)  # Display every 0.5s
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Collect sensor data
                sensor_data = self.collect_sensor_data()
                
                # Send to AI server for processing
                if self.server and self.server.connected:
                    self.send_to_server(sensor_data)
                    
                    # Receive AI commands
                    ai_command = self.receive_ai_commands()
                    if ai_command:
                        print(f"\n[AI Command] {ai_command}")
                        self.control_motors(ai_command)
                
                # Display sensor data periodically
                if display_counter % display_interval == 0:
                    self.display_status(sensor_data)
                
                display_counter += 1
                
                # Maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, update_interval - elapsed)
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        except Exception as e:
            print(f"\nError in main loop: {e}")
        finally:
            self.shutdown()
    
    def display_status(self, data: Dict[str, Any]):
        """Display current robot status"""
        print("\n" + "-" * 60)
        print(f"Time: {time.strftime('%H:%M:%S')}")
        
        # GPS
        if data.get('gps'):
            gps = data['gps']
            print(f"GPS: Lat={gps.get('latitude', 'N/A'):.6f}, "
                  f"Lon={gps.get('longitude', 'N/A'):.6f}, "
                  f"Sats={gps.get('satellites', 0)}")
        
        # IMU
        if data.get('imu') and data['imu'].get('euler'):
            euler = data['imu']['euler']
            print(f"IMU: Heading={euler.get('heading', 0):.1f}°, "
                  f"Roll={euler.get('roll', 0):.1f}°, "
                  f"Pitch={euler.get('pitch', 0):.1f}°")
        
        # Encoders
        if data.get('encoders'):
            enc = data['encoders']
            print(f"Encoders: Left={enc['left']['speed_rpm']:.1f} RPM, "
                  f"Right={enc['right']['speed_rpm']:.1f} RPM")
        
        # Motors
        if data.get('motors'):
            mot = data['motors']
            print(f"Motors: Left={mot['left_speed']}%, Right={mot['right_speed']}%")
        
        print("-" * 60)
    
    def shutdown(self):
        """Clean shutdown of all hardware"""
        print("\n" + "=" * 60)
        print("SHUTTING DOWN WASTE-E ROBOT")
        print("=" * 60)
        
        self.running = False
        
        # Stop motors
        if self.drive:
            print("Stopping motors...")
            self.drive.stop()
            self.drive.cleanup()
        
        # Disconnect GPS
        if self.gps:
            print("Disconnecting GPS...")
            self.gps.disconnect()
        
        # Disconnect IMU
        if self.imu:
            print("Disconnecting IMU...")
            self.imu.disconnect()
        
        # Disconnect encoders
        if self.left_encoder:
            self.left_encoder.disconnect()
        if self.right_encoder:
            self.right_encoder.disconnect()
        
        # Disconnect server
        if self.server:
            print("Disconnecting from server...")
            self.server.disconnect()
        
        # Cleanup GPIO
        try:
            GPIO.cleanup()
            print("GPIO cleaned up")
        except:
            pass
        
        print("=" * 60)
        print("SHUTDOWN COMPLETE")
        print("=" * 60)


def load_config() -> Dict[str, Any]:
    """Load robot configuration"""
    return {
        'gps': {
            'port': '/dev/ttyS1',
            'baudrate': 9600
        },
        'imu': {
            'i2c_bus': 0,
            'address': 0x28
        },
        'motors': {
            'left': {
                'enable': 32,  # Pin 32
                'in1': 29,     # Pin 29
                'in2': 31      # Pin 31
            },
            'right': {
                'enable': 12,  # Pin 12
                'in1': 11,     # Pin 11
                'in2': 13      # Pin 13
            }
        },
        'encoders': {
            'left': {
                'i2c_bus': 0,
                'address': 0x40
            },
            'right': {
                'i2c_bus': 0,
                'address': 0x41  # Different address via A1/A2 pins
            }
        },
        'server': {
            'host': '192.168.1.100',  # Change to your server IP
            'port': 5000,
            'protocol': 'tcp',
            'update_rate': 10.0  # 10 Hz
        },
        'control': {
            'loop_rate': 20.0  # 20 Hz main loop
        }
    }


def main():
    """Main entry point"""
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              WASTE-E AUTONOMOUS ROBOT                    ║")
    print("║         Waste Collection & Navigation System             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\n")
    
    # Load configuration
    config = load_config()
    
    # Create robot instance
    robot = WasteERobot(config)
    
    # Setup signal handlers for clean shutdown
    def signal_handler(sig, frame):
        print("\nSignal received, shutting down...")
        robot.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize hardware
    if not robot.initialize_hardware():
        print("\nFailed to initialize critical hardware. Exiting.")
        sys.exit(1)
    
    # Wait before starting
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Run main control loop
    robot.run()


if __name__ == "__main__":
    main()