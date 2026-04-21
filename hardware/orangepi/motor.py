import OPi.GPIO as GPIO
import time
from typing import Optional


class DCMotor:
    """
    36mm Diameter High Torque Planetary Gear Motor Controller
    Specifications: 12V, 115RPM
    
    Requires motor driver (L298N or similar)
    
    Physical GPIO Connections (using L298N driver):
    Motor A (Right Motor):
    - ENA (Enable/PWM) -> GPIO Pin (e.g., Pin 12, PWM0)
    - IN1 (Direction)  -> GPIO Pin (e.g., Pin 11)
    - IN2 (Direction)  -> GPIO Pin (e.g., Pin 13)
    
    Motor B (Left Motor):
    - ENB (Enable/PWM) -> GPIO Pin (e.g., Pin 32, PWM1)
    - IN3 (Direction)  -> GPIO Pin (e.g., Pin 29)
    - IN4 (Direction)  -> GPIO Pin (e.g., Pin 31)
    
    L298N Power:
    - 12V Power Supply -> L298N 12V input
    - GND -> L298N GND and Orange Pi GND (common ground)
    - Motor outputs -> DC Motor terminals
    
    Note: Remove L298N jumper if using PWM speed control
    """
    
    def __init__(self, enable_pin: int, in1_pin: int, in2_pin: int, 
                 pwm_frequency: int = 1000, board_mode: int = GPIO.BOARD):
        """
        Initialize DC Motor controller
        
        Args:
            enable_pin: GPIO pin for PWM (enable) - controls speed
            in1_pin: GPIO pin for direction control 1
            in2_pin: GPIO pin for direction control 2
            pwm_frequency: PWM frequency in Hz (default: 1000Hz)
            board_mode: GPIO.BOARD (physical pin numbering) or GPIO.BCM (GPIO numbering)
        """
        self.enable_pin = enable_pin
        self.in1_pin = in1_pin
        self.in2_pin = in2_pin
        self.pwm_frequency = pwm_frequency
        self.pwm = None
        self.current_speed = 0
        
        # Setup GPIO
        if GPIO.getmode() is None:
            GPIO.setmode(board_mode)
        GPIO.setwarnings(False)
        
        # Setup pins
        GPIO.setup(self.enable_pin, GPIO.OUT)
        GPIO.setup(self.in1_pin, GPIO.OUT)
        GPIO.setup(self.in2_pin, GPIO.OUT)
        
        # Initialize PWM
        self.pwm = GPIO.PWM(self.enable_pin, self.pwm_frequency)
        self.pwm.start(0)  # Start with 0% duty cycle (stopped)
        
        # Initialize stopped
        self.stop()
    
    def forward(self, speed: int = 100):
        """
        Rotate motor forward
        
        Args:
            speed: Speed percentage (0-100)
        """
        speed = max(0, min(100, speed))  # Clamp between 0-100
        
        GPIO.output(self.in1_pin, GPIO.HIGH)
        GPIO.output(self.in2_pin, GPIO.LOW)
        self.pwm.ChangeDutyCycle(speed)
        self.current_speed = speed
    
    def backward(self, speed: int = 100):
        """
        Rotate motor backward
        
        Args:
            speed: Speed percentage (0-100)
        """
        speed = max(0, min(100, speed))  # Clamp between 0-100
        
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.HIGH)
        self.pwm.ChangeDutyCycle(speed)
        self.current_speed = -speed
    
    def stop(self):
        """Stop the motor"""
        GPIO.output(self.in1_pin, GPIO.LOW)
        GPIO.output(self.in2_pin, GPIO.LOW)
        self.pwm.ChangeDutyCycle(0)
        self.current_speed = 0
    
    def brake(self):
        """Brake the motor (both pins HIGH for hard stop)"""
        GPIO.output(self.in1_pin, GPIO.HIGH)
        GPIO.output(self.in2_pin, GPIO.HIGH)
        self.pwm.ChangeDutyCycle(100)
        self.current_speed = 0
    
    def set_speed(self, speed: int):
        """
        Set motor speed and direction
        
        Args:
            speed: Speed with direction (-100 to 100)
                   Positive = forward, Negative = backward, 0 = stop
        """
        if speed > 0:
            self.forward(speed)
        elif speed < 0:
            self.backward(abs(speed))
        else:
            self.stop()
    
    def get_speed(self) -> int:
        """Get current motor speed (-100 to 100)"""
        return self.current_speed
    
    def cleanup(self):
        """Cleanup GPIO and stop motor"""
        self.stop()
        if self.pwm:
            self.pwm.stop()


class DifferentialDrive:
    """
    Differential drive controller for two DC motors
    Used for robot movement with left and right motors
    """
    
    def __init__(self, left_motor: DCMotor, right_motor: DCMotor):
        """
        Initialize differential drive
        
        Args:
            left_motor: Left motor instance
            right_motor: Right motor instance
        """
        self.left_motor = left_motor
        self.right_motor = right_motor
    
    def forward(self, speed: int = 100):
        """Move forward"""
        self.left_motor.forward(speed)
        self.right_motor.forward(speed)
    
    def backward(self, speed: int = 100):
        """Move backward"""
        self.left_motor.backward(speed)
        self.right_motor.backward(speed)
    
    def turn_left(self, speed: int = 100):
        """Turn left (left motor backward, right motor forward)"""
        self.left_motor.backward(speed)
        self.right_motor.forward(speed)
    
    def turn_right(self, speed: int = 100):
        """Turn right (left motor forward, right motor backward)"""
        self.left_motor.forward(speed)
        self.right_motor.backward(speed)
    
    def pivot_left(self, speed: int = 100):
        """Pivot left (only right motor forward)"""
        self.left_motor.stop()
        self.right_motor.forward(speed)
    
    def pivot_right(self, speed: int = 100):
        """Pivot right (only left motor forward)"""
        self.left_motor.forward(speed)
        self.right_motor.stop()
    
    def stop(self):
        """Stop both motors"""
        self.left_motor.stop()
        self.right_motor.stop()
    
    def brake(self):
        """Brake both motors"""
        self.left_motor.brake()
        self.right_motor.brake()
    
    def set_motors(self, left_speed: int, right_speed: int):
        """
        Set individual motor speeds
        
        Args:
            left_speed: Left motor speed (-100 to 100)
            right_speed: Right motor speed (-100 to 100)
        """
        self.left_motor.set_speed(left_speed)
        self.right_motor.set_speed(right_speed)
    
    def arcade_drive(self, throttle: float, turn: float):
        """
        Arcade-style drive control
        
        Args:
            throttle: Forward/backward speed (-1.0 to 1.0)
            turn: Turn rate (-1.0 to 1.0, left is negative)
        """
        # Clamp inputs
        throttle = max(-1.0, min(1.0, throttle))
        turn = max(-1.0, min(1.0, turn))
        
        # Calculate motor speeds
        left_speed = throttle + turn
        right_speed = throttle - turn
        
        # Normalize if exceeds limits
        max_magnitude = max(abs(left_speed), abs(right_speed))
        if max_magnitude > 1.0:
            left_speed /= max_magnitude
            right_speed /= max_magnitude
        
        # Convert to percentage
        self.left_motor.set_speed(int(left_speed * 100))
        self.right_motor.set_speed(int(right_speed * 100))
    
    def cleanup(self):
        """Cleanup both motors"""
        self.left_motor.cleanup()
        self.right_motor.cleanup()


def main():
    """
    Example usage for 36mm High Torque Planetary Gear Motor (12V, 115RPM)
    
    Hardware Setup:
    1. Connect L298N Motor Driver to Orange Pi:
       Motor A (Right):
       - ENA -> Pin 12 (PWM0, GPIO 110)
       - IN1 -> Pin 11 (GPIO 0)
       - IN2 -> Pin 13 (GPIO 6)
       
       Motor B (Left):
       - ENB -> Pin 32 (PWM1, GPIO 2)
       - IN3 -> Pin 29 (GPIO 7)
       - IN4 -> Pin 31 (GPIO 1)
    
    2. Power connections:
       - 12V power supply -> L298N 12V terminal
       - GND from power supply -> L298N GND
       - Orange Pi GND -> L298N GND (common ground!)
       - Remove 12V enable jumper on L298N if using PWM
    
    3. Motor connections:
       - Right motor -> OUT1 and OUT2
       - Left motor -> OUT3 and OUT4
    """
    
    try:
        # Initialize motors (using BOARD pin numbering)
        print("Initializing motors...")
        right_motor = DCMotor(enable_pin=12, in1_pin=11, in2_pin=13, board_mode=GPIO.BOARD)
        left_motor = DCMotor(enable_pin=32, in1_pin=29, in2_pin=31, board_mode=GPIO.BOARD)
        
        # Create differential drive
        robot = DifferentialDrive(left_motor, right_motor)
        
        print("Testing motor control... (Press Ctrl+C to stop)")
        
        # Test forward
        print("\nMoving forward at 50% speed")
        robot.forward(50)
        time.sleep(2)
        
        # Test backward
        print("Moving backward at 50% speed")
        robot.backward(50)
        time.sleep(2)
        
        # Test turn left
        print("Turning left")
        robot.turn_left(60)
        time.sleep(1.5)
        
        # Test turn right
        print("Turning right")
        robot.turn_right(60)
        time.sleep(1.5)
        
        # Test pivot left
        print("Pivot left")
        robot.pivot_left(50)
        time.sleep(1)
        
        # Test pivot right
        print("Pivot right")
        robot.pivot_right(50)
        time.sleep(1)
        
        # Test speed ramping
        print("\nSpeed ramp test")
        for speed in range(0, 101, 10):
            print(f"Speed: {speed}%")
            robot.forward(speed)
            time.sleep(0.5)
        
        # Stop
        print("\nStopping")
        robot.stop()
        
        # Test arcade drive
        print("\nTesting arcade drive...")
        # Forward with slight right turn
        robot.arcade_drive(throttle=0.7, turn=0.3)
        time.sleep(2)
        
        robot.stop()
        
    except KeyboardInterrupt:
        print("\nStopping motors...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        robot.cleanup()
        GPIO.cleanup()
        print("Motors stopped and GPIO cleaned up")


if __name__ == "__main__":
    main()
