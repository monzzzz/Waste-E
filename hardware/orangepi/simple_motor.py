import OPi.GPIO as GPIO
import time

# CHANGE THIS to your board if needed
GPIO.setboard(GPIO.ORANGEPI_5)

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Right motor pins (BOARD numbering)
ENA = 12
IN1 = 11
IN2 = 13

# Setup pins
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

# --- PWM (4 arguments version) ---
# Try this FIRST
pwm = GPIO.PWM(ENA, 1000, 0, 0)   # pin, frequency, duty, polarity

# If your board crashes here, STOP and tell me.

# Forward
GPIO.output(IN1, GPIO.HIGH)
GPIO.output(IN2, GPIO.LOW)
pwm.ChangeDutyCycle(40)   # 40% speed
time.sleep(2)

# Stop
pwm.ChangeDutyCycle(0)
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.LOW)
time.sleep(1)

# Backward
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.HIGH)
pwm.ChangeDutyCycle(40)
time.sleep(2)

# Stop everything
pwm.ChangeDutyCycle(0)
GPIO.cleanup()

print("Done")