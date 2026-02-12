"""
Check which motor on the leader arm is too far from center for calibration.
"""
from omegaconf import OmegaConf
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from pathlib import Path

# lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower_v1 --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 teleop.id=so101_leader_v1


# Connect directly to the motor bus, bypassing SOLeader's auto-calibration
cfg = OmegaConf.create({
    "port": "/dev/ttyACM0",
    "motors": {
        "shoulder_pan": [1, "sts3215"],
        "shoulder_lift": [2, "sts3215"],
        "elbow_flex": [3, "sts3215"],
        "wrist_flex": [4, "sts3215"],
        "wrist_roll": [5, "sts3215"],
        "gripper": [6, "sts3215"],
    }
})

print("\n=== LEADER ARM POSITION CHECK ===")
print("Connecting to motors...\n")

bus = DynamixelMotorsBus(
    port="/dev/ttyACM0",
    motors=cfg.motors,
)

bus.connect()

print("Reading motor positions...\n")
print(f"{'Motor Name':<20} {'Position':>8} {'Offset':>8} {'Status'}")
print("-" * 60)

problem_motors = []

# Read positions from each motor
for motor_name, (motor_id, model) in cfg.motors.items():
    try:
        # Read present position
        pos = bus.read("Present_Position", motor_id)
        
        # Calculate offset from center (2048 for 12-bit motors)
        offset = pos - 2048
        abs_offset = abs(offset)
        
        # Check if offset would exceed calibration limit (2047)
        if abs_offset > 2047:
            status = "❌ TOO FAR - FIX THIS!"
            problem_motors.append((motor_name, pos, offset))
        elif abs_offset > 1800:
            status = "⚠️  CLOSE TO LIMIT"
        else:
            status = "✓ OK"
        
        print(f"{motor_name:<20} {pos:>8.0f} {offset:>+8.0f} {status}")
    except Exception as e:
        print(f"{motor_name:<20} {'ERROR':>8} - Could not read: {e}")

bus.disconnect()

print("\n" + "=" * 60)
if problem_motors:
    print("\n🔴 PROBLEM MOTORS FOUND:")
    for name, pos, offset in problem_motors:
        direction = "clockwise" if offset > 0 else "counter-clockwise"
        print(f"\n  {name}")
        print(f"    Current position: {pos:.0f}")
        print(f"    Offset from center: {offset:+.0f}")
        print(f"    → Manually rotate this joint {direction} to bring it closer to 2048")
    print("\n💡 TIP: Disable torque, manually adjust the problematic joints,")
    print("   then run this script again to verify before calibrating.")
else:
    print("\n✅ All motors are within calibration range!")
    print("You can proceed with calibration.")