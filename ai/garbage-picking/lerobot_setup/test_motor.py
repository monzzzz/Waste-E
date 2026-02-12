"""
Keyboard teleop for a LeRobot follower arm.

Keys:
  1/2/3/4/5 : select joint
  W/S       : increase/decrease selected joint
  J/K       : close/open gripper (or gripper joint if present)
  SPACE     : torque OFF (panic)
  ENTER     : torque ON
  ESC       : quit
"""

import time
from omegaconf import OmegaConf
from pynput import keyboard
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from pathlib import Path

# ========= EDIT THESE TWO LINES =========
# 1) Import your robot class (path may differ in your repo)
# from lerobot.robots.so101.follower import SO101Follower

# 2) Create the robot (port/config args may differ)

cfg = OmegaConf.create({
    "port": "/dev/ttyACM0",
    "id": 0,
    "baudrate": 115200,
    "calibration_dir": Path("/home/monzzz/.cache/huggingface/lerobot/calibration/robots/so_follower"),
    "use_degrees": True,
    "cameras": {},
    "max_relative_target": None,  # Add this - limits how far motors can move in one step (None = no limit)
    "disable_torque_on_disconnect": True  # Add this - disables torque when disconnecting
})
robot = SOFollower(cfg)
# robot = None  # <-- replace with your robot instance
# =======================================

STEP_TICKS = 15       # small step; reduce if too jumpy
LOOP_HZ = 30

selected_joint = 0
torque_on = True
quit_flag = False

# Will hold joint positions as a dict {name: value}
q = None
joint_names = []


def set_torque(enabled: bool):
    global torque_on
    torque_on = enabled
    try:
        if enabled:
            robot.bus.enable_torque()
        else:
            robot.bus.disable_torque()
        print(f"[torque] {enabled}")
    except Exception as e:
        print(f"[torque] Could not set torque: {e}")


def get_positions():
    obs = robot.get_observation()
    # Filter out camera keys, keep only motor positions
    return {k: v for k, v in obs.items() if k.endswith(".pos")}


def command_positions(q_cmd):
    robot.send_action(q_cmd)


def on_press(key):
    global selected_joint, quit_flag, q

    try:
        k = key.char.lower()
    except AttributeError:
        k = None

    # Quit
    if key == keyboard.Key.esc:
        quit_flag = True
        return False

    # Panic / torque
    if key == keyboard.Key.space:
        set_torque(False)
        return
    if key == keyboard.Key.enter:
        set_torque(True)
        return

    # Select joint with number keys
    if k in ("1", "2", "3", "4", "5", "6"):  # Add "6" for gripper
        idx = int(k) - 1
        if 0 <= idx < len(joint_names):
            selected_joint = idx
            print(f"[select] {selected_joint+1}: {joint_names[selected_joint]}")
        return

    if q is None:
        return

    # Move selected joint
    if k == "w":
        name = joint_names[selected_joint]
        q[name] += STEP_TICKS
        command_positions(q)
    elif k == "s":
        name = joint_names[selected_joint]
        q[name] -= STEP_TICKS
        command_positions(q)

    # Gripper (if present) - use correct key name with .pos suffix
    if k == "j":  # close
        if "gripper.pos" in q:
            q["gripper.pos"] += STEP_TICKS
            command_positions(q)
    elif k == "k":  # open
        if "gripper.pos" in q:
            q["gripper.pos"] -= STEP_TICKS
            command_positions(q)


def main():
    global q, joint_names

    if robot is None:
        raise RuntimeError("Edit the script: set `robot = ...` to your follower instance.")

    # Connect
    for fn_name in ("connect", "open"):
        fn = getattr(robot, fn_name, None)
        if callable(fn):
            fn()
            break

    set_torque(True)

    # Read initial positions
    q0 = get_positions()

    # Normalize to dict format
    if isinstance(q0, dict):
        q = dict(q0)
        joint_names = list(q.keys())  # Include all joints
    else:
        raise RuntimeError("Joint positions are not a dict. Adapt script to your format.")

    print("\nControls:")
    print("  1/2/3/4/5/6 select joint | W/S move | J/K gripper | ENTER torque on | SPACE torque off | ESC quit")
    print("Joints:", joint_names)
    print("Selected:", joint_names[selected_joint])

    # Keyboard listener
    with keyboard.Listener(on_press=on_press) as listener:
        while not quit_flag:
            time.sleep(1 / LOOP_HZ)

    # Cleanup
    set_torque(False)
    for fn_name in ("disconnect", "close"):
        fn = getattr(robot, fn_name, None)
        if callable(fn):
            fn()
            break
    print("Bye.")


if __name__ == "__main__":
    main()
