from __future__ import annotations

import time

import gpiod


CONSUMER = "motor_gpio"

CHIP0 = "/dev/gpiochip0"
CHIP1 = "/dev/gpiochip1"

# Right motor driver pins
R_IN1 = (CHIP1, 26)  # GPIO1_D2
R_IN2 = (CHIP1, 27)  # GPIO1_D3
R_ENA = (CHIP1, 15)  # GPIO1_B7 / PWM13_M2

# Left motor driver pins
L_IN1 = (CHIP0, 13)  # GPIO0_B5
L_IN2 = (CHIP0, 14)  # GPIO0_B6
L_ENA = (CHIP1, 7)   # GPIO1_A7 / PWM3_M3

# XOR with these to correct wiring/motor orientation
LEFT_REVERSED = False
RIGHT_REVERSED = True

SUPPORTED_ACTIONS = {"forward", "backward", "left", "right", "stop"}


def _request_line(chip_path: str, offset: int, initial_value: int = 0):
    chip = gpiod.Chip(chip_path)
    settings = gpiod.line_settings.LineSettings(
        direction=gpiod.line.Direction.OUTPUT,
        output_value=(gpiod.line.Value.ACTIVE if initial_value else gpiod.line.Value.INACTIVE),
    )
    request = chip.request_lines({offset: settings}, consumer=CONSUMER)
    return chip, request, offset


def _set_line(request, offset: int, value: bool) -> None:
    request.set_value(offset, gpiod.line.Value.ACTIVE if value else gpiod.line.Value.INACTIVE)


def _set_direction(in1, in2, forward: bool, reversed_side: bool) -> None:
    go_forward = forward ^ reversed_side
    if go_forward:
        _set_line(in1[0], in1[1], True)
        _set_line(in2[0], in2[1], False)
    else:
        _set_line(in1[0], in1[1], False)
        _set_line(in2[0], in2[1], True)


def _stop_wheel(in1, in2, en) -> None:
    _set_line(en[0], en[1], False)
    _set_line(in1[0], in1[1], False)
    _set_line(in2[0], in2[1], False)


def _run_wheel(in1, in2, en, *, forward: bool, reversed_side: bool) -> None:
    _set_direction(in1, in2, forward=forward, reversed_side=reversed_side)
    _set_line(en[0], en[1], True)


class MotorDriver:
    """
    gpiod-based differential drive motor controller.

    GPIO pin mapping (Orange Pi 5 Pro):
      Left:  IN1=GPIO0_B5, IN2=GPIO0_B6, ENA=PWM3_M3
      Right: IN1=GPIO1_D2, IN2=GPIO1_D3, ENA=PWM13_M2

    Supports actions: forward, backward, left, right, stop.
    Use as a context manager or call open()/close() explicitly.
    """

    def __init__(self) -> None:
        self._chips: list = []
        self._lines: dict[str, object] = {}
        self._is_open = False

    def open(self) -> None:
        if self._is_open:
            return

        right_in1_chip, right_in1, right_in1_offset = _request_line(*R_IN1)
        right_in2_chip, right_in2, right_in2_offset = _request_line(*R_IN2)
        right_en_chip, right_en, right_en_offset = _request_line(*R_ENA)

        left_in1_chip, left_in1, left_in1_offset = _request_line(*L_IN1)
        left_in2_chip, left_in2, left_in2_offset = _request_line(*L_IN2)
        left_en_chip, left_en, left_en_offset = _request_line(*L_ENA)

        self._chips = [
            right_in1_chip, right_in2_chip, right_en_chip,
            left_in1_chip, left_in2_chip, left_en_chip,
        ]
        self._lines = {
            "right_in1": (right_in1, right_in1_offset),
            "right_in2": (right_in2, right_in2_offset),
            "right_en": (right_en, right_en_offset),
            "left_in1": (left_in1, left_in1_offset),
            "left_in2": (left_in2, left_in2_offset),
            "left_en": (left_en, left_en_offset),
        }
        self._is_open = True
        self.stop()

    def close(self) -> None:
        if not self._is_open:
            return
        try:
            self.stop()
        finally:
            for request, _offset in self._lines.values():
                request.release()
            for chip in self._chips:
                chip.close()
            self._lines.clear()
            self._chips.clear()
            self._is_open = False

    def __enter__(self) -> MotorDriver:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def apply_action(self, action: str) -> None:
        if not self._is_open:
            raise RuntimeError("motor driver is not open")
        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"unsupported action: {action!r}")

        li1 = self._lines["left_in1"]
        li2 = self._lines["left_in2"]
        len_ = self._lines["left_en"]
        ri1 = self._lines["right_in1"]
        ri2 = self._lines["right_in2"]
        ren = self._lines["right_en"]

        if action == "forward":
            _run_wheel(li1, li2, len_, forward=True, reversed_side=LEFT_REVERSED)
            _run_wheel(ri1, ri2, ren, forward=True, reversed_side=RIGHT_REVERSED)
        elif action == "backward":
            _run_wheel(li1, li2, len_, forward=False, reversed_side=LEFT_REVERSED)
            _run_wheel(ri1, ri2, ren, forward=False, reversed_side=RIGHT_REVERSED)
        elif action == "left":
            _run_wheel(li1, li2, len_, forward=False, reversed_side=LEFT_REVERSED)
            _run_wheel(ri1, ri2, ren, forward=True, reversed_side=RIGHT_REVERSED)
        elif action == "right":
            _run_wheel(li1, li2, len_, forward=True, reversed_side=LEFT_REVERSED)
            _run_wheel(ri1, ri2, ren, forward=False, reversed_side=RIGHT_REVERSED)
        else:
            self.stop()

    def pulse(self, action: str, duration_s: float) -> None:
        if duration_s < 0.0:
            raise ValueError("duration_s must be non-negative")
        self.apply_action(action)
        try:
            time.sleep(duration_s)
        finally:
            self.stop()

    def stop(self) -> None:
        if not self._is_open:
            return
        _stop_wheel(self._lines["left_in1"], self._lines["left_in2"], self._lines["left_en"])
        _stop_wheel(self._lines["right_in1"], self._lines["right_in2"], self._lines["right_en"])


def main() -> None:
    """Manual test — cycles through all actions with 1-second pulses."""
    import sys

    print("MotorDriver test — will move the robot through each action.")
    print("Make sure the robot is on a stand or clear floor.")
    input("Press Enter to start (Ctrl+C to abort)...\n")

    actions = [
        ("forward",  1.0),
        ("stop",     0.5),
        ("backward", 1.0),
        ("stop",     0.5),
        ("left",     0.8),
        ("stop",     0.5),
        ("right",    0.8),
        ("stop",     0.5),
    ]

    try:
        with MotorDriver() as driver:
            for action, duration_s in actions:
                print(f"  {action} for {duration_s:.1f} s")
                if action == "stop":
                    driver.stop()
                    time.sleep(duration_s)
                else:
                    driver.pulse(action, duration_s)
        print("\nTest complete — motors stopped.")
    except KeyboardInterrupt:
        print("\nAborted.")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
