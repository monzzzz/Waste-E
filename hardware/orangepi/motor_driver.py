from __future__ import annotations

import threading
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

_PWM_HZ = 200  # software PWM carrier frequency


def _request_line(chip_path: str, offset: int, initial_value: int = 0):
    chip = gpiod.Chip(chip_path)
    settings = gpiod.LineSettings(
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


class MotorDriver:
    """
    gpiod-based differential drive motor controller with software PWM speed control.

    GPIO pin mapping (Orange Pi 5 Pro):
      Left:  IN1=GPIO0_B5, IN2=GPIO0_B6, ENA=PWM3_M3
      Right: IN1=GPIO1_D2, IN2=GPIO1_D3, ENA=PWM13_M2

    Supports actions: forward, backward, left, right, stop.
    Call set_speed(0.0–1.0) to adjust motor speed.
    Use as a context manager or call open()/close() explicitly.
    """

    def __init__(self) -> None:
        self._chips: list = []
        self._lines: dict[str, object] = {}
        self._is_open = False
        self._lock = threading.Lock()
        self._speed: float = 1.0       # 0.0 – 1.0
        self._left_en: bool = False    # should left ENA be driven?
        self._right_en: bool = False   # should right ENA be driven?

    def open(self) -> None:
        if self._is_open:
            return

        right_in1_chip, right_in1, right_in1_offset = _request_line(*R_IN1)
        right_in2_chip, right_in2, right_in2_offset = _request_line(*R_IN2)
        right_en_chip,  right_en,  right_en_offset  = _request_line(*R_ENA)

        left_in1_chip, left_in1, left_in1_offset = _request_line(*L_IN1)
        left_in2_chip, left_in2, left_in2_offset = _request_line(*L_IN2)
        left_en_chip,  left_en,  left_en_offset  = _request_line(*L_ENA)

        self._chips = [
            right_in1_chip, right_in2_chip, right_en_chip,
            left_in1_chip,  left_in2_chip,  left_en_chip,
        ]
        self._lines = {
            "right_in1": (right_in1, right_in1_offset),
            "right_in2": (right_in2, right_in2_offset),
            "right_en":  (right_en,  right_en_offset),
            "left_in1":  (left_in1,  left_in1_offset),
            "left_in2":  (left_in2,  left_in2_offset),
            "left_en":   (left_en,   left_en_offset),
        }
        self._is_open = True
        threading.Thread(target=self._pwm_loop, daemon=True).start()
        self.stop()

    def close(self) -> None:
        if not self._is_open:
            return
        self._is_open = False
        try:
            self.stop()
            time.sleep(0.02)  # let PWM thread see is_open=False
        finally:
            for request, _offset in self._lines.values():
                request.release()
            for chip in self._chips:
                chip.close()
            self._lines.clear()
            self._chips.clear()

    def __enter__(self) -> MotorDriver:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── PWM thread ────────────────────────────────────────────────────────────

    def _pwm_loop(self) -> None:
        period = 1.0 / _PWM_HZ
        len_ = self._lines["left_en"]
        ren  = self._lines["right_en"]

        while self._is_open:
            t0 = time.monotonic()

            with self._lock:
                speed  = self._speed
                l_want = self._left_en
                r_want = self._right_en

            if not l_want and not r_want:
                # Stopped — keep ENAs low
                _set_line(len_[0], len_[1], False)
                _set_line(ren[0],  ren[1],  False)
                rem = period - (time.monotonic() - t0)
                if rem > 0:
                    time.sleep(rem)

            elif speed >= 0.99:
                # Full speed — keep ENAs high
                _set_line(len_[0], len_[1], l_want)
                _set_line(ren[0],  ren[1],  r_want)
                rem = period - (time.monotonic() - t0)
                if rem > 0:
                    time.sleep(rem)

            else:
                # PWM: on for duty cycle, off for remainder
                on_t  = period * speed
                off_t = period * (1.0 - speed)
                _set_line(len_[0], len_[1], l_want)
                _set_line(ren[0],  ren[1],  r_want)
                time.sleep(on_t)
                _set_line(len_[0], len_[1], False)
                _set_line(ren[0],  ren[1],  False)
                time.sleep(off_t)

    # ── public API ────────────────────────────────────────────────────────────

    def set_speed(self, speed: float) -> None:
        """Set motor speed. 0.0 = stopped, 1.0 = full speed."""
        with self._lock:
            self._speed = max(0.0, min(1.0, float(speed)))

    def get_speed(self) -> float:
        with self._lock:
            return self._speed

    def apply_action(self, action: str) -> None:
        if not self._is_open:
            raise RuntimeError("motor driver is not open")
        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"unsupported action: {action!r}")

        li1 = self._lines["left_in1"]
        li2 = self._lines["left_in2"]
        ri1 = self._lines["right_in1"]
        ri2 = self._lines["right_in2"]

        if action == "stop":
            # Clear direction pins and disable ENAs via PWM thread
            _set_line(li1[0], li1[1], False)
            _set_line(li2[0], li2[1], False)
            _set_line(ri1[0], ri1[1], False)
            _set_line(ri2[0], ri2[1], False)
            with self._lock:
                self._left_en  = False
                self._right_en = False
        else:
            if action == "forward":
                _set_direction(li1, li2, forward=True,  reversed_side=LEFT_REVERSED)
                _set_direction(ri1, ri2, forward=True,  reversed_side=RIGHT_REVERSED)
                l, r = True, True
            elif action == "backward":
                _set_direction(li1, li2, forward=False, reversed_side=LEFT_REVERSED)
                _set_direction(ri1, ri2, forward=False, reversed_side=RIGHT_REVERSED)
                l, r = True, True
            elif action == "left":
                _set_direction(li1, li2, forward=False, reversed_side=LEFT_REVERSED)
                _set_direction(ri1, ri2, forward=True,  reversed_side=RIGHT_REVERSED)
                l, r = True, True
            elif action == "right":
                _set_direction(li1, li2, forward=True,  reversed_side=LEFT_REVERSED)
                _set_direction(ri1, ri2, forward=False, reversed_side=RIGHT_REVERSED)
                l, r = True, True
            with self._lock:
                self._left_en  = l
                self._right_en = r

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
        self.apply_action("stop")


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
            driver.set_speed(0.6)  # run at 60% speed for the test
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
