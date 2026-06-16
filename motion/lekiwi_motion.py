"""Open-loop motion API for the LeKiwi mobile manipulator.

The LeKiwi is a holonomic (omni-wheel) base + 6-DOF arm. The Feetech bus
exposes no IMU or odometry, so all base movements use timed open-loop control.

    from motion import LeKiwiMotion

    with LeKiwiMotion() as kiwi:
        kiwi.walk(0.3)      # forward ≈0.3 m
        kiwi.walk(-0.3)     # backward
        kiwi.turn(45)       # +45 deg = left/CCW
        kiwi.turn(-45)      # right/CW
        kiwi.strafe(0.2)    # strafe right ≈0.2 m
        kiwi.wave()         # arm wave gesture

Requires lerobot ≥0.5 (pip install lerobot). Robot must be on /dev/ttyACM0.
"""

from __future__ import annotations

import math
import time

from lerobot.robots.lekiwi.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

# --- base movement tuning ---
WALK_SPEED = 0.15     # m/s  forward/backward cruise speed
STRAFE_SPEED = 0.15   # m/s  side-step cruise speed
TURN_SPEED = 45.0     # deg/s rotation cruise speed

# --- arm raw positions (Feetech 0-4095 units, no calibration required) ---
# Derived from the gamma test-rig (gamma/test2.py) and applied with normalize=False.
# shoulder_pan = 2048 is centred; 1848 / 2248 is ≈ ±10 steps left / right.
_ARM_WAVE_RAISE = {
    "arm_shoulder_pan":  2048,
    "arm_shoulder_lift": 1100,
    "arm_elbow_flex":    2940,
    "arm_wrist_flex":    2048,
    "arm_wrist_roll":    2048,
    "arm_gripper":       2048,
}
_ARM_WAVE_LEFT  = {**_ARM_WAVE_RAISE, "arm_shoulder_pan": 1848}
_ARM_WAVE_RIGHT = {**_ARM_WAVE_RAISE, "arm_shoulder_pan": 2248}
_ARM_REST = {
    "arm_shoulder_pan":  2048,
    "arm_shoulder_lift": 2048,
    "arm_elbow_flex":    2048,
    "arm_wrist_flex":    2048,
    "arm_wrist_roll":    2048,
    "arm_gripper":       2048,
}


class LeKiwiMotion:
    """Open-loop motion control for the LeKiwi robot.

    Args:
        port: serial port for the Feetech motor bus (default /dev/ttyACM0).
    """

    def __init__(self, port: str = "/dev/ttyACM0"):
        config = LeKiwiConfig(port=port, cameras={})
        self._robot = LeKiwi(config)
        self._robot.connect(calibrate=False)

    # ── base motions ─────────────────────────────────────────────────────────

    def walk(self, metres: float, speed: float = WALK_SPEED) -> None:
        """Walk forward (+) or backward (-) approximately `metres`.

        Open-loop timed — accuracy depends on floor surface and battery charge.
        """
        if metres == 0:
            return
        direction = math.copysign(1.0, metres)
        self._drive(y=direction * abs(speed), duration=abs(metres) / abs(speed))

    def strafe(self, metres: float, speed: float = STRAFE_SPEED) -> None:
        """Strafe right (+) or left (-) approximately `metres`.

        Holonomic side-step using the omni wheels.
        """
        if metres == 0:
            return
        direction = math.copysign(1.0, metres)
        self._drive(x=direction * abs(speed), duration=abs(metres) / abs(speed))

    def turn(self, degrees: float, omega: float = TURN_SPEED) -> None:
        """Turn in place by `degrees` (+ = left/CCW, - = right/CW).

        Open-loop timed.
        """
        if degrees == 0:
            return
        direction = math.copysign(1.0, degrees)
        self._drive(theta=direction * abs(omega), duration=abs(degrees) / abs(omega))

    def stop(self) -> None:
        """Stop all wheel motion."""
        self._robot.stop_base()

    # ── arm gestures ─────────────────────────────────────────────────────────

    def wave(self, reps: int = 2) -> None:
        """Wave with the arm (shoulder pan oscillation, mirroring G1 high-wave).

        Args:
            reps: number of left-right oscillations (default 2).
        """
        self._arm_set(_ARM_WAVE_RAISE)
        time.sleep(0.5)
        for _ in range(reps):
            self._arm_set(_ARM_WAVE_LEFT)
            time.sleep(0.25)
            self._arm_set(_ARM_WAVE_RIGHT)
            time.sleep(0.25)
        self._arm_set(_ARM_WAVE_RAISE)
        time.sleep(0.2)
        self._arm_set(_ARM_REST)
        time.sleep(0.5)

    # ── internals ────────────────────────────────────────────────────────────

    def _drive(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0,
               duration: float = 0.0) -> None:
        """Convert body-frame command to wheel speeds, drive, then stop."""
        wheel_cmds = self._robot._body_to_wheel_raw(x, y, theta)
        # Goal_Velocity is not in the normalised-data table, so normalize flag is moot;
        # the raw integer ticks from _body_to_wheel_raw are sent directly.
        self._robot.bus.sync_write("Goal_Velocity", wheel_cmds)
        time.sleep(duration)
        self.stop()

    def _arm_set(self, positions: dict[str, int]) -> None:
        """Write raw 0-4095 Feetech positions without calibration-based normalisation."""
        self._robot.bus.sync_write("Goal_Position", positions, normalize=False)

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "LeKiwiMotion":
        return self

    def __exit__(self, *exc):
        try:
            self.stop()
            self._robot.disconnect()
        except Exception:
            pass
        return False
