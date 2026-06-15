"""Closed-loop motion API for the Unitree G1.

The G1 loco SDK only exposes velocity control (`SetVelocity`) and one-shot
gestures, with no turn-to-angle or walk-to-distance primitive. Open-loop timed
motion is inaccurate (a commanded 90 deg turn came out ~59 deg). This module
wraps the SDK with closed-loop control on the robot's own feedback:

    * turn(degrees) -- uses IMU yaw from rt/lowstate (imu_state.rpy[2])
    * walk(metres)  -- uses odometry position from rt/odommodestate
    * wave(gesture) -- uses G1ArmActionClient (NOT loco WaveHand, which is a
                       no-op on the G1)

Each motion ramps the command down near the target and stops within tolerance.

Requires the Unitree SDK + cyclonedds, which on this machine are only installed
for the system interpreter -- run with /usr/bin/python3, not the project .venv.

Example:
    from motion import G1Motion

    with G1Motion("eth0") as g1:
        g1.turn(90)        # +90 deg = left/CCW
        g1.walk(1.0)       # forward 1 m
        g1.wave()          # high wave
"""

from __future__ import annotations

import math
import time

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# --- turn tuning (rad / rad-s) ---
TURN_OMEGA = 0.5        # cruise yaw rate
TURN_TOL_DEG = 3.0      # stop within this many deg of target
TURN_SLOW_DEG = 20.0    # ease off the rate within this many deg of target
TURN_MIN_OMEGA = 0.15   # floor so it keeps moving near the target
TURN_TIMEOUT_S = 20.0

# --- walk tuning (m / m-s) ---
WALK_SPEED = 0.3        # cruise forward speed
WALK_TOL_M = 0.03       # stop within this distance of target
WALK_SLOW_M = 0.25      # ease off within this distance of target
WALK_MIN_SPEED = 0.1
WALK_TIMEOUT_S = 30.0

GESTURES = sorted(action_map)


class G1Motion:
    """High-level, closed-loop motion control for the Unitree G1.

    Args:
        iface: network interface connected to the robot (default "eth0").
        auto_start: if True, enter walk/balance mode (FSM 500) on connect so
            turn()/walk() work immediately.
    """

    def __init__(self, iface: str = "eth0", auto_start: bool = True):
        ChannelFactoryInitialize(0, iface)

        self._yaw: float | None = None
        self._pos: tuple[float, float] | None = None

        self._lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_sub.Init(self._on_lowstate, 10)
        self._odom_sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
        self._odom_sub.Init(self._on_odom, 10)

        self._loco = LocoClient()
        self._loco.SetTimeout(10.0)
        self._loco.Init()

        self._arm = G1ArmActionClient()
        self._arm.SetTimeout(10.0)
        self._arm.Init()

        self._started = False
        if auto_start:
            self.start()

    # ── connection / lifecycle ───────────────────────────────────────────
    def _on_lowstate(self, msg: LowState_):
        self._yaw = msg.imu_state.rpy[2]

    def _on_odom(self, msg: SportModeState_):
        self._pos = (msg.position[0], msg.position[1])

    def start(self):
        """Open the locomotion pipeline (FSM 500 -> walk/balance mode)."""
        self._loco.Start()
        time.sleep(1.0)
        self._started = True

    def _ensure_started(self):
        if not self._started:
            self.start()

    def _wait_yaw(self) -> float:
        return self._wait(lambda: self._yaw, "rt/lowstate")

    def _wait_pos(self) -> tuple[float, float]:
        return self._wait(lambda: self._pos, "rt/odommodestate")

    @staticmethod
    def _wait(getter, topic, timeout=5.0):
        t0 = time.time()
        while getter() is None:
            if time.time() - t0 > timeout:
                raise RuntimeError(f"No {topic} received - is the robot up on this interface?")
            time.sleep(0.05)
        return getter()

    # ── motions ──────────────────────────────────────────────────────────
    def turn(self, degrees: float, omega: float = TURN_OMEGA) -> float:
        """Turn in place by `degrees` (signed: + = left/CCW, - = right/CW).

        Closed-loop on IMU yaw. Returns the measured rotation in degrees.
        """
        self._ensure_started()
        target = math.radians(degrees)
        direction = math.copysign(1.0, target)
        yaw_prev = self._wait_yaw()
        accumulated = 0.0

        t0 = time.time()
        while True:
            remaining = target - accumulated
            if abs(remaining) <= math.radians(TURN_TOL_DEG):
                break
            if time.time() - t0 > TURN_TIMEOUT_S:
                break
            frac = min(1.0, abs(math.degrees(remaining)) / TURN_SLOW_DEG)
            rate = max(TURN_MIN_OMEGA, omega * frac) * direction
            self._loco.SetVelocity(0.0, 0.0, rate, 0.5)
            time.sleep(0.1)
            yaw_now = self._wait_yaw()
            accumulated += math.atan2(math.sin(yaw_now - yaw_prev),
                                      math.cos(yaw_now - yaw_prev))
            yaw_prev = yaw_now

        self.stop()
        return math.degrees(accumulated)

    def walk(self, metres: float, speed: float = WALK_SPEED) -> float:
        """Walk straight by `metres` (signed: + = forward, - = backward).

        Closed-loop on odometry. Returns the measured signed distance in metres.
        """
        self._ensure_started()
        direction = math.copysign(1.0, metres)
        target = abs(metres)
        x0, y0 = self._wait_pos()

        t0 = time.time()
        travelled = 0.0
        while True:
            x, y = self._wait_pos()
            travelled = math.hypot(x - x0, y - y0)
            remaining = target - travelled
            if remaining <= WALK_TOL_M:
                break
            if time.time() - t0 > WALK_TIMEOUT_S:
                break
            frac = min(1.0, remaining / WALK_SLOW_M)
            vx = max(WALK_MIN_SPEED, speed * frac) * direction
            self._loco.SetVelocity(vx, 0.0, 0.0, 0.5)
            time.sleep(0.1)

        self.stop()
        return travelled * direction

    def wave(self, gesture: str = "high wave", release: bool = True,
             hold_s: float = 3.0) -> int:
        """Perform an arm gesture (default 'high wave'). See GESTURES for names.

        Uses G1ArmActionClient (the loco WaveHand is a no-op on the G1). Returns
        the ExecuteAction return code (0 = accepted). If `release`, returns the
        arms to the normal pose afterward.
        """
        if gesture not in action_map:
            raise ValueError(f"Unknown gesture {gesture!r}. Options: {', '.join(GESTURES)}")
        code = self._arm.ExecuteAction(action_map[gesture])
        time.sleep(hold_s)
        if release:
            self._arm.ExecuteAction(action_map["release arm"])
            time.sleep(2.0)
        return code

    def stop(self):
        """Stop all locomotion (zero velocity)."""
        self._loco.StopMove()
        time.sleep(0.3)

    # ── context manager ──────────────────────────────────────────────────
    def __enter__(self) -> "G1Motion":
        return self

    def __exit__(self, *exc):
        try:
            self.stop()
        except Exception:
            pass
        return False
