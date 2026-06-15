"""Closed-loop motion control for the Unitree G1.

    from motion import G1Motion
    with G1Motion("eth0") as g1:
        g1.turn(90)
        g1.walk(1.0)
        g1.wave()
"""

from .g1_motion import G1Motion, GESTURES

__all__ = ["G1Motion", "GESTURES"]
