"""Motion control for the Unitree G1 and LeKiwi robots.

    from motion import G1Motion       # Unitree G1 (needs unitree_sdk2py)
    from motion import LeKiwiMotion   # LeKiwi (needs lerobot)
"""

try:
    from .g1_motion import G1Motion, GESTURES
except ImportError:
    G1Motion = None  # type: ignore[assignment,misc]
    GESTURES = []

try:
    from .lekiwi_motion import LeKiwiMotion
except ImportError:
    LeKiwiMotion = None  # type: ignore[assignment,misc]

__all__ = ["G1Motion", "GESTURES", "LeKiwiMotion"]
