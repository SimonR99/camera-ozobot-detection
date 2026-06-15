# motion — Unitree G1 closed-loop motion API

A small Python API for driving the Unitree G1: **turn by an angle**, **walk a
distance**, and **wave / gesture**. The G1 SDK only offers raw velocity control
and one-shot gestures, so timing-based moves are inaccurate (a commanded 90° turn
came out ~59°). This package closes the loop on the robot's own feedback:

| Method | Feedback | Verified on hardware |
|---|---|---|
| `turn(degrees)` | IMU yaw (`rt/lowstate`) | 90° → 87.3° |
| `walk(metres)`  | odometry (`rt/odommodestate`) | 0.5 m → 0.47 m |
| `wave(gesture)` | `G1ArmActionClient` | high wave |

## Requirements

The Unitree SDK (`unitree_sdk2py`) and `cyclonedds` are installed for the **system**
interpreter only — run everything with `/usr/bin/python3`, **not** the project
`.venv`. The robot must be reachable on the network interface (default `eth0`,
robot controller at `192.168.123.161`).

## Library use

```python
from motion import G1Motion, GESTURES

with G1Motion("eth0") as g1:      # connects + enters walk mode (FSM 500)
    g1.turn(90)                   # +90° = left/CCW, returns measured degrees
    g1.walk(1.0)                  # forward 1 m, returns measured metres
    g1.turn(-90)                  # right
    g1.walk(-0.5)                 # backward
    g1.wave("high wave")          # any name in GESTURES; auto-releases after
```

`turn()` and `walk()` return the **measured** amount. Signs: `+` = left / forward,
`-` = right / backward. Optional `omega=` (rad/s) and `speed=` (m/s) tune the cruise
rate. Gestures: `high wave`, `face wave`, `shake hand`, `high five`, `hug`, `clap`,
`heart`, `hands up`, `reject`, … (`GESTURES` lists them all).

## CLI

```bash
/usr/bin/python3 -m motion turn 90
/usr/bin/python3 -m motion walk 1.0
/usr/bin/python3 -m motion walk -0.5 --speed 0.2
/usr/bin/python3 -m motion wave "face wave"
/usr/bin/python3 -m motion wave list
/usr/bin/python3 -m motion stop
```

Add `--iface <name>` to change interface and `--yes` to skip the safety prompt.

## Notes / tuning

- Odometry drifts over long ranges — fine for a few metres per call.
- `walk()` measures straight-line displacement, so a curved path under-reports.
- Tighten accuracy via the `*_TOL_*` / `*_SLOW_*` constants at the top of
  `g1_motion.py` (trade-off: a little overshoot).
- Gestures use `G1ArmActionClient`, **not** loco `WaveHand` (the latter is a
  no-op on the G1 — accepted but nothing moves).
```
