# ozobots — colour sheet → Unitree G1 motion

Show a **white sheet with coloured tape strips** to a camera; the robot reads the
**colour order**, announces each correction action **in French**, and **performs
the matching motion** on a Unitree G1.

Two packages, plus a controller that bridges them:

| Package | Runs under | Role |
|---------|-----------|------|
| [`vision/`](vision/README.md) | project `.venv` (OpenCV) | camera → ordered colour mission → French speech |
| [`motion/`](motion/README.md) | system `/usr/bin/python3` (Unitree SDK) | closed-loop `turn` / `walk` / `wave` on the G1 |
| `vision/controller.py` | project `.venv` | decode the sheet, then run the motion for each colour |

> **Interpreter split.** OpenCV is installed in the project `.venv`; the Unitree
> SDK (`unitree_sdk2py`, `cyclonedds`) is installed only for the system
> `/usr/bin/python3`. The controller therefore runs under `.venv` for vision and
> calls `motion` as a subprocess (`/usr/bin/python3 -m motion …`), so the two
> never have to share one interpreter.

## Colour → action

In the order the colours appear on the sheet:

| Colour  | French (spoken)                       | G1 motion        |
|---------|---------------------------------------|------------------|
| green   | Avance d'un mètre                     | walk forward 1 m |
| blue    | Tourne de quarante-cinq degrés        | turn +45° (left) |
| yellow  | Tourne de moins quarante-cinq degrés  | turn −45° (right)|
| orange  | Salue de la main                      | wave             |

(`black` separates strips; other calibrated colours can be added in
`calibration.json` and `vision/missions.py`.)

## Quick start

```bash
pip install -e .          # installs vision + its deps into the .venv

# 1. See the decoding only (no robot): reads the bundled sample sheet
python -m vision.run --image image.png

# 2. Dry-run the full controller (decode + plan the motions, no movement)
python -m vision.controller --image image.png

# 3. Live, really drive the robot (camera + G1 on eth0)
python -m vision.controller --camera 4 --execute --iface eth0
```

`--execute` is required for the robot to move; without it the controller prints
and speaks the plan but sends no motion commands. `--camera N` uses a local
device (RealSense colour stream is usually `4`); `--ros-topic /camera/color/image_raw`
shares the camera through the RealSense ROS 2 node instead.

## ROS 2 bringup

[`launch/ozobot_bringup.launch.py`](launch/ozobot_bringup.launch.py) starts the
RealSense camera node (publishing `/camera/color/image_raw`) and the G1 ROS 2
bridge together:

```bash
ros2 launch launch/ozobot_bringup.launch.py
```

## Calibration

`calibration.json` holds the HSV ranges for each colour (and an optional
`"actions"` block to override the French phrasing). Re-sample colours under your
runtime lighting for reliable classification.

## Tests

```bash
pytest        # vision/tests (motion needs the robot SDK, so it is not collected)
```

## Layout

```
vision/      colour detection + mission mapping + French TTS + controller
motion/      Unitree G1 closed-loop motion API + CLI
launch/      ROS 2 bringup (RealSense camera + G1 bridge)
calibration.json   calibrated colours, combinations, detection params, actions
image.png    sample colour sheet
```
