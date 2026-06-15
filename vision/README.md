# vision — colour sheet → mission → French speech

A self-contained OpenCV pipeline that reads a **white sheet** held to a camera,
detects the **ordered sequence** of colour-tape strips on it, turns that sequence
into a **mission** (one action per colour), and **speaks the correction actions
in French**.

Frame input works from either a **local webcam / RealSense** device or a **ROS 2
image topic**, selected with the same flags everywhere.

## Layout

```
vision/
  colors.py        # HSV colour definitions + per-column classification
  calibration.py   # load/save calibration JSON
  color_library.py # named colours + combinations (JSON v3)
  detector.py      # BandDetector: position/angle search reads the colour order
  frame_source.py  # webcam (--camera) AND ROS 2 (--ros-topic) backends
  missions.py      # ordered colour -> French action mapping
  tts.py           # French TTS with backend fallback (spd-say/espeak/pyttsx3/gtts/print)
  pipeline.py      # MissionPipeline: frame -> detection -> mission -> speech
  run.py           # CLI: read a sheet and speak its mission
  controller.py    # CLI: read a sheet and DRIVE the Unitree G1 (see ../motion)
  actions.fr.json  # default colour -> French action phrasing
```

## Frame source

```bash
# Local USB / RealSense colour device (RealSense colour stream is usually 4)
python -m vision.run --camera 4

# Shared via the realsense2_camera ROS 2 node
python -m vision.run --ros-topic /camera/color/image_raw
```

`--ros-topic` takes precedence over `--camera`; `--ros-timeout` bounds the wait.

## Read a sheet and speak the mission

```bash
# Decode the bundled sample image (no camera needed)
python -m vision.run --image image.png

# Live camera; auto-speaks each new mission once it is stable
python -m vision.run --camera 4

# One shot, write the mission to JSON
python -m vision.run --once --save-json mission.json
```

Output for `image.png`:

```
Mission: yellow-blue-orange
  1. yellow   -> Tourne de moins quarante-cinq degrés
  2. blue     -> Tourne de quarante-cinq degrés
  3. orange   -> Salue de la main
```

Useful flags: `--steps` (speak each action separately instead of one sentence),
`--no-tts` (print only), `--tts-backend spd-say|espeak-ng|espeak|pyttsx3|gtts|print`,
`--no-preview`.

## Colour → action mapping

Defaults (`vision/missions.py`, mirrored in `actions.fr.json`) are phrased to
match the robot motions in [`../motion`](../motion):

| Colour  | French action                         | Robot motion       |
|---------|---------------------------------------|--------------------|
| green   | Avance d'un mètre                     | walk forward 1 m   |
| blue    | Tourne de quarante-cinq degrés        | turn +45°          |
| yellow  | Tourne de moins quarante-cinq degrés  | turn −45°          |
| orange  | Salue de la main                      | wave               |
| red     | Arrête-toi                            | (stop)             |

Override per-deployment by passing `--actions actions.fr.json`, or by adding an
`"actions"` block to `calibration.json` so phrasing lives next to the colours.
`black` is a separator (no action); unmapped colours are read aloud as
"couleur inconnue".

## Drive the robot

See [`vision/controller.py`](controller.py) and the project
[README](../README.md): the controller decodes the sheet here, then runs the
matching `motion` command on the Unitree G1.

```bash
# Dry run (no robot, just prints/speaks the plan)
python -m vision.controller --image image.png

# Live and actually moving (needs the robot + system SDK)
python -m vision.controller --camera 4 --execute --iface eth0
```

## Tests

```bash
pytest            # configured to collect vision/tests
```

## Use in code

```python
from pathlib import Path
import cv2
from vision import MissionPipeline, FrenchTTS

pipeline = MissionPipeline(calibration_path=Path("calibration.json"))
frame = cv2.imread("image.png")
obs = pipeline.process(frame)
if obs.detected:
    print(obs.mission.colors)         # ['yellow', 'blue', 'orange']
    pipeline.narrate(obs.mission)     # speaks the French mission
```
