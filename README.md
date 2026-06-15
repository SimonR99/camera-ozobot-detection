# Ozobot Band Detection

OpenCV pipeline to detect **color-tape blocks** from a camera feed. You calibrate any colors you like, group them into named **combinations**, and the detector flags when tape on the floor forms one of those combinations — at any position or rotation in the frame.

The classic Ozobot block (red + green + blue) is just one combination you can define. Detection is **unordered**: a combination matches whenever the same *set* of colors is read, regardless of left-to-right order.

## Concepts

- **Color** — a named HSV range you calibrate from the camera (`scripts/calibrate.py`). Any number of arbitrary colors are supported, not just red/green/blue.
- **Combination** — a named group of colors (2 or more; 3 is the typical Ozobot block) managed with `scripts/combinations.py`. The set of colors is what matters, not their order.
- The reserved color name **`black`** is treated as a separator, not a tape color.

If no combinations are defined yet, detection falls back to the generic rule "≥ 3 distinct calibrated colors form a block."

## Install

```bash
pip install -e .
```

## Camera source

Every script reads frames from one of two backends, selected by the same flags:

- `--camera N` — local OpenCV device index (default `0`). On an **Intel RealSense**
  the colour stream is usually index **4** (`0`=depth, `2`=infrared), and the device
  reads as *busy* if the `realsense2_camera` ROS node already holds it.
- `--ros-topic TOPIC` — subscribe to a ROS 2 `sensor_msgs/Image` topic instead of a
  local device, e.g. `--ros-topic /camera/color/image_raw`. Use this to share the
  camera with the robot stack (the RealSense node owns the device and publishes the
  image). Requires a sourced ROS 2 workspace; `--ros-timeout` (default 5s) bounds the
  wait for a frame. `--ros-topic` takes precedence over `--camera`.

```bash
# Local USB/RealSense colour device
python scripts/demo.py --camera 4

# Shared via the realsense2_camera ROS node
python scripts/demo.py --ros-topic /camera/color/image_raw
```

### Bring up the camera + robot bridge

`launch/ozobot_bringup.launch.py` starts the RealSense node and the
`g1_ros2_bridge` together (the bridge runs with its own camera disabled so the two
don't fight over the device). After sourcing ROS 2 and the `unitree_g1_ros2`
workspace:

```bash
ros2 launch launch/ozobot_bringup.launch.py
```

It publishes `/camera/color/image_raw`; then run any script with
`--ros-topic /camera/color/image_raw`. Useful overrides:
`pointcloud_enable:=false`, `g1_interface:=eth0`.

## Quick start

### 1. Calibrate (recommended)

Three-step workflow on the live camera feed:

1. **Click** a color anywhere on screen
2. **Adjust** the range with `+` / `-` (yellow preview shows what will match)
3. **Validate** with `v`, then **type a name** and press Enter

```bash
python scripts/calibrate.py --camera 0 --output calibration.json
```

| Key | Action |
|-----|--------|
| **Click** | Pick color → adjust mode |
| `+` / `-` | Narrow / widen range (adjust mode) |
| `v` | Validate → name entry |
| **Enter** | Save color with typed name to JSON |
| `p` | Paste mode: `#RRGGBB`, `r,g,b`, or existing color name to copy |
| `d` | Delete a saved color by name |
| `Esc` | Cancel current step |
| `q` | Quit |

Saved colors are highlighted in real time. Paste an existing name in paste mode to duplicate and tweak it under a new name.

**Tips to avoid false positives/negatives:**
- Sample each color under the same lighting you will use at runtime.
- Use official Ozobot markers (or equivalent RGB: red `236/32/39`, green `73/183/73`, blue `17/131/198`).
- If colors bleed together, increase `min_segment_width_px` with `]`.
- If detection misses faint colors, increase padding with `+`.
- If non-band colors trigger detection, decrease padding with `-` or enable black separators with `t`.

### 2. Group colors into combinations

Open the interactive manager/viewer, pick saved colors to build a combination, name it, and save. The live banner shows which combination the camera currently matches.

```bash
python scripts/combinations.py --calibration calibration.json
```

| Key | Action |
|-----|--------|
| **Click** swatch / `1`–`9` | Toggle that saved color in/out of the current selection |
| `n` | Name + save the current selection as a combination (needs ≥ 2 colors) |
| `c` | Clear the current selection |
| `d` | Delete a saved combination by name |
| `Esc` | Cancel name/delete entry (or clear selection) |
| `q` | Quit |

Combinations are stored in the same JSON file as the colors. Use `--no-camera` to manage combinations without a live feed.

### 3. Run detection

```bash
python scripts/demo.py --calibration calibration.json
```

Detection is restricted to a **region of interest** — by default the **bottom half**
of the frame (height) and the **middle half** (width), i.e. a `¼ ignore / ½ detect /
¼ ignore` split horizontally. A 3-color block only counts when it is read inside that
zone; the zone is drawn on screen, brightens when a block is found, and a banner shows
the detected colors. Override or disable the region:

```bash
# custom region (frame fractions)
python scripts/demo.py --region-x-min 0.2 --region-x-max 0.8 --region-y-min 0.6 --region-y-max 1.0
# whole frame
python scripts/demo.py --full-frame
```

### 4. Identify band colors and save JSON

```bash
python scripts/identify_band.py --calibration calibration.json --output band_detection.json
```

Press `s` in the preview to save the detected color sequence. **Click** on the band to move the scan line to that row. Or auto-save when stable (with combinations defined, auto-save fires only on a real combination match):

```bash
python scripts/identify_band.py --auto --stable-frames 8 --output band_detection.json
```

One-shot (no window):

```bash
python scripts/identify_band.py --once --output band_detection.json
```

Example `band_detection.json`:

```json
{
  "timestamp": "2026-06-15T12:00:00+00:00",
  "band_detected": true,
  "combination_detected": true,
  "matched_combinations": ["ozobot"],
  "colors": ["red", "green", "blue"],
  "color_code": "red-green-blue",
  "unique_colors": ["blue", "green", "red"],
  "confidence": 1.0,
  "segments": [...],
  "band_segments": [...]
}
```

### 5. Use in code

```python
from pathlib import Path
import cv2
from ozobot_bands import BandDetector

detector = BandDetector(calibration_path=Path("calibration.json"))

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

result = detector.detect(frame)

if result.combination_detected:
    print(f"Matched: {result.matched_combinations}")  # e.g. ['ozobot']
elif result.band_detected:
    print(f"Colors found (no combination): {result.colors_sequence}")
```

## API

### `BandDetector.detect(frame_bgr)` → `BandDetectionResult`

| Field | Description |
|-------|-------------|
| `combination_detected` | `True` when the colors read match a defined combination |
| `matched_combinations` | Names of the matched combinations (set-equality on colors) |
| `band_detected` | `True` when enough distinct tape colors form a block (generic rule) |
| `colors_sequence` | Ordered list of detected colors on the scan line |
| `confidence` | 0.0–1.0 |
| `color_runs` | Raw segments as `(color, start_col, end_col)` |
| `roi` | Scan region `(x, y, width, height)` |

## How it works

1. Search positions and angles in the frame for the richest reading (works at any rotation/offset).
2. Extract a thin strip along the scan line, convert to HSV, and classify each column against **every** calibrated color.
3. Merge columns into contiguous color runs, dropping short noise segments.
4. Take the distinct tape colors read and match them, as an unordered set, against the defined combinations. With none defined, fall back to "≥ 3 distinct colors."

## Project layout

```
ozobot_bands/
  colors.py        # Color definitions and HSV classification
  color_library.py # Named colors + combinations (JSON v3) and set matcher
  calibration.py   # Save/load calibration JSON
  detector.py      # BandDetector pipeline (combination matching)
scripts/
  calibrate.py     # Interactive color picker/modifier
  combinations.py  # Interactive combination manager + live viewer
  demo.py          # Live webcam demo
  identify_band.py # Camera → identify colors / combinations → save JSON
  check_synthetic.py # Run detection on generated synthetic tags
```

## Tests

```bash
pytest
```
