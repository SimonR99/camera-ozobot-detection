# Ozobot Band Detection

OpenCV pipeline to detect **Ozobot-style color bands** (red, green, blue) from a camera feed. Returns a `band_detected` flag when all three chromatic colors appear as contiguous segments on a scan line.

## Install

```bash
pip install -e .
```

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

### 2. Run detection

```bash
python scripts/demo.py --calibration calibration.json
```

### 3. Identify band colors and save JSON

```bash
python scripts/identify_band.py --calibration calibration.json --output band_detection.json
```

Press `s` in the preview to save the detected color sequence. **Click** on the band to move the scan line to that row. Or auto-save when stable:

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
  "colors": ["red", "green", "blue"],
  "color_code": "red-green-blue",
  "unique_colors": ["blue", "green", "red"],
  "confidence": 1.0,
  "segments": [...],
  "band_segments": [...]
}
```

### 4. Use in code

```python
from pathlib import Path
import cv2
from ozobot_bands import BandDetector

detector = BandDetector(calibration_path=Path("calibration.json"))

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

result = detector.detect(frame)

if result.band_detected:
    print(f"Band found: {result.colors_sequence}")  # e.g. ['red', 'green', 'blue']
    print(f"Confidence: {result.confidence}")
```

## API

### `BandDetector.detect(frame_bgr)` → `BandDetectionResult`

| Field | Description |
|-------|-------------|
| `band_detected` | `True` when ≥3 distinct band colors (R/G/B) are found |
| `colors_sequence` | Ordered list of detected colors on the scan line |
| `confidence` | 0.0–1.0 (1.0 when all three primaries are present) |
| `color_runs` | Raw segments as `(color, start_col, end_col)` |
| `roi` | Scan region `(x, y, width, height)` |

## How it works

1. Extract a horizontal strip from the frame (centered vertically, configurable).
2. Convert to HSV and classify each column using calibrated color ranges.
3. Merge columns into contiguous color runs, filtering short noise segments.
4. Set `band_detected` when at least three distinct Ozobot colors appear as runs.

## Project layout

```
ozobot_bands/
  colors.py       # Ozobot color definitions and HSV classification
  calibration.py  # Save/load calibration JSON
  detector.py     # BandDetector pipeline
scripts/
  calibrate.py    # Interactive calibration UI
  demo.py         # Live webcam demo
  identify_band.py # Camera → identify colors → save JSON
```
