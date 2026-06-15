"""Tests for Ozobot band detection."""

import numpy as np
import cv2

from ozobot_bands.colors import OzobotColor, default_hsv_ranges, classify_hsv_columns
from ozobot_bands.detector import BandDetector, DetectionParams


def _make_band_frame(colors_bgr: list, segment_width: int = 40, height: int = 120) -> np.ndarray:
    """Build a synthetic horizontal color band."""
    width = segment_width * len(colors_bgr)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for i, bgr in enumerate(colors_bgr):
        x0 = i * segment_width
        frame[:, x0:x0 + segment_width] = bgr
    return frame


def test_detect_three_color_band():
    ranges = default_hsv_ranges()
    params = DetectionParams(
        min_segment_width_px=5,
        scan_strip_height_ratio=0.5,
        scan_line_length_px=280,
    )
    detector = BandDetector(hsv_ranges=ranges, params=params)

    frame = _make_band_frame([
        (0, 0, 0),        # black
        (39, 32, 236),    # red
        (0, 0, 0),        # black
        (73, 183, 73),    # green
        (0, 0, 0),        # black
        (198, 131, 17),   # blue
        (0, 0, 0),        # black
    ])

    result = detector.detect(frame)
    assert result.band_detected
    assert len(set(result.colors_sequence)) >= 3
    assert result.confidence == 1.0


def test_no_band_on_uniform_frame():
    ranges = default_hsv_ranges()
    detector = BandDetector(
        hsv_ranges=ranges,
        params=DetectionParams(
            angle_search_enabled=False,
            position_search_enabled=False,
        ),
    )

    frame = np.full((100, 200, 3), (200, 200, 200), dtype=np.uint8)
    result = detector.detect(frame)
    assert not result.band_detected


def test_classify_synthetic_red_column():
    ranges = default_hsv_ranges()
    red_bgr = np.full((10, 20, 3), (39, 32, 236), dtype=np.uint8)
    hsv = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2HSV)
    classifications = classify_hsv_columns(hsv, ranges)
    assert OzobotColor(classifications[10]) == OzobotColor.RED
