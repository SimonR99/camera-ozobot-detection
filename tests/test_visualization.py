"""Tests for calibration visualization."""

import numpy as np
import cv2

from ozobot_bands.color_library import ColorLibrary, NamedColor
from ozobot_bands.colors import HSVRange, OzobotColor, default_hsv_ranges
from ozobot_bands.synthetic import generate_tag_image
from ozobot_bands.visualization import draw_named_color_highlights, hsv_range_mask


def test_hsv_range_mask_on_synthetic_red():
    frame, _ = generate_tag_image(seed=1, color_width_range=(20, 20), gap_max_px=0)
    ranges = default_hsv_ranges()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = hsv_range_mask(hsv, ranges[OzobotColor.RED])
    assert np.any(mask > 0)


def test_highlight_named_library():
    frame, _ = generate_tag_image(seed=2)
    library = ColorLibrary()
    library.upsert(
        NamedColor(
            name="red",
            hsv_range=default_hsv_ranges()[OzobotColor.RED],
            reference_bgr=(39, 32, 236),
        )
    )
    highlighted = draw_named_color_highlights(frame, library, alpha=0.5)
    empty = draw_named_color_highlights(frame, ColorLibrary(), alpha=0.5)
    assert not np.array_equal(highlighted, frame)
    assert np.array_equal(empty, frame)
