"""Tests for color library and visualization."""

import numpy as np
import pytest

from ozobot_bands.colors import HSVRange, OzobotColor, default_hsv_ranges
from ozobot_bands.color_library import (
    ColorLibrary,
    NamedColor,
    load_color_library,
    parse_color_string,
    save_color_library,
)
from ozobot_bands.synthetic import generate_tag_image
from ozobot_bands.visualization import draw_named_color_highlights, hsv_range_mask


def test_parse_color_hex():
    assert parse_color_string("#FF0000") == (0, 0, 255)
    assert parse_color_string("255,0,0") == (0, 0, 255)


def test_save_load_library(tmp_path):
    path = tmp_path / "cal.json"
    library = ColorLibrary()
    library.upsert(
        NamedColor(
            name="red",
            hsv_range=HSVRange(0, 10, 100, 255, 80, 255),
            sample_point=(10, 20),
            reference_bgr=(39, 32, 236),
        )
    )
    save_color_library(path, library)
    loaded = load_color_library(path)
    assert "red" in loaded.colors
    assert loaded.colors["red"].sample_point == (10, 20)


def test_highlight_named_colors():
    frame, _ = generate_tag_image(seed=3)
    library = ColorLibrary()
    library.upsert(
        NamedColor(
            name="red",
            hsv_range=default_hsv_ranges()[OzobotColor.RED],
            reference_bgr=(39, 32, 236),
        )
    )
    out = draw_named_color_highlights(frame, library, alpha=0.5)
    assert not np.array_equal(out, frame)
