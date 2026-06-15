"""Detect Ozobot-style 3-color bands from camera frames."""

from ozobot_bands.calibration import load_calibration, save_calibration
from ozobot_bands.detector import BandDetector, BandDetectionResult, DetectionParams
from ozobot_bands.colors import OzobotColor, HSVRange, default_hsv_ranges
from ozobot_bands.color_library import (
    Combination,
    ColorLibrary,
    NamedColor,
    load_color_library,
    match_combinations,
    save_color_library,
)

__version__ = "0.1.0"

__all__ = [
    "BandDetector",
    "BandDetectionResult",
    "DetectionParams",
    "OzobotColor",
    "HSVRange",
    "default_hsv_ranges",
    "load_calibration",
    "save_calibration",
    "ColorLibrary",
    "NamedColor",
    "Combination",
    "load_color_library",
    "save_color_library",
    "match_combinations",
]
