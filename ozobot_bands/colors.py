"""Ozobot official color definitions and color classification."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np


# Reserved color name treated as a band/tape separator rather than a tape color.
SEPARATOR_NAME = "black"
UNKNOWN_NAME = "unknown"


class OzobotColor(IntEnum):
    RED = 1
    GREEN = 2
    BLUE = 3
    BLACK = 4
    UNKNOWN = 0


# Official Ozobot RGB values (converted to BGR for OpenCV).
DEFAULT_BGR: Dict[OzobotColor, Tuple[int, int, int]] = {
    OzobotColor.RED: (39, 32, 236),
    OzobotColor.GREEN: (73, 183, 73),
    OzobotColor.BLUE: (198, 131, 17),
    OzobotColor.BLACK: (0, 0, 0),
}

BAND_COLORS = frozenset({OzobotColor.RED, OzobotColor.GREEN, OzobotColor.BLUE})

COLOR_NAMES: Dict[OzobotColor, str] = {
    OzobotColor.RED: "red",
    OzobotColor.GREEN: "green",
    OzobotColor.BLUE: "blue",
    OzobotColor.BLACK: "black",
    OzobotColor.UNKNOWN: "unknown",
}


@dataclass(frozen=True)
class HSVRange:
    """Inclusive HSV range. Hue wraps at 180 in OpenCV."""

    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

    def contains(self, h: int, s: int, v: int) -> bool:
        if s < self.s_min or s > self.s_max or v < self.v_min or v > self.v_max:
            return False
        if self.h_min <= self.h_max:
            return self.h_min <= h <= self.h_max
        # Hue wrap (e.g. red spans 0 and 180).
        return h >= self.h_min or h <= self.h_max


def default_hsv_ranges() -> Dict[OzobotColor, HSVRange]:
    """Conservative default HSV ranges derived from official BGR values."""
    return {
        OzobotColor.RED: HSVRange(170, 10, 100, 255, 80, 255),
        OzobotColor.GREEN: HSVRange(35, 85, 50, 255, 50, 255),
        OzobotColor.BLUE: HSVRange(90, 130, 80, 255, 50, 255),
        OzobotColor.BLACK: HSVRange(0, 180, 0, 255, 0, 50),
    }


def classify_pixel_hsv(
    h: int,
    s: int,
    v: int,
    ranges: Dict[OzobotColor, HSVRange],
) -> OzobotColor:
    """Classify a single HSV pixel against calibrated ranges."""
    matches = [color for color, range_ in ranges.items() if range_.contains(h, s, v)]
    if not matches:
        return OzobotColor.UNKNOWN
    if OzobotColor.BLACK in matches and len(matches) > 1:
        chromatic = [c for c in matches if c != OzobotColor.BLACK]
        if chromatic:
            return chromatic[0]
    return matches[0]


def classify_hsv_columns(
    hsv_strip: np.ndarray,
    ranges: Dict[OzobotColor, HSVRange],
) -> np.ndarray:
    """Classify each column of an HSV strip (height x width) by median HSV."""
    if hsv_strip.size == 0:
        return np.array([], dtype=np.int8)

    width = hsv_strip.shape[1]
    classifications = np.zeros(width, dtype=np.int8)

    for col in range(width):
        column = hsv_strip[:, col, :]
        h = int(np.median(column[:, 0]))
        s = int(np.median(column[:, 1]))
        v = int(np.median(column[:, 2]))
        classifications[col] = classify_pixel_hsv(h, s, v, ranges).value

    return classifications


def extract_label_runs(labels: List[str], min_width: int) -> List[Tuple[str, int, int]]:
    """Contiguous runs of equal labels as (label, start, end), dropping short runs.

    Runs shorter than ``min_width`` are discarded; remaining adjacent runs that
    share a label are then merged. Merging matters because a sub-threshold noise
    blip in the middle of one band would otherwise split it into two same-color
    runs (corrupting a detected color sequence).
    """
    if not labels:
        return []

    segments: List[Tuple[str, int, int]] = []
    current = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((current, start, i))
            current = labels[i]
            start = i
    segments.append((current, start, len(labels)))

    merged: List[Tuple[str, int, int]] = []
    for label, seg_start, seg_end in segments:
        if seg_end - seg_start < min_width:
            continue
        if merged and merged[-1][0] == label:
            prev_label, prev_start, _ = merged[-1]
            merged[-1] = (prev_label, prev_start, seg_end)
        else:
            merged.append((label, seg_start, seg_end))
    return merged


def extract_color_runs(
    classifications: np.ndarray,
    min_width: int,
) -> List[Tuple[OzobotColor, int, int]]:
    """Return contiguous Ozobot color runs as (color, start_col, end_col)."""
    if classifications.size == 0:
        return []
    labels = [str(int(c)) for c in classifications]
    return [
        (OzobotColor(int(label)), start, end)
        for label, start, end in extract_label_runs(labels, min_width)
    ]


def _range_hue_center(range_: HSVRange) -> float:
    """Hue midpoint of a range, accounting for wrap-around (e.g. red)."""
    if range_.h_min <= range_.h_max:
        return (range_.h_min + range_.h_max) / 2.0
    span = (180 - range_.h_min) + range_.h_max
    return (range_.h_min + span / 2.0) % 180.0


def _hue_distance(a: float, b: float) -> float:
    """Circular distance between two hues on the 0..180 OpenCV scale."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def classify_pixel_named(
    h: int,
    s: int,
    v: int,
    ranges: Dict[str, HSVRange],
) -> Optional[str]:
    """Classify an HSV pixel against named ranges, or None if nothing matches.

    When several ranges match, the separator color loses to any tape color, and
    among the rest the range whose center is nearest (hue-weighted) wins. This
    keeps overlapping custom colors from matching arbitrarily.
    """
    matches = [name for name, range_ in ranges.items() if range_.contains(h, s, v)]
    if not matches:
        return None
    pool = [name for name in matches if name != SEPARATOR_NAME] or matches
    if len(pool) == 1:
        return pool[0]

    best_name = pool[0]
    best_dist = float("inf")
    for name in pool:
        range_ = ranges[name]
        ch = _range_hue_center(range_)
        cs = (range_.s_min + range_.s_max) / 2.0
        cv = (range_.v_min + range_.v_max) / 2.0
        dist = _hue_distance(h, ch) * 2.0 + abs(s - cs) / 8.0 + abs(v - cv) / 8.0
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def classify_named_columns(
    hsv_strip: np.ndarray,
    ranges: Dict[str, HSVRange],
) -> List[str]:
    """Classify each column of an HSV strip by median HSV into a color name.

    Columns matching no range are labelled ``UNKNOWN_NAME``.
    """
    if hsv_strip.size == 0:
        return []

    labels: List[str] = []
    for col in range(hsv_strip.shape[1]):
        column = hsv_strip[:, col, :]
        h = int(np.median(column[:, 0]))
        s = int(np.median(column[:, 1]))
        v = int(np.median(column[:, 2]))
        name = classify_pixel_named(h, s, v, ranges)
        labels.append(name if name is not None else UNKNOWN_NAME)
    return labels
