"""Load and save color calibration data."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from vision.colors import (
    DEFAULT_BGR,
    HSVRange,
    OzobotColor,
    classify_pixel_hsv,
    default_hsv_ranges,
)


CALIBRATION_VERSION = 1


def hsv_range_from_samples(
    samples: List[Tuple[int, int, int]],
    hue_padding: int = 8,
    sv_padding: int = 40,
) -> HSVRange:
    """Build an HSV range from sampled (h, s, v) tuples with padding."""
    if not samples:
        raise ValueError("At least one HSV sample is required")

    hs = [s[0] for s in samples]
    ss = [s[1] for s in samples]
    vs = [s[2] for s in samples]

    h_min, h_max = min(hs), max(hs)
    s_min = max(0, min(ss) - sv_padding)
    s_max = min(255, max(ss) + sv_padding)
    v_min = max(0, min(vs) - sv_padding)
    v_max = min(255, max(vs) + sv_padding)

    if h_max - h_min > 90:
        low_hue = [h for h in hs if h < 90]
        high_hue = [h for h in hs if h >= 90]
        if low_hue and high_hue:
            h_min = max(min(low_hue) - hue_padding, 0)
            h_max = min(max(high_hue) + hue_padding, 180)
        else:
            h_min = max(h_min - hue_padding, 0)
            h_max = min(h_max + hue_padding, 180)
    else:
        h_min = max(h_min - hue_padding, 0)
        h_max = min(h_max + hue_padding, 180)

    return HSVRange(h_min, h_max, s_min, s_max, v_min, v_max)


def sample_region_hsv(
    frame_bgr: np.ndarray,
    center: Tuple[int, int],
    radius: int = 25,
) -> List[Tuple[int, int, int]]:
    """Sample HSV values from a circular region around center."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    cx, cy = center
    h, w = frame_bgr.shape[:2]
    samples: List[Tuple[int, int, int]] = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h:
                pixel = hsv[y, x]
                samples.append((int(pixel[0]), int(pixel[1]), int(pixel[2])))

    return samples


def infer_color_at_pixel(
    frame_bgr: np.ndarray,
    x: int,
    y: int,
    ranges: Dict[OzobotColor, HSVRange],
) -> OzobotColor:
    """Guess Ozobot color at a pixel using calibrated ranges, then official BGR refs."""
    h, w = frame_bgr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return OzobotColor.UNKNOWN

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    pixel_hsv = hsv[y, x]
    classified = classify_pixel_hsv(
        int(pixel_hsv[0]), int(pixel_hsv[1]), int(pixel_hsv[2]), ranges
    )
    if classified != OzobotColor.UNKNOWN:
        return classified

    bgr = frame_bgr[y, x]
    best_color = OzobotColor.BLACK
    best_dist = float("inf")
    for color, ref_bgr in DEFAULT_BGR.items():
        dist = sum(int((int(bgr[i]) - ref_bgr[i]) ** 2) for i in range(3))
        if dist < best_dist:
            best_dist = dist
            best_color = color
    return best_color


def sample_color_at_click(
    frame_bgr: np.ndarray,
    x: int,
    y: int,
    ranges: Dict[OzobotColor, HSVRange],
    forced_color: Optional[OzobotColor] = None,
    radius: int = 25,
    hue_padding: int = 8,
    sv_padding: int = 40,
) -> Tuple[OzobotColor, HSVRange]:
    """Sample HSV around a click and return the color slot and updated range."""
    color = forced_color or infer_color_at_pixel(frame_bgr, x, y, ranges)
    if color == OzobotColor.UNKNOWN:
        raise ValueError(f"Could not infer color at ({x}, {y})")

    samples = sample_region_hsv(frame_bgr, (x, y), radius)
    new_range = hsv_range_from_samples(
        samples, hue_padding=hue_padding, sv_padding=sv_padding
    )
    return color, new_range


def ranges_to_dict(ranges: Dict[OzobotColor, HSVRange]) -> dict:
    return {
        COLOR_NAME_MAP[color]: {
            "h_min": r.h_min,
            "h_max": r.h_max,
            "s_min": r.s_min,
            "s_max": r.s_max,
            "v_min": r.v_min,
            "v_max": r.v_max,
        }
        for color, r in ranges.items()
    }


COLOR_NAME_MAP = {
    OzobotColor.RED: "red",
    OzobotColor.GREEN: "green",
    OzobotColor.BLUE: "blue",
    OzobotColor.BLACK: "black",
}

NAME_TO_COLOR = {v: k for k, v in COLOR_NAME_MAP.items()}


def ranges_from_dict(data: dict) -> Dict[OzobotColor, HSVRange]:
    ranges: Dict[OzobotColor, HSVRange] = {}
    for name, params in data.items():
        color = NAME_TO_COLOR[name]
        ranges[color] = HSVRange(
            params["h_min"],
            params["h_max"],
            params["s_min"],
            params["s_max"],
            params["v_min"],
            params["v_max"],
        )
    return ranges


def save_calibration(
    path: Path,
    ranges: Dict[OzobotColor, HSVRange],
    detection_params: Optional[dict] = None,
    sampled_colors: Optional[Set[OzobotColor]] = None,
    sample_points: Optional[Dict[OzobotColor, Tuple[int, int]]] = None,
) -> None:
    payload = {
        "version": CALIBRATION_VERSION,
        "hsv_ranges": ranges_to_dict(ranges),
        "detection": detection_params or {},
    }
    if sampled_colors is not None:
        payload["sampled_colors"] = sorted(
            COLOR_NAME_MAP[c] for c in sampled_colors
        )
    if sample_points is not None:
        payload["sample_points"] = {
            COLOR_NAME_MAP[color]: [int(x), int(y)]
            for color, (x, y) in sample_points.items()
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path) -> Tuple[
    Dict[OzobotColor, HSVRange],
    dict,
    Set[OzobotColor],
    Dict[OzobotColor, Tuple[int, int]],
]:
    from vision.color_library import load_color_library, ozobot_ranges_from_library

    library = load_color_library(path)
    ranges = ozobot_ranges_from_library(library)
    detection = library.detection

    sampled: Set[OzobotColor] = set()
    for name, enum in NAME_TO_COLOR.items():
        if name in library.colors:
            sampled.add(enum)

    points: Dict[OzobotColor, Tuple[int, int]] = {}
    for name, enum in NAME_TO_COLOR.items():
        entry = library.colors.get(name)
        if entry and entry.sample_point is not None:
            points[enum] = entry.sample_point

    if not sampled and ranges:
        sampled = set(ranges.keys())

    return ranges, detection, sampled, points


def get_ranges(calibration_path: Optional[Path] = None) -> Dict[OzobotColor, HSVRange]:
    if calibration_path and calibration_path.exists():
        from vision.color_library import load_color_library, ozobot_ranges_from_library

        library = load_color_library(calibration_path)
        mapped = ozobot_ranges_from_library(library)
        if mapped:
            return mapped
    return default_hsv_ranges()
