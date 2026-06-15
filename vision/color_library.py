"""Named color library for calibration (v2 JSON format)."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import cv2
import numpy as np

from vision.colors import HSVRange, OzobotColor
from vision.calibration import hsv_range_from_samples, NAME_TO_COLOR


LIBRARY_VERSION = 3


@dataclass
class NamedColor:
    name: str
    hsv_range: HSVRange
    sample_point: Optional[Tuple[int, int]] = None
    reference_bgr: Optional[Tuple[int, int, int]] = None


@dataclass
class Combination:
    """A named group of colors that, seen together, forms a recognized block.

    Colors are treated as an unordered set: the combination matches whenever the
    same set of colors appears, regardless of their left-to-right order.
    """

    name: str
    colors: List[str]

    def color_set(self) -> FrozenSet[str]:
        return frozenset(self.colors)


@dataclass
class ColorLibrary:
    colors: Dict[str, NamedColor] = field(default_factory=dict)
    combinations: Dict[str, Combination] = field(default_factory=dict)
    detection: dict = field(default_factory=dict)

    def get(self, name: str) -> Optional[NamedColor]:
        return self.colors.get(name)

    def upsert(self, entry: NamedColor) -> None:
        self.colors[entry.name] = entry

    def remove(self, name: str) -> bool:
        if name in self.colors:
            del self.colors[name]
            return True
        return False

    def names(self) -> List[str]:
        return sorted(self.colors.keys())

    def duplicate(self, source_name: str, new_name: str) -> NamedColor:
        source = self.colors[source_name]
        entry = NamedColor(
            name=new_name,
            hsv_range=source.hsv_range,
            sample_point=source.sample_point,
            reference_bgr=source.reference_bgr,
        )
        self.upsert(entry)
        return entry

    def upsert_combination(self, combination: Combination) -> None:
        self.combinations[combination.name] = combination

    def remove_combination(self, name: str) -> bool:
        if name in self.combinations:
            del self.combinations[name]
            return True
        return False

    def combination_names(self) -> List[str]:
        return sorted(self.combinations.keys())


def match_combinations(
    detected_colors: Set[str],
    combinations: Dict[str, Combination],
) -> List[str]:
    """Names of combinations whose color set equals the detected color set.

    Unordered-set semantics: a combination matches when exactly its colors are
    present (no more, no fewer), regardless of order or repetition.
    """
    detected = frozenset(detected_colors)
    return sorted(
        name
        for name, combination in combinations.items()
        if combination.color_set() == detected
    )


def hsv_range_to_dict(range_: HSVRange) -> dict:
    return {
        "h_min": range_.h_min,
        "h_max": range_.h_max,
        "s_min": range_.s_min,
        "s_max": range_.s_max,
        "v_min": range_.v_min,
        "v_max": range_.v_max,
    }


def hsv_range_from_dict(data: dict) -> HSVRange:
    return HSVRange(
        data["h_min"],
        data["h_max"],
        data["s_min"],
        data["s_max"],
        data["v_min"],
        data["v_max"],
    )


def named_color_to_dict(entry: NamedColor) -> dict:
    payload = {"hsv_range": hsv_range_to_dict(entry.hsv_range)}
    if entry.sample_point is not None:
        payload["sample_point"] = [int(entry.sample_point[0]), int(entry.sample_point[1])]
    if entry.reference_bgr is not None:
        payload["reference_bgr"] = list(entry.reference_bgr)
    return payload


def named_color_from_dict(name: str, data: dict) -> NamedColor:
    sample_point = None
    if "sample_point" in data:
        sample_point = (int(data["sample_point"][0]), int(data["sample_point"][1]))
    reference_bgr = None
    if "reference_bgr" in data:
        bgr = data["reference_bgr"]
        reference_bgr = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return NamedColor(
        name=name,
        hsv_range=hsv_range_from_dict(data["hsv_range"]),
        sample_point=sample_point,
        reference_bgr=reference_bgr,
    )


def save_color_library(path: Path, library: ColorLibrary) -> None:
    payload = {
        "version": LIBRARY_VERSION,
        "colors": {name: named_color_to_dict(c) for name, c in library.colors.items()},
        "combinations": {
            name: list(combo.colors) for name, combo in library.combinations.items()
        },
        "detection": library.detection,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _combinations_from_payload(payload: dict) -> Dict[str, Combination]:
    return {
        name: Combination(name=name, colors=list(colors))
        for name, colors in payload.get("combinations", {}).items()
    }


def load_color_library(path: Path) -> ColorLibrary:
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    payload = json.loads(path.read_text())
    version = payload.get("version", 1)

    if version >= 2:
        colors = {
            name: named_color_from_dict(name, data)
            for name, data in payload.get("colors", {}).items()
        }
        return ColorLibrary(
            colors=colors,
            combinations=_combinations_from_payload(payload),
            detection=payload.get("detection", {}),
        )

    # Migrate v1 format.
    library = ColorLibrary(detection=payload.get("detection", {}))
    for name, params in payload.get("hsv_ranges", {}).items():
        sample_point = None
        if name in payload.get("sample_points", {}):
            xy = payload["sample_points"][name]
            sample_point = (int(xy[0]), int(xy[1]))
        library.upsert(
            NamedColor(
                name=name,
                hsv_range=hsv_range_from_dict(params),
                sample_point=sample_point,
            )
        )
    return library


def ozobot_ranges_from_library(library: ColorLibrary) -> Dict[OzobotColor, HSVRange]:
    """Map standard Ozobot names in the library to enum keys for the detector."""
    ranges: Dict[OzobotColor, HSVRange] = {}
    for name, enum in NAME_TO_COLOR.items():
        if name in library.colors:
            ranges[enum] = library.colors[name].hsv_range
    return ranges


def parse_color_string(text: str) -> Tuple[int, int, int]:
    """Parse pasted color as BGR. Supports #RRGGBB, rgb(r,g,b), r,g,b."""
    text = text.strip()
    if not text:
        raise ValueError("Empty color string")

    if text.startswith("#") and len(text) in (7, 4):
        hex_body = text[1:]
        if len(hex_body) == 3:
            hex_body = "".join(c * 2 for c in hex_body)
        r = int(hex_body[0:2], 16)
        g = int(hex_body[2:4], 16)
        b = int(hex_body[4:6], 16)
        return (b, g, r)

    rgb_match = re.match(
        r"^\s*rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$",
        text,
        re.IGNORECASE,
    )
    if rgb_match:
        r, g, b = (int(rgb_match.group(i)) for i in range(1, 4))
        return (b, g, r)

    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        r, g, b = (int(p) for p in parts)
        return (b, g, r)

    raise ValueError(f"Cannot parse color: {text!r}")


def hsv_range_from_bgr(
    bgr: Tuple[int, int, int],
    hue_padding: int = 8,
    sv_padding: int = 40,
) -> HSVRange:
    pixel = np.array([[list(bgr)]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    samples = [(int(hsv[0]), int(hsv[1]), int(hsv[2]))]
    return hsv_range_from_samples(samples, hue_padding=hue_padding, sv_padding=sv_padding)


def preview_bgr_for_entry(entry: NamedColor) -> Tuple[int, int, int]:
    if entry.reference_bgr is not None:
        return entry.reference_bgr
    if entry.sample_point is not None:
        return (200, 200, 200)
    # Derive from HSV range midpoint — approximate via V channel.
    v = (entry.hsv_range.v_min + entry.hsv_range.v_max) // 2
    return (v, v, v)
