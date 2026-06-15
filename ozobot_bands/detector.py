"""OpenCV pipeline for detecting Ozobot-style 3-color bands."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from ozobot_bands.calibration import NAME_TO_COLOR, get_ranges
from ozobot_bands.colors import (
    COLOR_NAMES,
    SEPARATOR_NAME,
    UNKNOWN_NAME,
    HSVRange,
    OzobotColor,
    classify_named_columns,
    extract_label_runs,
)
from ozobot_bands.color_library import (
    Combination,
    load_color_library,
    match_combinations,
    ozobot_ranges_from_library,
)


@dataclass
class DetectionParams:
    """Tunable detection thresholds (overridable via calibration file)."""

    min_segment_width_px: int = 8
    scan_strip_height_ratio: float = 0.15
    min_band_colors: int = 3
    require_black_separators: bool = False
    roi_y_center_ratio: float = 0.5
    roi_width_ratio: float = 1.0
    # Detection region of interest as frame-fraction bounds. A band is only
    # detected when its scan center falls inside this rectangle. Defaults to the
    # whole frame (no restriction).
    detect_x_min_ratio: float = 0.0
    detect_x_max_ratio: float = 1.0
    detect_y_min_ratio: float = 0.0
    detect_y_max_ratio: float = 1.0
    scan_line_length_px: int = 0
    scan_line_length_ratio: float = 0.22
    position_search_enabled: bool = True
    position_search_step_px: int = 0
    max_position_candidates: int = 24
    angle_search_enabled: bool = True
    angle_search_step_deg: float = 3.0
    angle_refine_step_deg: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "DetectionParams":
        return cls(
            min_segment_width_px=data.get("min_segment_width_px", 8),
            scan_strip_height_ratio=data.get("scan_strip_height_ratio", 0.15),
            min_band_colors=data.get("min_band_colors", 3),
            require_black_separators=data.get("require_black_separators", False),
            roi_y_center_ratio=data.get("roi_y_center_ratio", 0.5),
            roi_width_ratio=data.get("roi_width_ratio", 1.0),
            detect_x_min_ratio=data.get("detect_x_min_ratio", 0.0),
            detect_x_max_ratio=data.get("detect_x_max_ratio", 1.0),
            detect_y_min_ratio=data.get("detect_y_min_ratio", 0.0),
            detect_y_max_ratio=data.get("detect_y_max_ratio", 1.0),
            scan_line_length_px=data.get("scan_line_length_px", 0),
            scan_line_length_ratio=data.get("scan_line_length_ratio", 0.22),
            position_search_enabled=data.get("position_search_enabled", True),
            position_search_step_px=data.get("position_search_step_px", 0),
            max_position_candidates=data.get("max_position_candidates", 24),
            angle_search_enabled=data.get("angle_search_enabled", True),
            angle_search_step_deg=data.get("angle_search_step_deg", 3.0),
            angle_refine_step_deg=data.get("angle_refine_step_deg", 1.0),
        )


@dataclass
class BandDetectionResult:
    """Result of band detection on a single frame."""

    band_detected: bool
    colors_sequence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    color_runs: List[Tuple[str, int, int]] = field(default_factory=list)
    scan_line_y: int = 0
    roi: Tuple[int, int, int, int] = (0, 0, 0, 0)
    scan_angle_deg: float = 0.0
    scan_line: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0))
    scan_center: Tuple[int, int] = (0, 0)
    combination_detected: bool = False
    matched_combinations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "band_detected": self.band_detected,
            "colors_sequence": self.colors_sequence,
            "confidence": self.confidence,
            "color_runs": self.color_runs,
            "scan_line_y": self.scan_line_y,
            "roi": self.roi,
            "scan_angle_deg": self.scan_angle_deg,
            "scan_line": self.scan_line,
            "scan_center": self.scan_center,
            "combination_detected": self.combination_detected,
            "matched_combinations": self.matched_combinations,
        }


class BandDetector:
    """Detect Ozobot-style bands (3 chromatic colors) from camera frames."""

    def __init__(
        self,
        calibration_path: Optional[Path] = None,
        params: Optional[DetectionParams] = None,
        hsv_ranges: Optional[Dict[OzobotColor, HSVRange]] = None,
        color_ranges: Optional[Dict[str, HSVRange]] = None,
        combinations: Optional[Dict[str, Combination]] = None,
    ):
        self.calibration_path = calibration_path
        detection_data: dict = {}
        # `hsv_ranges` (enum-keyed) is kept only for per-pixel color inference on
        # clicks; `color_ranges` (name-keyed) drives the actual detection and may
        # include arbitrary user-calibrated colors, not just red/green/blue.
        self.combinations: Dict[str, Combination] = combinations or {}

        if calibration_path and calibration_path.exists():
            library = load_color_library(calibration_path)
            self.color_ranges = {
                name: entry.hsv_range for name, entry in library.colors.items()
            }
            self.hsv_ranges = ozobot_ranges_from_library(library)
            self.combinations = library.combinations
            detection_data = library.detection
        elif color_ranges is not None:
            self.color_ranges = dict(color_ranges)
            self.hsv_ranges = {
                NAME_TO_COLOR[name]: r
                for name, r in color_ranges.items()
                if name in NAME_TO_COLOR
            }
        elif hsv_ranges is not None:
            self.hsv_ranges = hsv_ranges
            self.color_ranges = {COLOR_NAMES[c]: r for c, r in hsv_ranges.items()}
        else:
            self.hsv_ranges = get_ranges()
            self.color_ranges = {COLOR_NAMES[c]: r for c, r in self.hsv_ranges.items()}

        self.params = params or DetectionParams.from_dict(detection_data)

    def _tape_color_names(self) -> List[str]:
        return [name for name in self.color_ranges if name != SEPARATOR_NAME]

    def _min_colors_needed(self) -> int:
        """Fewest distinct colors a reading must have to be worth considering."""
        if self.combinations:
            return min(len(c.colors) for c in self.combinations.values())
        return self.params.min_band_colors

    def _target_colors(self) -> int:
        """Most colors we hope to read — used for the early-exit threshold."""
        if self.combinations:
            return max(len(c.colors) for c in self.combinations.values())
        return self.params.min_band_colors

    def _scan_line_length(self, frame_h: int, frame_w: int) -> int:
        if self.params.scan_line_length_px > 0:
            return self.params.scan_line_length_px
        return max(48, int(min(frame_h, frame_w) * self.params.scan_line_length_ratio))

    def _position_step(self, line_len: int) -> int:
        if self.params.position_search_step_px > 0:
            return self.params.position_search_step_px
        return max(12, line_len // 4)

    def _strip_height(self, frame_h: int) -> int:
        return max(3, int(frame_h * self.params.scan_strip_height_ratio))

    def _detect_region(self, frame_h: int, frame_w: int) -> Tuple[int, int, int, int]:
        """Pixel bounds (x0, y0, x1, y1) of the detection region of interest."""
        p = self.params
        x0 = int(frame_w * p.detect_x_min_ratio)
        x1 = int(frame_w * p.detect_x_max_ratio)
        y0 = int(frame_h * p.detect_y_min_ratio)
        y1 = int(frame_h * p.detect_y_max_ratio)
        return x0, y0, x1, y1

    def _region_is_full(self) -> bool:
        p = self.params
        return (
            p.detect_x_min_ratio <= 0.0
            and p.detect_x_max_ratio >= 1.0
            and p.detect_y_min_ratio <= 0.0
            and p.detect_y_max_ratio >= 1.0
        )

    def _in_region(self, cx: int, cy: int, frame_h: int, frame_w: int) -> bool:
        x0, y0, x1, y1 = self._detect_region(frame_h, frame_w)
        return x0 <= cx <= x1 and y0 <= cy <= y1

    def _fixed_scan_center(self, frame_h: int, frame_w: int) -> Tuple[int, int]:
        x0, y0, x1, y1 = self._detect_region(frame_h, frame_w)
        cx = (x0 + x1) // 2
        cy = int(frame_h * self.params.roi_y_center_ratio)
        cy = min(max(cy, y0), y1)
        return cx, cy

    def _build_band_color_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

        for name in self._tape_color_names():
            band_range = self.color_ranges[name]
            if band_range.h_min <= band_range.h_max:
                lower = np.array(
                    [band_range.h_min, band_range.s_min, band_range.v_min],
                    dtype=np.uint8,
                )
                upper = np.array(
                    [band_range.h_max, band_range.s_max, band_range.v_max],
                    dtype=np.uint8,
                )
                mask |= cv2.inRange(hsv, lower, upper)
            else:
                lower_high = np.array(
                    [band_range.h_min, band_range.s_min, band_range.v_min],
                    dtype=np.uint8,
                )
                upper_high = np.array([180, band_range.s_max, band_range.v_max], dtype=np.uint8)
                lower_low = np.array(
                    [0, band_range.s_min, band_range.v_min],
                    dtype=np.uint8,
                )
                upper_low = np.array(
                    [band_range.h_max, band_range.s_max, band_range.v_max],
                    dtype=np.uint8,
                )
                mask |= cv2.inRange(hsv, lower_high, upper_high)
                mask |= cv2.inRange(hsv, lower_low, upper_low)

        return mask

    def _position_candidates(
        self,
        frame_bgr: np.ndarray,
        line_len: int,
        step: int,
    ) -> List[Tuple[int, int]]:
        h, w = frame_bgr.shape[:2]
        margin = line_len // 2 + self._strip_height(h) // 2 + 1
        if margin * 2 >= min(h, w):
            return [self._fixed_scan_center(h, w)]

        band_mask = self._build_band_color_mask(frame_bgr)
        scored: List[Tuple[int, int, int]] = []
        half_w = line_len // 2
        half_h = max(4, line_len // 4)

        rx0, ry0, rx1, ry1 = self._detect_region(h, w)
        cy_lo, cy_hi = max(margin, ry0), min(h - margin, ry1)
        cx_lo, cx_hi = max(margin, rx0), min(w - margin, rx1)

        for cy in range(cy_lo, cy_hi, step):
            for cx in range(cx_lo, cx_hi, step):
                x0, x1 = cx - half_w, cx + half_w
                y0, y1 = cy - half_h, cy + half_h
                score = int(band_mask[y0:y1, x0:x1].sum())
                if score > 0:
                    scored.append((score, cx, cy))

        scored.sort(key=lambda item: item[0], reverse=True)
        candidates: List[Tuple[int, int]] = []
        min_sep = max(step // 2, 8)

        for _, cx, cy in scored:
            if len(candidates) >= self.params.max_position_candidates:
                break
            if any(abs(cx - px) < min_sep and abs(cy - py) < min_sep for px, py in candidates):
                continue
            candidates.append((cx, cy))

        if candidates:
            return self._expand_position_candidates(candidates, step, h, w, margin)

        fallback: List[Tuple[int, int]] = []
        for cy in range(cy_lo, cy_hi, step):
            for cx in range(cx_lo, cx_hi, step):
                fallback.append((cx, cy))
        if not fallback:
            return [self._fixed_scan_center(h, w)]
        return self._expand_position_candidates(
            fallback[: self.params.max_position_candidates],
            step,
            h,
            w,
            margin,
        )

    def _expand_position_candidates(
        self,
        candidates: List[Tuple[int, int]],
        step: int,
        frame_h: int,
        frame_w: int,
        margin: int,
    ) -> List[Tuple[int, int]]:
        refined: List[Tuple[int, int]] = []
        offsets = (-step // 2, 0, step // 2)
        seen: Set[Tuple[int, int]] = set()
        expand_limit = min(len(candidates), 8)

        for cx, cy in candidates[:expand_limit]:
            for dy in offsets:
                for dx in offsets:
                    nx, ny = cx + dx, cy + dy
                    if not (margin <= nx < frame_w - margin and margin <= ny < frame_h - margin):
                        continue
                    key = (nx, ny)
                    if key in seen:
                        continue
                    seen.add(key)
                    refined.append(key)
                    if len(refined) >= self.params.max_position_candidates:
                        return refined

        return refined or list(candidates[:expand_limit])

    def _scan_line_endpoints(
        self,
        center_x: int,
        center_y: int,
        length: int,
        angle_deg: float,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        half = length / 2.0
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        x0 = int(round(center_x - half * cos_a))
        y0 = int(round(center_y - half * sin_a))
        x1 = int(round(center_x + half * cos_a))
        y1 = int(round(center_y + half * sin_a))
        return (x0, y0), (x1, y1)

    def _local_roi(
        self,
        frame_h: int,
        frame_w: int,
        center_x: int,
        center_y: int,
        line_len: int,
        strip_h: int,
        angle_deg: float,
    ) -> Tuple[int, int, int, int]:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        cos_p = -sin_a
        sin_p = cos_a
        half_len = (line_len - 1) / 2.0
        half_thick = (strip_h - 1) / 2.0

        corners = []
        for col_off, row_off in (
            (-half_len, -half_thick),
            (half_len, -half_thick),
            (half_len, half_thick),
            (-half_len, half_thick),
        ):
            x = center_x + col_off * cos_a + row_off * cos_p
            y = center_y + col_off * sin_a + row_off * sin_p
            corners.append((x, y))

        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        x0 = max(0, int(math.floor(min(xs))))
        y0 = max(0, int(math.floor(min(ys))))
        x1 = min(frame_w, int(math.ceil(max(xs))))
        y1 = min(frame_h, int(math.ceil(max(ys))))
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def _extract_angled_strip(
        self,
        frame_bgr: np.ndarray,
        center_x: int,
        center_y: int,
        angle_deg: float,
        line_len: int,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[Tuple[int, int], Tuple[int, int]]]:
        h, w = frame_bgr.shape[:2]
        strip_h = self._strip_height(h)
        if line_len < 1:
            scan_line = self._scan_line_endpoints(center_x, center_y, 1, angle_deg)
            roi = self._local_roi(h, w, center_x, center_y, 1, strip_h, angle_deg)
            return frame_bgr[:0, :0], roi, scan_line

        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        cos_p = -sin_a
        sin_p = cos_a

        cols = np.arange(line_len, dtype=np.float32) - (line_len - 1) / 2.0
        rows = np.arange(strip_h, dtype=np.float32) - (strip_h - 1) / 2.0
        cc, rr = np.meshgrid(cols, rows)

        map_x = (center_x + cc * cos_a + rr * cos_p).astype(np.float32)
        map_y = (center_y + cc * sin_a + rr * sin_p).astype(np.float32)
        strip = cv2.remap(
            frame_bgr,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        scan_line = self._scan_line_endpoints(center_x, center_y, line_len, angle_deg)
        roi = self._local_roi(h, w, center_x, center_y, line_len, strip_h, angle_deg)
        return strip, roi, scan_line

    def _evaluate_band(
        self,
        runs: List[Tuple[str, int, int]],
    ) -> Tuple[bool, List[str], float, List[str]]:
        band_runs = [
            (n, s, e) for n, s, e in runs if n not in (SEPARATOR_NAME, UNKNOWN_NAME)
        ]
        colors_in_order = [n for n, _, _ in band_runs]
        unique_band_colors = {n for n, _, _ in band_runs}

        # A combination matches when the colors read equal its color set, in any
        # order. This is computed on every candidate reading but does NOT steer
        # the geometric search (see _score_result) — otherwise the search would
        # happily settle on a 2-color sub-slice of a real 3-color block.
        matched = (
            match_combinations(unique_band_colors, self.combinations)
            if self.combinations
            else []
        )

        if len(unique_band_colors) < self._min_colors_needed():
            return False, colors_in_order, 0.0, matched

        if self.params.require_black_separators:
            band_indices = [
                i for i, (n, _, _) in enumerate(runs) if n not in (SEPARATOR_NAME, UNKNOWN_NAME)
            ]
            for i in range(len(band_indices) - 1):
                idx_a = band_indices[i]
                idx_b = band_indices[i + 1]
                if idx_b - idx_a < 2:
                    return False, colors_in_order, 0.0, matched
                for mid in range(idx_a + 1, idx_b):
                    if runs[mid][0] != SEPARATOR_NAME:
                        return False, colors_in_order, 0.0, matched

        confidence = min(1.0, len(unique_band_colors) / 3.0)
        if len(unique_band_colors) >= 3:
            confidence = 1.0

        return True, colors_in_order, confidence, matched

    def _score_result(self, result: BandDetectionResult) -> float:
        if not result.band_detected:
            return 0.0
        unique_colors = len(set(result.colors_sequence))
        # Reading quality dominates so the search reads the fullest block; the
        # combination bonus is only a tie-breaker between equally rich readings.
        score = result.confidence + unique_colors * 0.1
        if result.combination_detected:
            score += 0.05
        return score

    def _detect_at(
        self,
        frame_bgr: np.ndarray,
        center_x: int,
        center_y: int,
        angle_deg: float,
        line_len: int,
    ) -> BandDetectionResult:
        strip, roi, scan_line = self._extract_angled_strip(
            frame_bgr, center_x, center_y, angle_deg, line_len
        )
        if strip.size == 0:
            return BandDetectionResult(
                band_detected=False,
                roi=roi,
                scan_line_y=center_y,
                scan_angle_deg=angle_deg,
                scan_line=scan_line,
                scan_center=(center_x, center_y),
            )

        hsv_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        # Thin center scan line — medians across a tall strip wash out small center tags.
        row_idx = hsv_strip.shape[0] // 2
        thin_rows = max(1, min(3, hsv_strip.shape[0]))
        y_start = max(0, row_idx - thin_rows // 2)
        y_end = min(hsv_strip.shape[0], y_start + thin_rows)
        scan_hsv = hsv_strip[y_start:y_end]
        labels = classify_named_columns(scan_hsv, self.color_ranges)
        runs = extract_label_runs(labels, self.params.min_segment_width_px)

        band_detected, colors_sequence, confidence, matched = self._evaluate_band(runs)
        color_runs = [(name, start, end) for name, start, end in runs]

        return BandDetectionResult(
            band_detected=band_detected,
            colors_sequence=colors_sequence,
            confidence=confidence,
            color_runs=color_runs,
            scan_line_y=center_y,
            roi=roi,
            scan_angle_deg=angle_deg,
            scan_line=scan_line,
            scan_center=(center_x, center_y),
            combination_detected=bool(matched),
            matched_combinations=matched,
        )

    def _search_angles_at(
        self,
        frame_bgr: np.ndarray,
        center_x: int,
        center_y: int,
        line_len: int,
    ) -> BandDetectionResult:
        coarse_step = max(0.5, self.params.angle_search_step_deg)
        best: Optional[BandDetectionResult] = None
        best_score = -1.0
        # Early-exit once a reading is as rich as we hope for and (if combinations
        # are defined) actually matches one.
        perfect_score = 1.0 + self._target_colors() * 0.1
        if self.combinations:
            perfect_score += 0.05

        angle = 0.0
        while angle < 180.0:
            result = self._detect_at(frame_bgr, center_x, center_y, angle, line_len)
            score = self._score_result(result)
            if score > best_score:
                best = result
                best_score = score
                if best_score >= perfect_score:
                    return best
            angle += coarse_step

        if best is None:
            return self._detect_at(frame_bgr, center_x, center_y, 0.0, line_len)

        refine_step = max(0.25, self.params.angle_refine_step_deg)
        if refine_step < coarse_step and best_score > 0:
            center = best.scan_angle_deg
            for offset in (-coarse_step, -refine_step, 0.0, refine_step, coarse_step):
                angle = (center + offset) % 180.0
                result = self._detect_at(frame_bgr, center_x, center_y, angle, line_len)
                score = self._score_result(result)
                if score > best_score:
                    best = result
                    best_score = score

        return best

    def _search_positions(self, frame_bgr: np.ndarray) -> BandDetectionResult:
        h, w = frame_bgr.shape[:2]
        line_len = self._scan_line_length(h, w)
        step = self._position_step(line_len)
        margin = line_len // 2 + self._strip_height(h) // 2 + 1
        candidates = self._position_candidates(frame_bgr, line_len, step)

        best: Optional[BandDetectionResult] = None
        best_score = -1.0
        # Early-exit once a reading is as rich as we hope for and (if combinations
        # are defined) actually matches one.
        perfect_score = 1.0 + self._target_colors() * 0.1
        if self.combinations:
            perfect_score += 0.05

        for center_x, center_y in candidates:
            if self.params.angle_search_enabled:
                result = self._search_angles_at(frame_bgr, center_x, center_y, line_len)
            else:
                result = self._detect_at(frame_bgr, center_x, center_y, 0.0, line_len)

            if not result.band_detected:
                continue
            cx, cy = result.scan_center
            if not self._in_region(cx, cy, h, w):
                continue

            score = self._score_result(result)
            if score > best_score:
                best = result
                best_score = score
                if best_score >= perfect_score:
                    return best

        if best is not None:
            return best

        center_x, center_y = self._fixed_scan_center(h, w)
        if self.params.angle_search_enabled:
            return self._search_angles_at(frame_bgr, center_x, center_y, line_len)
        return self._detect_at(frame_bgr, center_x, center_y, 0.0, line_len)

    def detect(self, frame_bgr: np.ndarray) -> BandDetectionResult:
        """Run detection on a BGR frame. Sets band_detected when 3-color band is found."""
        h, w = frame_bgr.shape[:2]
        line_len = self._scan_line_length(h, w)

        if self.params.position_search_enabled:
            return self._search_positions(frame_bgr)

        center_x, center_y = self._fixed_scan_center(h, w)
        if self.params.angle_search_enabled:
            return self._search_angles_at(frame_bgr, center_x, center_y, line_len)
        return self._detect_at(frame_bgr, center_x, center_y, 0.0, line_len)

    def draw_debug(
        self,
        frame_bgr: np.ndarray,
        result: BandDetectionResult,
    ) -> np.ndarray:
        """Overlay ROI, scan line, and detection status on a copy of the frame."""
        debug = frame_bgr.copy()
        h, w = debug.shape[:2]

        # Detection region of interest (only drawn when restricted to a sub-area).
        if not self._region_is_full():
            rx0, ry0, rx1, ry1 = self._detect_region(h, w)
            zone_color = (0, 200, 0) if result.band_detected else (160, 160, 160)
            # Dim everything outside the active zone so it reads at a glance.
            shade = debug.copy()
            cv2.rectangle(shade, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.rectangle(shade, (rx0, ry0), (rx1, ry1), (255, 255, 255), -1)
            mask = shade[:, :, 0] == 0
            debug[mask] = (0.5 * debug[mask]).astype(debug.dtype)
            cv2.rectangle(debug, (rx0, ry0), (rx1, ry1), zone_color, 2)
            cv2.putText(
                debug,
                "detection zone",
                (rx0 + 6, max(ry0 + 22, 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                zone_color,
                2,
            )

        x, y, rw, rh = result.roi
        color = (0, 255, 0) if result.band_detected else (0, 0, 255)
        cv2.rectangle(debug, (x, y), (x + rw, y + rh), color, 2)
        p0, p1 = result.scan_line
        cv2.line(debug, p0, p1, color, 2)
        cx, cy = result.scan_center
        cv2.drawMarker(debug, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)

        if result.combination_detected:
            status = "MATCH: " + ", ".join(result.matched_combinations)
        elif result.band_detected:
            status = "BAND DETECTED"
        else:
            status = "no band"
        label = f"{status} | colors: {', '.join(result.colors_sequence) or 'none'}"
        cv2.putText(
            debug,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        return debug
