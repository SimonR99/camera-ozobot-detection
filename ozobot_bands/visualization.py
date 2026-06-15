"""HSV masks and live preview overlays for color calibration."""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ozobot_bands.color_library import ColorLibrary, NamedColor, preview_bgr_for_entry
from ozobot_bands.colors import HSVRange


PENDING_TINT_BGR = (255, 255, 0)  # cyan/yellow for in-progress selection


def hsv_range_mask(hsv: np.ndarray, range_: HSVRange) -> np.ndarray:
    """Binary mask (uint8 0/255) for pixels inside an HSV range."""
    if range_.h_min <= range_.h_max:
        lower = np.array([range_.h_min, range_.s_min, range_.v_min], dtype=np.uint8)
        upper = np.array([range_.h_max, range_.s_max, range_.v_max], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    lower1 = np.array([range_.h_min, range_.s_min, range_.v_min], dtype=np.uint8)
    upper1 = np.array([180, range_.s_max, range_.v_max], dtype=np.uint8)
    lower2 = np.array([0, range_.s_min, range_.v_min], dtype=np.uint8)
    upper2 = np.array([range_.h_max, range_.s_max, range_.v_max], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))


def _apply_mask_tint(
    display: np.ndarray,
    mask: np.ndarray,
    tint_bgr: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    if not np.any(mask):
        return display
    tint = np.zeros_like(display)
    tint[:] = tint_bgr
    blended = cv2.addWeighted(tint, alpha, display, 1.0 - alpha, 0)
    # Keep highlight visible even when tint matches the source color.
    boosted = np.clip(blended.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    display = np.where(mask[..., None] > 0, boosted, display)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(display, contours, -1, tint_bgr, 2)
    return display


def draw_named_color_highlights(
    frame_bgr: np.ndarray,
    library: ColorLibrary,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay highlights for every saved named color."""
    if not library.colors:
        return frame_bgr.copy()

    display = frame_bgr.copy()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    for entry in library.colors.values():
        mask = hsv_range_mask(hsv, entry.hsv_range)
        tint = preview_bgr_for_entry(entry)
        display = _apply_mask_tint(display, mask, tint, alpha)

    return display


def draw_pending_highlight(
    frame_bgr: np.ndarray,
    pending_range: HSVRange,
    alpha: float = 0.55,
) -> np.ndarray:
    """Highlight pixels matching the pending (not yet saved) range."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = hsv_range_mask(hsv, pending_range)
    return _apply_mask_tint(frame_bgr.copy(), mask, PENDING_TINT_BGR, alpha)


def draw_calibration_ui(
    frame_bgr: np.ndarray,
    library: ColorLibrary,
    mode: str,
    text_input: str,
    hue_padding: int,
    sv_padding: int,
    pending_point: Optional[Tuple[int, int]],
    sample_radius: int,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Draw mode banner, text field, and saved color legend."""
    display = frame_bgr
    h, w = frame_bgr.shape[:2]

    banners = {
        "view": "CLICK color | P paste | D delete name | +/- when adjusting",
        "adjust": "ADJUST range +/- | V validate | Esc cancel",
        "name": "TYPE name tag | Enter save | Esc cancel",
        "paste": "PASTE #RRGGBB or r,g,b | Enter create | Esc cancel",
        "delete": "TYPE color name to delete | Enter confirm | Esc cancel",
    }
    cv2.putText(
        display,
        banners.get(mode, mode),
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    if mode in ("name", "paste", "delete"):
        prompt = {
            "name": "Name:",
            "paste": "Paste:",
            "delete": "Delete:",
        }[mode]
        cv2.rectangle(display, (8, 38), (w - 8, 72), (40, 40, 40), -1)
        cv2.putText(
            display,
            f"{prompt} {text_input}_",
            (14, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    elif mode == "adjust":
        cv2.putText(
            display,
            f"range pad H={hue_padding} SV={sv_padding}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

    json_hint = f"JSON: {output_path}" if output_path else ""
    cv2.putText(
        display,
        json_hint,
        (10, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )

    y = h - 28
    for name in library.names():
        entry = library.colors[name]
        tint = preview_bgr_for_entry(entry)
        cv2.putText(
            display,
            f"{name}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            tint,
            2,
        )
        y -= 22

    if pending_point is not None:
        px, py = pending_point
        cv2.circle(display, (px, py), sample_radius, (0, 255, 255), 2)
        cv2.drawMarker(display, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

    for entry in library.colors.values():
        if entry.sample_point is None:
            continue
        px, py = entry.sample_point
        tint = preview_bgr_for_entry(entry)
        cv2.circle(display, (px, py), 6, tint, 2)

    return display
