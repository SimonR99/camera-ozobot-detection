"""Generate synthetic Ozobot-style tag images for pipeline testing."""

import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

OZO_RED = (39, 32, 236)
OZO_GREEN = (73, 183, 73)
OZO_BLUE = (198, 131, 17)

COLOR_BGR = {
    "red": OZO_RED,
    "green": OZO_GREEN,
    "blue": OZO_BLUE,
}


def _make_background(
    width: int,
    height: int,
    style: str,
    rng: random.Random,
) -> np.ndarray:
    """Build a white-ish background with slight variation."""
    if style == "uniform":
        v = rng.randint(220, 255)
        return np.full((height, width, 3), (v, v, v), dtype=np.uint8)

    if style == "warm":
        return np.full(
            (height, width, 3),
            (rng.randint(230, 255), rng.randint(235, 255), rng.randint(240, 255)),
            dtype=np.uint8,
        )

    if style == "cool":
        return np.full(
            (height, width, 3),
            (rng.randint(240, 255), rng.randint(240, 255), rng.randint(230, 255)),
            dtype=np.uint8,
        )

    if style == "gradient":
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        low = rng.randint(210, 235)
        high = rng.randint(240, 255)
        for y in range(height):
            t = y / max(height - 1, 1)
            val = int(low + (high - low) * t)
            frame[y, :] = (val, val, val)
        return frame

    # noise
    base = rng.randint(225, 250)
    noise = np.random.randint(-12, 13, size=(height, width, 3), dtype=np.int16)
    frame = np.clip(base + noise, 200, 255).astype(np.uint8)
    return frame


def generate_tag_image(
    width: int = 640,
    height: int = 480,
    tag_height: int = 14,
    color_width_range: Tuple[int, int] = (14, 22),
    gap_max_px: int = 3,
    seed: Optional[int] = None,
    color_order: Optional[List[str]] = None,
    angle_deg: float = 0.0,
    tag_center: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Create a frame with a small 3-color tag on a varied white background.

    Gaps between color segments use the local background color (0 to gap_max_px wide).
    angle_deg rotates the tag around tag_center (or frame center when omitted).
    tag_center places the tag anywhere on the frame.
    Returns (bgr_image, metadata).
    """
    rng = random.Random(seed)
    style = rng.choice(["uniform", "noise", "gradient", "warm", "cool"])
    frame = _make_background(width, height, style, rng)

    order = color_order or ["red", "green", "blue"]
    if len(order) != 3:
        raise ValueError("color_order must contain exactly 3 colors")

    segment_widths = [
        rng.randint(color_width_range[0], color_width_range[1]) for _ in order
    ]
    gaps = [rng.randint(0, gap_max_px) for _ in range(len(order) - 1)]

    tag_width = sum(segment_widths) + sum(gaps)
    center_x, center_y = tag_center or (width // 2, height // 2)
    x0 = center_x - tag_width // 2
    y0 = center_y - tag_height // 2
    y1 = y0 + tag_height

    cursor = x0
    segments: List[dict] = []

    for i, name in enumerate(order):
        w = segment_widths[i]
        bgr = COLOR_BGR[name]
        frame[y0:y1, cursor:cursor + w] = bgr
        segments.append({"color": name, "x0": cursor, "x1": cursor + w, "width": w})
        cursor += w

        if i < len(gaps):
            gap = gaps[i]
            if gap > 0:
                # Gap shows background — sample from frame edges or regenerate patch.
                gap_patch = _make_background(gap, tag_height, style, rng)
                frame[y0:y1, cursor:cursor + gap] = gap_patch
                segments.append({"color": "gap", "x0": cursor, "x1": cursor + gap, "width": gap})
                cursor += gap

    if angle_deg % 360 != 0:
        matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)
        frame = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    meta = {
        "seed": seed,
        "background_style": style,
        "tag_bbox": (x0, y0, tag_width, tag_height),
        "tag_center": (center_x, center_y),
        "segments": segments,
        "color_order": order,
        "gaps_px": gaps,
        "angle_deg": angle_deg,
    }
    return frame, meta


def save_tag_image(path: str, seed: int, **kwargs) -> Tuple[np.ndarray, dict]:
    """Generate and save a synthetic tag image."""
    frame, meta = generate_tag_image(seed=seed, **kwargs)
    cv2.imwrite(path, frame)
    return frame, meta
