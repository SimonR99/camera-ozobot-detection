#!/usr/bin/env python3
"""Fast live color highlighter.

Outlines every calibrated color it sees and shows which combinations are
present, at full camera speed. Unlike demo.py it does NOT run the scan-line
position/angle search (that is what makes the demo slow); it just masks each
calibrated color with cv2.inRange, which is vectorized and runs in real time.

Because combinations use unordered-set matching, "all of a combination's colors
are visible as blobs" is exactly a match — so this is a faithful, fast validator.

Usage:
  python scripts/highlight.py --calibration calibration.json        # live camera
  python scripts/highlight.py --image images/image_1.jpg            # one still image

Keys (live): q quit
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ozobot_bands.color_library import (
    ColorLibrary,
    load_color_library,
    match_combinations,
    preview_bgr_for_entry,
)
from ozobot_bands.colors import SEPARATOR_NAME
from ozobot_bands.visualization import hsv_range_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast live color/combination highlighter")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calibration", type=Path, default=Path("calibration.json"))
    parser.add_argument("--image", type=Path, default=None, help="Run once on a still image instead of the camera")
    parser.add_argument("--out", type=Path, default=Path("highlight_out.jpg"), help="Output path for --image mode")
    parser.add_argument("--width", type=int, default=960, help="Resize frames to this width for speed (0 = keep original)")
    parser.add_argument("--min-area-frac", type=float, default=0.0008, help="Ignore blobs smaller than this fraction of the frame")
    parser.add_argument("--include-black", action="store_true", help="Also highlight the 'black' separator color")
    return parser.parse_args()


def _highlight_colors(library: ColorLibrary, include_black: bool):
    return [
        (name, entry)
        for name, entry in library.colors.items()
        if include_black or name != SEPARATOR_NAME
    ]


def annotate(
    frame_bgr: np.ndarray,
    library: ColorLibrary,
    min_area: int,
    include_black: bool,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Outline every visible calibrated color; return (display, present, matched)."""
    display = frame_bgr.copy()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    present: List[str] = []

    for name, entry in _highlight_colors(library, include_black):
        mask = hsv_range_mask(hsv, entry.hsv_range)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not big:
            continue
        present.append(name)
        tint = preview_bgr_for_entry(entry)
        for contour in big:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display, (x, y), (x + w, y + h), tint, 3)
        biggest = max(big, key=cv2.contourArea)
        bx, by, _, _ = cv2.boundingRect(biggest)
        cv2.putText(
            display, name, (bx, max(20, by - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, tint, 2,
        )

    matched = match_combinations(set(present), library.combinations)
    return display, sorted(present), matched


def draw_banner(display: np.ndarray, present: List[str], matched: List[str], fps: float) -> None:
    h, w = display.shape[:2]
    cv2.rectangle(display, (0, 0), (w, 30), (30, 30, 30), -1)
    cv2.putText(
        display, f"colors: {', '.join(present) or 'none'}", (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
    )
    if fps:
        cv2.putText(
            display, f"{fps:4.0f} fps", (w - 90, 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
        )
    banner = "MATCH: " + ", ".join(matched) if matched else "no combination"
    color = (0, 255, 0) if matched else (160, 160, 160)
    cv2.rectangle(display, (0, h - 34), (w, h), (30, 30, 30), -1)
    cv2.putText(display, banner, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or frame.shape[1] == width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, int(frame.shape[0] * scale)))


def min_area_for(frame: np.ndarray, frac: float) -> int:
    h, w = frame.shape[:2]
    return max(50, int(h * w * frac))


def run_image(args: argparse.Namespace, library: ColorLibrary) -> None:
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")
    frame = resize_width(frame, args.width)
    display, present, matched = annotate(
        frame, library, min_area_for(frame, args.min_area_frac), args.include_black
    )
    draw_banner(display, present, matched, 0.0)
    cv2.imwrite(str(args.out), display)
    print(f"colors: {present}")
    print(f"match : {matched or 'none'}")
    print(f"saved : {args.out}")


def run_camera(args: argparse.Namespace, library: ColorLibrary) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    window = "Color Highlighter (q to quit)"
    cv2.namedWindow(window)
    print("Fast highlighter running. Press q to quit.")
    prev = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_width(frame, args.width)
        display, present, matched = annotate(
            frame, library, min_area_for(frame, args.min_area_frac), args.include_black
        )
        now = time.time()
        dt = now - prev
        prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
        draw_banner(display, present, matched, fps)
        cv2.imshow(window, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if not args.calibration.exists():
        raise SystemExit(f"No calibration file at {args.calibration}")
    library = load_color_library(args.calibration)
    if args.image is not None:
        run_image(args, library)
    else:
        run_camera(args, library)


if __name__ == "__main__":
    main()
