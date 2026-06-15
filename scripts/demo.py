#!/usr/bin/env python3
"""Live webcam demo for Ozobot band detection.

Detection is restricted to a region of interest (default: the bottom half of the
frame in height and the middle half in width). A band only counts when its three
colors are read inside that zone. When a 3-color block is detected the zone turns
green and a banner is shown.
"""

import argparse
from pathlib import Path

import cv2

from ozobot_bands import BandDetector, add_source_args, open_checked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ozobot band detection demo")
    add_source_args(parser)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
        help="Calibration file (uses defaults if missing)",
    )
    # Detection region as frame fractions. Defaults: bottom half (height),
    # middle half (width) -> a 1/4-no / 1/2-yes / 1/4-no split horizontally.
    parser.add_argument("--region-x-min", type=float, default=0.25)
    parser.add_argument("--region-x-max", type=float, default=0.75)
    parser.add_argument("--region-y-min", type=float, default=0.5)
    parser.add_argument("--region-y-max", type=float, default=1.0)
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Detect across the whole frame (ignore the region restriction)",
    )
    return parser.parse_args()


def apply_region(detector: BandDetector, args: argparse.Namespace) -> None:
    if args.full_frame:
        x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    else:
        x0, x1, y0, y1 = (
            args.region_x_min,
            args.region_x_max,
            args.region_y_min,
            args.region_y_max,
        )
    detector.params.detect_x_min_ratio = x0
    detector.params.detect_x_max_ratio = x1
    detector.params.detect_y_min_ratio = y0
    detector.params.detect_y_max_ratio = y1


def draw_detection_banner(display, result) -> None:
    """Big, obvious overlay shown when a 3-color block is detected in the zone."""
    if not result.band_detected:
        return
    h, w = display.shape[:2]
    banner_h = max(48, h // 12)
    # Solid bar so it cleanly covers draw_debug's own status text underneath.
    cv2.rectangle(display, (0, 0), (w, banner_h), (0, 150, 0), -1)

    if result.combination_detected:
        text = "MATCH: " + ", ".join(result.matched_combinations)
    else:
        text = "3 BANDS DETECTED"
    colors = " | ".join(result.colors_sequence)
    cv2.putText(
        display,
        f"{text}   [{colors}]",
        (16, int(banner_h * 0.66)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    cal_path = args.calibration if args.calibration.exists() else None
    detector = BandDetector(calibration_path=cal_path)
    apply_region(detector, args)

    cap = open_checked(args)

    window = "Ozobot Band Detection"
    print("Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        display = detector.draw_debug(frame, result)
        draw_detection_banner(display, result)

        if result.combination_detected:
            print(
                f"MATCH {result.matched_combinations} | colors={result.colors_sequence}"
            )
        elif result.band_detected:
            print(
                f"BAND DETECTED | colors={result.colors_sequence} "
                f"confidence={result.confidence:.2f}"
            )

        cv2.imshow(window, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
