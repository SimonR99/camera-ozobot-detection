#!/usr/bin/env python3
"""Live webcam demo for Ozobot band detection."""

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cal_path = args.calibration if args.calibration.exists() else None
    detector = BandDetector(calibration_path=cal_path)

    cap = open_checked(args)

    window = "Ozobot Band Detection"
    print("Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        display = detector.draw_debug(frame, result)

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
