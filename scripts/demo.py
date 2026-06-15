#!/usr/bin/env python3
"""Live webcam demo for Ozobot band detection."""

import argparse
from pathlib import Path

import cv2

from ozobot_bands import BandDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ozobot band detection demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
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

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    window = "Ozobot Band Detection"
    print("Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        display = detector.draw_debug(frame, result)

        if result.band_detected:
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
