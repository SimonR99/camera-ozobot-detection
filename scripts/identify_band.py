#!/usr/bin/env python3
"""Identify Ozobot band colors from the camera and save results to JSON.

Live preview shows detection status. Press 's' to capture and write JSON,
or use --auto to save when a band is detected stably.

Keys (interactive mode):
  click  - move scan line to click position and identify colors there
  s      - save current band colors to JSON
  q      - quit
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from ozobot_bands import BandDetector
from ozobot_bands.calibration import infer_color_at_pixel
from ozobot_bands.colors import COLOR_NAMES
from ozobot_bands.detector import BandDetectionResult, DetectionParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify Ozobot band colors and save to JSON",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
        help="Calibration file (uses defaults if missing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("band_detection.json"),
        help="JSON file to write",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-save when band is detected for --stable-frames consecutive frames",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=8,
        help="Frames required before auto-save (default: 8)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Grab one frame, save JSON, and exit (no preview window)",
    )
    parser.add_argument(
        "--save-image",
        type=Path,
        default=None,
        help="Also save the captured frame as an image (path)",
    )
    parser.add_argument(
        "--min-segment-width",
        type=int,
        default=None,
        help="Override min segment width in pixels",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window (use with --auto or --once)",
    )
    return parser.parse_args()


def result_to_json(
    result: BandDetectionResult,
    frame_shape: Optional[tuple] = None,
) -> dict:
    """Build a JSON-serializable record for a band detection."""
    segments = [
        {"color": color, "start_px": start, "end_px": end, "width_px": end - start}
        for color, start, end in result.color_runs
    ]
    band_segments = [
        s for s in segments if s["color"] in ("red", "green", "blue")
    ]

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "band_detected": result.band_detected,
        "colors": result.colors_sequence,
        "color_code": "-".join(result.colors_sequence) if result.colors_sequence else "",
        "unique_colors": sorted({c for c in result.colors_sequence}),
        "confidence": round(result.confidence, 4),
        "segments": segments,
        "band_segments": band_segments,
        "scan_line_y": result.scan_line_y,
        "roi": {
            "x": result.roi[0],
            "y": result.roi[1],
            "width": result.roi[2],
            "height": result.roi[3],
        },
    }
    if frame_shape is not None:
        record["frame_height"] = frame_shape[0]
        record["frame_width"] = frame_shape[1]
    return record


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print(f"Saved: {path}")
    if record["band_detected"]:
        print(f"  colors: {record['color_code']}")
        print(f"  confidence: {record['confidence']}")
    else:
        print("  no band detected")


def build_detector(args: argparse.Namespace) -> BandDetector:
    cal_path = args.calibration if args.calibration.exists() else None
    params = None
    if args.min_segment_width is not None:
        base = DetectionParams()
        if cal_path:
            from ozobot_bands.calibration import infer_color_at_pixel, load_calibration

            _, detection_data, _, _ = load_calibration(cal_path)
            base = DetectionParams.from_dict(detection_data)
        params = DetectionParams(
            min_segment_width_px=args.min_segment_width,
            scan_strip_height_ratio=base.scan_strip_height_ratio,
            min_band_colors=base.min_band_colors,
            require_black_separators=base.require_black_separators,
            roi_y_center_ratio=base.roi_y_center_ratio,
            roi_width_ratio=base.roi_width_ratio,
        )
    return BandDetector(calibration_path=cal_path, params=params)


def capture_and_save(
    detector: BandDetector,
    frame,
    args: argparse.Namespace,
    label: str = "manual",
) -> BandDetectionResult:
    result = detector.detect(frame)
    record = result_to_json(result, frame.shape[:2])
    record["capture_mode"] = label

    write_json(args.output, record)

    if args.save_image:
        args.save_image.parent.mkdir(parents=True, exist_ok=True)
        debug = detector.draw_debug(frame, result)
        cv2.imwrite(str(args.save_image), debug)
        print(f"Saved image: {args.save_image}")

    return result


def run_once(detector: BandDetector, cap: cv2.VideoCapture, args: argparse.Namespace) -> int:
    ret, frame = cap.read()
    if not ret:
        raise SystemExit("Failed to read from camera")
    capture_and_save(detector, frame, args, label="once")
    return 0 if detector.detect(frame).band_detected else 1


def run_interactive(detector: BandDetector, cap: cv2.VideoCapture, args: argparse.Namespace) -> int:
    window = "Identify Ozobot Band (click=scan, s=save, q=quit)"
    show_preview = not args.no_preview
    stable_count = 0
    saved_auto = False
    mouse_state: dict = {"click_xy": None, "last_click": None}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state["click_xy"] = (x, y)

    if show_preview:
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_mouse)
        print("Click on the band to scan that row. Press 's' to save JSON, 'q' to quit.")
        if args.auto:
            print(f"Auto-save enabled after {args.stable_frames} stable frames.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mouse_state["click_xy"] is not None:
            x, y = mouse_state["click_xy"]
            mouse_state["click_xy"] = None
            h = frame.shape[0]
            detector.params.roi_y_center_ratio = y / h
            clicked_color = infer_color_at_pixel(frame, x, y, detector.hsv_ranges)
            color_label = COLOR_NAMES.get(clicked_color, "unknown")
            mouse_state["last_click"] = (x, y, color_label)
            print(
                f"Scan line -> y={y} ({detector.params.roi_y_center_ratio:.2f}) "
                f"pixel color: {color_label}"
            )

        result = detector.detect(frame)

        if args.auto and not saved_auto:
            if result.band_detected:
                stable_count += 1
                if stable_count >= args.stable_frames:
                    capture_and_save(detector, frame, args, label="auto")
                    saved_auto = True
            else:
                stable_count = 0

        if show_preview:
            display = detector.draw_debug(frame, result)
            last = mouse_state.get("last_click")
            if last:
                lx, ly, name = last
                cv2.circle(display, (lx, ly), 12, (255, 255, 255), 2)
                cv2.putText(
                    display,
                    name,
                    (lx + 14, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
            hint = "AUTO SAVED" if saved_auto else "click=scan | s=save | q=quit"
            cv2.putText(
                display,
                hint,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                capture_and_save(detector, frame, args, label="manual")
        elif args.auto and saved_auto:
            break

    if show_preview:
        cv2.destroyAllWindows()
    return 0


def main() -> None:
    args = parse_args()
    detector = build_detector(args)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    try:
        if args.once:
            code = run_once(detector, cap, args)
            raise SystemExit(code)
        run_interactive(detector, cap, args)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
