#!/usr/bin/env python3
"""Generate synthetic tag images and verify band detection."""

import argparse
from pathlib import Path

import cv2

from ozobot_bands.detector import BandDetector, DetectionParams
from ozobot_bands.synthetic import generate_tag_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test detection on synthetic Ozobot tags")
    parser.add_argument("--count", type=int, default=50, help="Number of random images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_output/synthetic"),
        help="Directory for sample images and debug overlays",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="Starting RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    params = DetectionParams(
        min_segment_width_px=5,
        scan_strip_height_ratio=0.08,
        roi_y_center_ratio=0.5,
    )
    detector = BandDetector(params=params)

    passed = 0
    failed_seeds = []

    for i in range(args.count):
        seed = args.seed_start + i
        frame, meta = generate_tag_image(seed=seed, gap_max_px=3)
        result = detector.detect(frame)

        ok = (
            result.band_detected
            and set(result.colors_sequence) == {"red", "green", "blue"}
        )
        if ok:
            passed += 1
        else:
            failed_seeds.append(
                (seed, result.band_detected, result.colors_sequence, result.color_runs)
            )

        # Save first 5 and any failures for visual inspection.
        if i < 5 or not ok:
            tag = "pass" if ok else "FAIL"
            stem = f"{tag}_seed{seed}_{meta['background_style']}"
            cv2.imwrite(str(args.output_dir / f"{stem}.png"), frame)
            debug = detector.draw_debug(frame, result)
            cv2.imwrite(str(args.output_dir / f"{stem}_debug.png"), debug)

    print(f"Results: {passed}/{args.count} passed")
    if failed_seeds:
        print("Failures:")
        for seed, detected, colors, runs in failed_seeds[:10]:
            print(f"  seed={seed} detected={detected} colors={colors} runs={runs}")
        raise SystemExit(1)

    print(f"Sample images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
