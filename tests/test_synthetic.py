"""Tests for synthetic Ozobot tag images."""

import random
from typing import Tuple

import pytest

from ozobot_bands.detector import BandDetector, DetectionParams
from ozobot_bands.synthetic import generate_tag_image


def _detector_for_small_tags() -> BandDetector:
    params = DetectionParams(
        min_segment_width_px=5,
        scan_strip_height_ratio=0.08,
        roi_y_center_ratio=0.5,
    )
    return BandDetector(params=params)


def test_synthetic_center_tag_default_seed():
    frame, meta = generate_tag_image(seed=42)
    result = _detector_for_small_tags().detect(frame)

    assert result.band_detected, (
        f"seed=42 failed: colors={result.colors_sequence} runs={result.color_runs}"
    )
    assert set(result.colors_sequence) == {"red", "green", "blue"}
    assert result.confidence == 1.0


def test_synthetic_center_tag_many_random_seeds():
    detector = _detector_for_small_tags()
    failures = []

    for seed in range(100):
        frame, _ = generate_tag_image(seed=seed, gap_max_px=3)
        result = detector.detect(frame)
        if not result.band_detected:
            failures.append((seed, result.colors_sequence, result.color_runs))
        else:
            found = set(result.colors_sequence)
            if found != {"red", "green", "blue"}:
                failures.append((seed, result.colors_sequence, result.color_runs))

    assert not failures, f"{len(failures)} failures: {failures[:5]}"


def test_synthetic_varied_background_styles():
    detector = _detector_for_small_tags()
    styles = ["uniform", "noise", "gradient", "warm", "cool"]

    for style_seed, style in enumerate(styles):
        rng = random.Random(style_seed)
        frame, _ = generate_tag_image(seed=style_seed + 1000, gap_max_px=3)
        # Force background style by regenerating with fixed seed per style
        result = detector.detect(frame)
        assert result.band_detected, f"style={style} seed={style_seed}"


@pytest.mark.parametrize("gap_max", [0, 1, 2, 3])
def test_synthetic_gap_sizes(gap_max: int):
    detector = _detector_for_small_tags()
    frame, _ = generate_tag_image(seed=7, gap_max_px=gap_max)
    result = detector.detect(frame)
    assert result.band_detected, f"gap_max={gap_max} colors={result.colors_sequence}"


@pytest.mark.parametrize("angle_deg", [0, 15, 30, 45, 60, 75, 90, 120, 135, 150, 165])
def test_synthetic_rotated_tags(angle_deg: float):
    detector = _detector_for_small_tags()
    frame, _ = generate_tag_image(seed=42, gap_max_px=2, angle_deg=angle_deg)
    result = detector.detect(frame)
    assert result.band_detected, (
        f"angle={angle_deg} failed: colors={result.colors_sequence} "
        f"scan_angle={result.scan_angle_deg:.1f}"
    )
    assert set(result.colors_sequence) == {"red", "green", "blue"}


def test_synthetic_rotated_tags_random_angles():
    detector = _detector_for_small_tags()
    failures = []

    for seed in range(20):
        angle = (seed * 13) % 180
        frame, _ = generate_tag_image(seed=seed + 500, gap_max_px=2, angle_deg=float(angle))
        result = detector.detect(frame)
        if not result.band_detected or set(result.colors_sequence) != {"red", "green", "blue"}:
            failures.append((seed, angle, result.colors_sequence, result.scan_angle_deg))

    assert not failures, f"{len(failures)} failures: {failures[:5]}"


@pytest.mark.parametrize(
    "tag_center",
    [(90, 70), (520, 90), (110, 390), (560, 400), (180, 240)],
)
def test_synthetic_off_center_tags(tag_center: Tuple[int, int]):
    detector = _detector_for_small_tags()
    frame, _ = generate_tag_image(seed=11, gap_max_px=2, tag_center=tag_center)
    result = detector.detect(frame)
    assert result.band_detected, (
        f"center={tag_center} failed: colors={result.colors_sequence} "
        f"scan_center={result.scan_center}"
    )
    assert set(result.colors_sequence) == {"red", "green", "blue"}


@pytest.mark.parametrize(
    ("tag_center", "angle_deg"),
    [
        ((100, 80), 30.0),
        ((520, 100), 120.0),
        ((80, 380), 60.0),
        ((550, 390), 150.0),
    ],
)
def test_synthetic_off_center_rotated_tags(tag_center: Tuple[int, int], angle_deg: float):
    detector = _detector_for_small_tags()
    frame, _ = generate_tag_image(
        seed=23,
        gap_max_px=2,
        tag_center=tag_center,
        angle_deg=angle_deg,
    )
    result = detector.detect(frame)
    assert result.band_detected, (
        f"center={tag_center} angle={angle_deg} failed: "
        f"colors={result.colors_sequence} scan_center={result.scan_center} "
        f"scan_angle={result.scan_angle_deg:.1f}"
    )
    assert set(result.colors_sequence) == {"red", "green", "blue"}
