"""Tests for combinations: storage, set matching, run extraction, detection."""

import numpy as np

from ozobot_bands.color_library import (
    ColorLibrary,
    Combination,
    NamedColor,
    load_color_library,
    match_combinations,
    save_color_library,
)
from ozobot_bands.colors import (
    HSVRange,
    OzobotColor,
    UNKNOWN_NAME,
    default_hsv_ranges,
    extract_label_runs,
)
from ozobot_bands.colors import COLOR_NAMES
from ozobot_bands.detector import BandDetector, DetectionParams
from ozobot_bands.synthetic import generate_tag_image


def _name_ranges():
    return {COLOR_NAMES[c]: r for c, r in default_hsv_ranges().items()}


def _detector(combos):
    return BandDetector(
        color_ranges=_name_ranges(),
        combinations=combos,
        params=DetectionParams(min_segment_width_px=5, scan_strip_height_ratio=0.08),
    )


# --- storage --------------------------------------------------------------

def test_save_load_combinations_roundtrip(tmp_path):
    path = tmp_path / "cal.json"
    library = ColorLibrary()
    library.upsert(NamedColor(name="red", hsv_range=HSVRange(0, 10, 100, 255, 80, 255)))
    library.upsert_combination(Combination("ozobot", ["red", "green", "blue"]))
    save_color_library(path, library)

    loaded = load_color_library(path)
    assert "ozobot" in loaded.combinations
    assert loaded.combinations["ozobot"].color_set() == frozenset({"red", "green", "blue"})


def test_load_v2_library_has_no_combinations(tmp_path):
    path = tmp_path / "v2.json"
    path.write_text('{"version": 2, "colors": {}, "detection": {}}')
    loaded = load_color_library(path)
    assert loaded.combinations == {}


# --- set matching ---------------------------------------------------------

def test_match_is_order_independent():
    combos = {"go": Combination("go", ["red", "green", "blue"])}
    assert match_combinations({"blue", "red", "green"}, combos) == ["go"]


def test_match_requires_exact_set():
    combos = {"rg": Combination("rg", ["red", "green"])}
    # superset should not match
    assert match_combinations({"red", "green", "blue"}, combos) == []
    # subset should not match
    assert match_combinations({"red"}, combos) == []
    assert match_combinations({"red", "green"}, combos) == ["rg"]


def test_match_returns_all_matching_names_sorted():
    combos = {
        "b": Combination("b", ["red", "green"]),
        "a": Combination("a", ["green", "red"]),
    }
    assert match_combinations({"red", "green"}, combos) == ["a", "b"]


# --- run extraction merge fix --------------------------------------------

def test_short_noise_blip_does_not_split_a_run():
    # red(10) | noise(2, below min) | red(10) should merge into one red run.
    labels = ["red"] * 10 + ["green"] * 2 + ["red"] * 10
    runs = extract_label_runs(labels, min_width=5)
    assert runs == [("red", 0, 22)]


def test_distinct_short_runs_dropped_without_merging_neighbors():
    labels = ["red"] * 10 + ["green"] * 2 + ["blue"] * 10
    runs = extract_label_runs(labels, min_width=5)
    assert [r[0] for r in runs] == ["red", "blue"]


# --- detection against combinations --------------------------------------

def test_detects_defined_three_color_combination():
    frame, _ = generate_tag_image(seed=42)  # red, green, blue block
    detector = _detector({"ozobot": Combination("ozobot", ["red", "green", "blue"])})
    result = detector.detect(frame)
    assert result.combination_detected
    assert result.matched_combinations == ["ozobot"]


def test_two_color_combination_does_not_match_three_color_block():
    # The position/angle search must read the full 3-color block, not a 2-color
    # sub-slice that would falsely satisfy a 2-color combination.
    frame, _ = generate_tag_image(seed=42)
    detector = _detector({"rg": Combination("rg", ["red", "green"])})
    result = detector.detect(frame)
    assert not result.combination_detected
    assert set(result.colors_sequence) == {"red", "green", "blue"}


def test_no_combination_match_when_block_is_unknown_set():
    frame, _ = generate_tag_image(seed=42)
    detector = _detector({"rb": Combination("rb", ["red", "blue"])})
    result = detector.detect(frame)
    assert not result.combination_detected
