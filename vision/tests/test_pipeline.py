"""End-to-end-ish tests on synthetic frames: frame -> ordered mission.

No camera and no audio: TTS is muted and frames are generated in-memory.
"""

import numpy as np

from vision.colors import HSVRange
from vision.detector import BandDetector, DetectionParams
from vision.missions import build_mission
from vision.pipeline import MissionPipeline
from vision.tts import FrenchTTS


# Distinct, saturated BGR colours that classify cleanly.
BGR = {
    "red": (39, 32, 236),
    "green": (73, 183, 73),
    "blue": (198, 131, 17),
    "orange": (0, 140, 255),
}

# Wide HSV ranges keyed by name so the detector classifies the synthetic strips.
RANGES = {
    "red": HSVRange(170, 10, 100, 255, 80, 255),
    "green": HSVRange(35, 85, 50, 255, 50, 255),
    "blue": HSVRange(90, 130, 80, 255, 50, 255),
    "orange": HSVRange(10, 25, 100, 255, 100, 255),
    "black": HSVRange(0, 180, 0, 255, 0, 50),
}


def _band(colors, seg=40, height=120):
    width = seg * len(colors)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for i, name in enumerate(colors):
        frame[:, i * seg:(i + 1) * seg] = BGR[name]
    return frame


def _detector():
    return BandDetector(
        color_ranges=RANGES,
        params=DetectionParams(
            min_segment_width_px=5,
            scan_strip_height_ratio=0.5,
            scan_line_length_px=320,
        ),
    )


def test_detector_reads_colour_order():
    frame = _band(["blue", "green", "orange"])
    result = _detector().detect(frame)
    assert result.band_detected
    seq = [c for c in result.colors_sequence if c != "black"]
    assert set(seq) >= {"blue", "green", "orange"}


def test_pipeline_builds_mission_from_frame():
    pipeline = MissionPipeline(tts=FrenchTTS(enabled=False))
    pipeline.detector = _detector()  # use synthetic-friendly ranges
    pipeline.action_map = {
        "blue": "Tourne", "green": "Avance", "orange": "Salue",
    }
    obs = pipeline.process(_band(["blue", "green", "orange"]))
    assert obs.detected
    assert obs.mission.colors[:1] == [obs.result.colors_sequence[0]]
    assert all(a for a in obs.mission.actions)


def test_repeated_trailing_colour_is_read():
    # green-blue-orange-blue has only 3 unique colours; the trailing blue repeat
    # must still be captured (segment-count tie-breaker in _score_result).
    frame = _band(["green", "blue", "orange", "blue"])
    result = _detector().detect(frame)
    seq = [c for c in result.colors_sequence if c != "black"]
    assert seq.count("blue") == 2
    assert seq[-1] == "blue"


def test_uniform_white_sheet_has_no_mission():
    frame = np.full((120, 240, 3), (245, 245, 245), dtype=np.uint8)
    result = _detector().detect(frame)
    mission = build_mission(result.colors_sequence, {})
    assert mission.is_empty
