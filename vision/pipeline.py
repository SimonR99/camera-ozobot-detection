"""Glue: frame -> band detection -> ordered mission -> spoken French actions.

:class:`MissionPipeline` wraps a :class:`vision.detector.BandDetector` and adds the
mission mapping plus debounced French narration. It is frame-source agnostic —
``process(frame)`` takes any BGR ``ndarray``, so the same object serves the
webcam loop, a ROS 2 callback, or a single still image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from vision.detector import BandDetector, BandDetectionResult, DetectionParams

from vision.missions import Mission, build_mission, load_action_map
from vision.tts import FrenchTTS


@dataclass
class MissionObservation:
    """A detection plus the mission derived from it for one frame."""

    result: BandDetectionResult
    mission: Mission

    @property
    def detected(self) -> bool:
        # A mission needs at least the detector's minimum number of colours.
        return self.result.band_detected and not self.mission.is_empty


class MissionPipeline:
    """Detect colour strips on a white sheet and narrate them in French."""

    def __init__(
        self,
        calibration_path: Optional[Path] = None,
        actions_path: Optional[Path] = None,
        tts: Optional[FrenchTTS] = None,
        params: Optional[DetectionParams] = None,
        stable_frames: int = 6,
    ):
        cal = calibration_path if calibration_path and calibration_path.exists() else None
        self.detector = BandDetector(calibration_path=cal, params=params)
        self.action_map = load_action_map(cal, actions_path)
        self.tts = tts if tts is not None else FrenchTTS()
        self.stable_frames = stable_frames

        # Debounce state so we narrate a mission once, not every frame.
        self._stable_count = 0
        self._spoken_code: Optional[str] = None

    def process(self, frame: np.ndarray) -> MissionObservation:
        """Run detection on one frame and build (but do not speak) its mission."""
        result = self.detector.detect(frame)
        mission = build_mission(result.colors_sequence, self.action_map)
        return MissionObservation(result=result, mission=mission)

    def narrate(self, mission: Mission, full: bool = True) -> None:
        """Speak a mission immediately: the whole narration, or step phrases."""
        if mission.is_empty:
            return
        if full:
            self.tts.say(mission.narration())
        else:
            self.tts.say_many(mission.actions)

    def update(self, frame: np.ndarray, full: bool = True) -> MissionObservation:
        """Process a frame and auto-narrate once a mission is stable and new.

        Intended for the live loop: speaks only after the same colour code has
        held for ``stable_frames`` frames, and never repeats the same mission
        until a different (or no) mission is seen, so the sheet can be re-read.
        """
        obs = self.process(frame)
        code = "-".join(obs.mission.colors) if obs.detected else None

        if code is None:
            self._stable_count = 0
            self._spoken_code = None
            return obs

        self._stable_count += 1
        if self._stable_count >= self.stable_frames and code != self._spoken_code:
            self.narrate(obs.mission, full=full)
            self._spoken_code = code
        return obs

    def reset(self) -> None:
        """Forget debounce state so the next stable mission is spoken again."""
        self._stable_count = 0
        self._spoken_code = None
