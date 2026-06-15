"""Vision mission pipeline (self-contained).

Reads a white sheet from a webcam or a ROS 2 image topic, detects the ordered
sequence of colour-tape strips on it, turns that sequence into an ordered
*mission* (one action per colour), and speaks the correction actions in French.

Layout:

* :mod:`vision.colors` / :mod:`vision.calibration` / :mod:`vision.color_library`
  — calibrated HSV colours and per-column classification;
* :mod:`vision.detector` — :class:`BandDetector`, the position/angle search that
  reads the colour run order off the sheet at any rotation;
* :mod:`vision.frame_source` — webcam *and* ROS 2 image-topic backends;
* :mod:`vision.missions` — ordered colour -> French action mapping;
* :mod:`vision.tts` — French text-to-speech with backend fallback;
* :mod:`vision.pipeline` — ties detection + mission + speech together.
"""

from vision.detector import BandDetector, BandDetectionResult, DetectionParams
from vision.frame_source import (
    FrameSource,
    add_source_args,
    open_checked,
    open_frame_source,
)
from vision.missions import (
    DEFAULT_ACTIONS_FR,
    Mission,
    MissionStep,
    build_mission,
    load_action_map,
)
from vision.pipeline import MissionObservation, MissionPipeline
from vision.tts import FrenchTTS, available_backends

__all__ = [
    "BandDetector",
    "BandDetectionResult",
    "DetectionParams",
    "FrameSource",
    "add_source_args",
    "open_checked",
    "open_frame_source",
    "DEFAULT_ACTIONS_FR",
    "Mission",
    "MissionStep",
    "build_mission",
    "load_action_map",
    "MissionObservation",
    "MissionPipeline",
    "FrenchTTS",
    "available_backends",
]
