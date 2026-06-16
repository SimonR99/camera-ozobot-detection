#!/usr/bin/env python3
"""Read a white sheet, build a mission from its colours, and speak it in French.

Frame sources (shared with the rest of the project):

    # Local webcam / RealSense colour device
    python -m vision.run --camera 4 --calibration calibration.json

    # Shared via the realsense2_camera ROS 2 node
    python -m vision.run --ros-topic /camera/color/image_raw

Modes:
    (default)   live preview; auto-speaks each new mission once it is stable
    --once      grab one frame, speak the mission, exit (no window)
    --image P   read a still image instead of a camera (great for image.png)

Each detected colour, in left-to-right / top-to-bottom order, maps to one French
correction action (see vision/missions.py and vision/actions.fr.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow `python vision/run.py` as well as `python -m vision.run`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.frame_source import add_source_args, open_checked  # noqa: E402

from vision.pipeline import MissionPipeline  # noqa: E402
from vision.tts import FrenchTTS, available_backends  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read colour strips on a white sheet and speak the mission in French",
    )
    add_source_args(parser)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
        help="Calibration file with colour ranges (and optional actions block)",
    )
    parser.add_argument(
        "--actions",
        type=Path,
        default=None,
        help="JSON file overriding the colour -> French action map",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Read a still image instead of a camera (e.g. image.png)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Grab a single frame, speak the mission, then exit (no window)",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=6,
        help="Frames a mission must persist before it is spoken (live mode)",
    )
    parser.add_argument(
        "--steps",
        action="store_true",
        help="Speak each action separately instead of one mission sentence",
    )
    parser.add_argument(
        "--tts-backend",
        default="auto",
        help="Force a TTS backend (auto/spd-say/espeak-ng/espeak/pyttsx3/gtts/print)",
    )
    parser.add_argument(
        "--tts-rate",
        type=int,
        default=None,
        help="Speech rate passed to the CLI backends (backend-specific scale)",
    )
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        help="Kokoro speech speed multiplier (1.0 = normal)",
    )
    parser.add_argument(
        "--tts-voice",
        default=None,
        help="Kokoro voice name (default ff_siwis, the French voice)",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable audio; missions are still printed",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Write the detected mission as JSON to this path",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live preview window (live mode)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Resize frames to this width before processing (0 = keep original)",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable the default 180-degree frame rotation",
    )
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> MissionPipeline:
    tts = FrenchTTS(
        backend=args.tts_backend,
        rate=args.tts_rate,
        enabled=not args.no_tts,
        voice=args.tts_voice,
        speed=args.tts_speed,
    )
    cal = args.calibration if args.calibration.exists() else None
    return MissionPipeline(
        calibration_path=cal,
        actions_path=args.actions,
        tts=tts,
        stable_frames=args.stable_frames,
    )


def preprocess(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if not args.no_flip:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if args.width > 0:
        h, w = frame.shape[:2]
        long_side = max(h, w)
        if long_side > args.width:
            scale = args.width / long_side
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def report(obs, pipeline: MissionPipeline, save_json: Path = None) -> None:
    mission = obs.mission
    if obs.detected:
        print(f"Mission: {'-'.join(mission.colors)}")
        for step in mission.steps:
            print(f"  {step.index}. {step.color:8s} -> {step.phrase()}")
    else:
        print("No mission detected on the sheet.")
    if save_json is not None:
        record = mission.to_dict()
        record["band_detected"] = obs.result.band_detected
        record["confidence"] = round(obs.result.confidence, 4)
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(record, indent=2))
        print(f"Saved mission JSON: {save_json}")


def run_image(pipeline: MissionPipeline, args: argparse.Namespace) -> int:
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")
    frame = preprocess(frame, args)
    obs = pipeline.process(frame)
    report(obs, pipeline, args.save_json)
    if obs.detected:
        pipeline.narrate(obs.mission, full=not args.steps)
    return 0 if obs.detected else 1


def run_once(pipeline: MissionPipeline, args: argparse.Namespace) -> int:
    cap = open_checked(args)
    try:
        ret, frame = cap.read()
        if not ret:
            raise SystemExit("Failed to read a frame from the source")
        frame = preprocess(frame, args)
        obs = pipeline.process(frame)
        report(obs, pipeline, args.save_json)
        if obs.detected:
            pipeline.narrate(obs.mission, full=not args.steps)
        return 0 if obs.detected else 1
    finally:
        cap.release()


def run_live(pipeline: MissionPipeline, args: argparse.Namespace) -> int:
    cap = open_checked(args)
    window = "Vision Mission (q=quit, s=speak, r=reset)"
    show = not args.no_preview
    if show:
        cv2.namedWindow(window)
    print(f"{pipeline.tts.describe()} | {pipeline.detector.calibration_path or 'defaults'}")
    print("Press q to quit, s to re-speak the current mission, r to reset.")
    pipeline.tts.warm_up()  # load the neural model now, not on first mission
    last_code = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = preprocess(frame, args)
            obs = pipeline.update(frame, full=not args.steps)

            code = "-".join(obs.mission.colors) if obs.detected else None
            if code != last_code:
                report(obs, pipeline, None)
                last_code = code

            if show:
                display = pipeline.detector.draw_debug(frame, obs.result)
                _overlay_mission(display, obs.mission)
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s") and obs.detected:
                    pipeline.narrate(obs.mission, full=not args.steps)
                if key == ord("r"):
                    pipeline.reset()
                    last_code = None
        if args.save_json is not None:
            report(obs, pipeline, args.save_json)
        return 0
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


def _overlay_mission(display, mission) -> None:
    if mission.is_empty:
        return
    for i, step in enumerate(mission.steps):
        text = f"{step.index}. {step.color} -> {step.phrase()}"
        cv2.putText(
            display,
            text,
            (10, 90 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )


def main() -> None:
    args = parse_args()
    if args.tts_backend not in ("auto",) + tuple(available_backends()):
        print(
            f"Note: backend {args.tts_backend!r} unavailable; "
            f"available: {', '.join(available_backends())}",
            file=sys.stderr,
        )
    pipeline = build_pipeline(args)

    if args.image is not None:
        raise SystemExit(run_image(pipeline, args))
    if args.once:
        raise SystemExit(run_once(pipeline, args))
    raise SystemExit(run_live(pipeline, args))


if __name__ == "__main__":
    main()
