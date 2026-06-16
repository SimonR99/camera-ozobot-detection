#!/usr/bin/env python3
"""Present a sheet, decode its colours, and drive the Unitree G1 accordingly.

This is the bridge between the two packages:

    vision  (decode the colour order off a white sheet)  ->
    motion  (execute the move for each colour on the G1)

Colour -> motion (in the order the colours appear on the sheet):

    green   -> walk forward 1 m
    blue    -> turn +45 deg (left)
    yellow  -> turn -45 deg (right)
    orange  -> wave

Interpreter split (important): vision needs OpenCV from the project ``.venv``,
while the Unitree SDK is installed only for the system ``/usr/bin/python3`` (see
motion/README.md). So this script runs under ``.venv`` and invokes motion through
its CLI as a subprocess::

    /usr/bin/python3 -m motion --yes --iface eth0 walk 1.0

Safety: the robot only moves with ``--execute``. Without it the controller is a
dry run that prints/speaks the plan but sends no motion commands.

    # Dry run on the sample sheet (no robot, no SDK needed)
    python -m vision.controller --image image.png

    # Live, really drive the robot
    python -m vision.controller --camera 4 --execute --iface eth0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.frame_source import add_source_args, open_checked  # noqa: E402
from vision.missions import Mission, load_action_map  # noqa: E402
from vision.pipeline import MissionPipeline  # noqa: E402
from vision.tts import FrenchTTS  # noqa: E402

# Colour -> ("motion subcommand", argument). ``None`` argument = no positional
# (wave). Order is taken from the sheet, not this dict.
COLOR_MOTIONS: dict = {
    "green": ("walk", "1.0"),    # forward 1 m
    "blue": ("turn", "45"),      # +45 deg (left / CCW)
    "yellow": ("turn", "-45"),   # -45 deg (right / CW)
    "orange": ("wave", None),    # arm gesture
}


class MotionExecutor:
    """Run motion commands by shelling out to the ``motion`` CLI.

    Kept as a subprocess on purpose: the Unitree SDK lives in the system
    interpreter, this controller in the ``.venv``. ``execute=False`` is a dry run
    that only logs the commands (safe default — the robot does not move).
    """

    def __init__(
        self,
        python_bin: str = "/usr/bin/python3",
        iface: str = "eth0",
        execute: bool = False,
        timeout: float = 60.0,
    ):
        self.python_bin = python_bin
        self.iface = iface
        self.execute = execute
        self.timeout = timeout

    def _command(self, subcmd: str, arg: Optional[str]) -> List[str]:
        # The motion CLI's --yes/--iface are parent options: argparse requires
        # them *before* the subcommand, not after.
        cmd = [self.python_bin, "-m", "motion", "--yes", "--iface", self.iface, subcmd]
        if arg is not None:
            cmd.append(arg)
        return cmd

    def run(self, color: str) -> bool:
        """Execute the motion mapped to ``color``. Returns True on success."""
        plan = COLOR_MOTIONS.get(color)
        if plan is None:
            print(f"  · {color}: no motion mapped, skipping")
            return False
        subcmd, arg = plan
        cmd = self._command(subcmd, arg)
        pretty = subcmd + (f" {arg}" if arg else "")
        if not self.execute:
            print(f"  · {color}: DRY RUN -> motion {pretty}")
            return True
        print(f"  · {color}: motion {pretty}  ({' '.join(cmd)})")
        try:
            result = subprocess.run(cmd, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            print(f"    ! motion {pretty} timed out after {self.timeout:.0f}s")
            return False
        except FileNotFoundError:
            print(f"    ! cannot run {self.python_bin!r}; is the system interpreter correct?")
            return False
        if result.returncode != 0:
            print(f"    ! motion {pretty} exited with code {result.returncode}")
            return False
        return True

    def stop(self) -> None:
        if not self.execute:
            return
        try:
            subprocess.run(self._command("stop", None), timeout=self.timeout)
        except Exception:
            pass


def execute_mission(
    mission: Mission,
    executor: MotionExecutor,
    tts: FrenchTTS,
    action_map: dict,
) -> None:
    """Speak each French action and run its motion, in the sheet's colour order."""
    print(f"Executing mission: {'-'.join(mission.colors)}")
    for step in mission.steps:
        phrase = action_map.get(step.color, step.phrase())
        print(f"  {step.index}. {step.color} -> {phrase}")
        tts.say(phrase)
        executor.run(step.color)
    print("Mission complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode a colour sheet and drive the Unitree G1 accordingly",
    )
    add_source_args(parser)
    parser.add_argument("--calibration", type=Path, default=Path("calibration.json"))
    parser.add_argument("--actions", type=Path, default=None,
                        help="Override colour -> French action phrasing")
    parser.add_argument("--image", type=Path, default=None,
                        help="Decode a still image instead of a camera")
    parser.add_argument("--once", action="store_true",
                        help="Grab one frame, run the mission, exit")
    parser.add_argument("--execute", action="store_true",
                        help="Actually drive the robot (default: dry run)")
    parser.add_argument("--iface", default="eth0", help="Robot network interface")
    parser.add_argument("--python", dest="python_bin", default="/usr/bin/python3",
                        help="Interpreter with the Unitree SDK (default /usr/bin/python3)")
    parser.add_argument("--stable-frames", type=int, default=6,
                        help="Frames a sheet must persist before it triggers (live)")
    parser.add_argument("--no-tts", action="store_true", help="Disable French speech")
    parser.add_argument("--tts-backend", default="auto")
    parser.add_argument("--tts-speed", type=float, default=1.0,
                        help="Kokoro speech speed multiplier (1.0 = normal)")
    parser.add_argument("--tts-voice", default=None,
                        help="Kokoro voice name (default ff_siwis)")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def build(args: argparse.Namespace) -> Tuple[MissionPipeline, MotionExecutor, FrenchTTS, dict]:
    cal = args.calibration if args.calibration.exists() else None
    tts = FrenchTTS(
        backend=args.tts_backend,
        enabled=not args.no_tts,
        voice=args.tts_voice,
        speed=args.tts_speed,
    )
    tts.warm_up()  # preload the neural model so the first action speaks instantly
    pipeline = MissionPipeline(
        calibration_path=cal,
        actions_path=args.actions,
        tts=tts,
        stable_frames=args.stable_frames,
    )
    executor = MotionExecutor(
        python_bin=args.python_bin, iface=args.iface, execute=args.execute
    )
    action_map = load_action_map(cal, args.actions)
    return pipeline, executor, tts, action_map


def run_single(frame, pipeline, executor, tts, action_map) -> int:
    obs = pipeline.process(frame)
    if not obs.detected:
        print("No colour mission detected on the sheet.")
        return 1
    execute_mission(obs.mission, executor, tts, action_map)
    return 0


def run_live(args, pipeline, executor, tts, action_map) -> int:
    cap = open_checked(args)
    window = "Sheet -> G1 controller (q=quit)"
    show = not args.no_preview
    if show:
        cv2.namedWindow(window)
    mode = "EXECUTE" if executor.execute else "DRY RUN"
    print(f"[{mode}] {tts.describe()} | iface={executor.iface}")
    print("Present a sheet to the camera. Press q to quit.")
    armed = True  # require the sheet to clear before re-triggering
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            obs = pipeline.process(frame)

            if show:
                display = pipeline.detector.draw_debug(frame, obs.result)
                cv2.imshow(window, display)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if obs.detected:
                if armed:
                    # Confirm stability before committing the robot to move.
                    if _confirm_stable(cap, pipeline, obs, args.stable_frames):
                        execute_mission(obs.mission, executor, tts, action_map)
                        armed = False
            else:
                armed = True
    finally:
        executor.stop()
        cap.release()
        if show:
            cv2.destroyAllWindows()
    return 0


def _confirm_stable(cap, pipeline, first_obs, stable_frames: int) -> bool:
    """Require the same colour code across ``stable_frames`` reads before moving."""
    code = "-".join(first_obs.mission.colors)
    count = 1
    while count < stable_frames:
        ret, frame = cap.read()
        if not ret:
            return False
        obs = pipeline.process(frame)
        if obs.detected and "-".join(obs.mission.colors) == code:
            count += 1
        else:
            return False
    return True


def main() -> None:
    args = parse_args()
    pipeline, executor, tts, action_map = build(args)

    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"Could not read image: {args.image}")
        raise SystemExit(run_single(frame, pipeline, executor, tts, action_map))

    if args.once:
        cap = open_checked(args)
        try:
            ret, frame = cap.read()
            if not ret:
                raise SystemExit("Failed to read a frame from the source")
            raise SystemExit(run_single(frame, pipeline, executor, tts, action_map))
        finally:
            cap.release()

    raise SystemExit(run_live(args, pipeline, executor, tts, action_map))


if __name__ == "__main__":
    main()
