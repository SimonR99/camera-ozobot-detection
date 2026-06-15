#!/usr/bin/env python3
"""Interactive combination manager and live viewer.

A *combination* is a named group of saved colors that, seen together on the
floor, forms a recognized block (think of three tape stripes making an Ozobot
code). Colors are matched as an unordered set: the order the camera reads them
in does not matter.

Workflow:
  1. Pick the saved colors that make up the combination (click a swatch or press
     its number key) — they appear in the "selecting" row.
  2. Press N, type a name, Enter to save the combination.
  3. The live banner shows which saved combination the camera currently matches.

Keys:
  1-9        - toggle the Nth saved color in/out of the current selection
  click      - toggle the clicked color swatch
  n          - name + save the current selection as a combination (needs >= 2)
  c          - clear the current selection
  d          - delete a saved combination by name
  Esc        - cancel name/delete entry (or clear selection in view mode)
  q          - quit

Colors come from your calibration file; calibrate them first with
scripts/calibrate.py. This script never modifies colors, only combinations.
"""

import argparse
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ozobot_bands.color_library import (
    ColorLibrary,
    Combination,
    load_color_library,
    match_combinations,
    save_color_library,
)
from ozobot_bands.color_library import preview_bgr_for_entry
from ozobot_bands.detector import BandDetector

DEFAULT_COMBO_SIZE = 3
SWATCH_W = 150
SWATCH_H = 34


class Mode(str, Enum):
    VIEW = "view"
    NAME = "name"
    DELETE = "delete"


def toggle_selection(selection: List[str], name: str) -> List[str]:
    """Add `name` to the selection, or remove it if already present.

    Returns a new list; order reflects insertion so the UI is stable, but the
    combination itself is order-independent.
    """
    if name in selection:
        return [n for n in selection if n != name]
    return selection + [name]


def valid_combination_name(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_-]+$", name))


def selection_to_combination(name: str, selection: List[str]) -> Combination:
    return Combination(name=name, colors=list(selection))


@dataclass
class ManagerState:
    mode: Mode = Mode.VIEW
    text_input: str = ""
    selection: List[str] = field(default_factory=list)
    swatches: List[Tuple[str, int, int, int, int]] = field(default_factory=list)
    mouse: Dict[str, Any] = field(default_factory=lambda: {"click_xy": None})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combination manager / live viewer")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
        help="Color library file to read colors from and write combinations to",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Manage combinations without a live camera feed",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=10,
        help="Run detection every N frames for the live match banner",
    )
    return parser.parse_args()


def on_mouse(event: int, x: int, y: int, _flags: int, mouse: Dict[str, Any]) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse["click_xy"] = (x, y)


def color_at_click(
    swatches: List[Tuple[str, int, int, int, int]], x: int, y: int
) -> Optional[str]:
    for name, sx, sy, sw, sh in swatches:
        if sx <= x <= sx + sw and sy <= y <= sy + sh:
            return name
    return None


def persist(library: ColorLibrary, path: Path) -> None:
    save_color_library(path, library)


def build_detector(path: Path) -> Optional[BandDetector]:
    if not path.exists():
        return None
    return BandDetector(calibration_path=path)


def draw_swatches(
    display: np.ndarray,
    library: ColorLibrary,
    selection: List[str],
) -> List[Tuple[str, int, int, int, int]]:
    """Draw the numbered, clickable color legend; return hitboxes."""
    swatches: List[Tuple[str, int, int, int, int]] = []
    x0 = 10
    y = 80
    for idx, name in enumerate(library.names()):
        entry = library.colors[name]
        tint = preview_bgr_for_entry(entry)
        selected = name in selection
        cv2.rectangle(display, (x0, y), (x0 + SWATCH_W, y + SWATCH_H), tint, -1)
        border = (0, 255, 255) if selected else (40, 40, 40)
        cv2.rectangle(display, (x0, y), (x0 + SWATCH_W, y + SWATCH_H), border, 3 if selected else 1)
        key_hint = f"{idx + 1}." if idx < 9 else "  "
        cv2.putText(
            display,
            f"{key_hint} {name}",
            (x0 + 6, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )
        swatches.append((name, x0, y, SWATCH_W, SWATCH_H))
        y += SWATCH_H + 6
    return swatches


def draw_manager_ui(
    frame_bgr: np.ndarray,
    library: ColorLibrary,
    state: ManagerState,
    match_names: List[str],
    path: Path,
) -> Tuple[np.ndarray, List[Tuple[str, int, int, int, int]]]:
    display = frame_bgr.copy()
    h, w = display.shape[:2]

    banners = {
        Mode.VIEW: "Pick colors (click / 1-9) | N name+save | C clear | D delete combo | Q quit",
        Mode.NAME: "TYPE combination name | Enter save | Esc cancel",
        Mode.DELETE: "TYPE combination name to delete | Enter confirm | Esc cancel",
    }
    cv2.putText(
        display, banners[state.mode], (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )

    selecting = ", ".join(state.selection) if state.selection else "(none)"
    cv2.putText(
        display, f"selecting: {selecting}", (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
    )

    swatches = draw_swatches(display, library, state.selection)

    if state.mode in (Mode.NAME, Mode.DELETE):
        prompt = "Name:" if state.mode == Mode.NAME else "Delete:"
        cv2.rectangle(display, (8, 58), (w - 8, 92), (40, 40, 40), -1)
        cv2.putText(
            display, f"{prompt} {state.text_input}_", (14, 82),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )

    # Saved combinations panel (right side).
    cx = max(10, w - 320)
    cy = 80
    cv2.putText(
        display, "Combinations:", (cx, cy - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )
    for name in library.combination_names():
        combo = library.combinations[name]
        hit = name in match_names
        color = (0, 255, 0) if hit else (200, 200, 200)
        cv2.putText(
            display, f"{name}: {', '.join(combo.colors)}", (cx, cy + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if hit else 1,
        )
        cy += 24

    banner = (
        "MATCH: " + ", ".join(match_names) if match_names else "no combination match"
    )
    cv2.putText(
        display, banner, (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 255, 0) if match_names else (180, 180, 180), 2,
    )
    return display, swatches


def handle_text_key(key: int, state: ManagerState) -> bool:
    """Handle char input for name/delete modes. Returns True if consumed."""
    if key == 27:  # Esc
        state.mode = Mode.VIEW
        state.text_input = ""
        return True
    if key in (8, 127):  # backspace
        state.text_input = state.text_input[:-1]
        return True
    if key == 13:  # Enter — caller handles submit
        return False
    if 32 <= key <= 126:
        state.text_input += chr(key)
        return True
    return False


def save_selection(state: ManagerState, library: ColorLibrary, path: Path) -> bool:
    name = state.text_input.strip()
    if not valid_combination_name(name):
        print("Name must be letters, numbers, underscore, hyphen only")
        return False
    if len(state.selection) < 2:
        print("A combination needs at least 2 colors")
        return False
    library.upsert_combination(selection_to_combination(name, state.selection))
    persist(library, path)
    print(f"Saved combination '{name}' = {state.selection} -> {path}")
    return True


def toggle_color_into_selection(state: ManagerState, name: str) -> None:
    state.selection = toggle_selection(state.selection, name)


def run(args: argparse.Namespace) -> None:
    path = args.calibration
    if path.exists():
        library = load_color_library(path)
        print(f"Loaded {path}: {len(library.colors)} colors, "
              f"{len(library.combinations)} combinations")
    else:
        raise SystemExit(
            f"No calibration file at {path}. Calibrate colors first "
            f"(scripts/calibrate.py)."
        )

    if not library.colors:
        raise SystemExit("No colors in the library yet — calibrate some first.")

    state = ManagerState()
    detector = build_detector(path)
    match_names: List[str] = []
    frame_idx = 0

    cap = None
    if not args.no_camera:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Camera {args.camera} unavailable — running without live view")
            cap = None

    window = "Combination Manager"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse, state.mouse)
    print(__doc__)

    blank = np.full((480, 960, 3), 30, dtype=np.uint8)

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = blank.copy()

        if detector is not None and cap is not None and frame_idx % max(1, args.detect_every) == 0:
            result = detector.detect(frame)
            match_names = result.matched_combinations
        frame_idx += 1

        # Handle a pending click against the swatch hitboxes from last render.
        click_xy = state.mouse.get("click_xy")
        if click_xy and state.mode == Mode.VIEW:
            state.mouse["click_xy"] = None
            name = color_at_click(state.swatches, *click_xy)
            if name:
                toggle_color_into_selection(state, name)

        display, swatches = draw_manager_ui(frame, library, state, match_names, path)
        state.swatches = swatches
        cv2.imshow(window, display)

        key = cv2.waitKey(1)
        if key == -1:
            continue
        key &= 0xFF

        if state.mode in (Mode.NAME, Mode.DELETE):
            if handle_text_key(key, state):
                continue
            if key == 13:
                if state.mode == Mode.NAME:
                    if save_selection(state, library, path):
                        state.selection = []
                        state.mode = Mode.VIEW
                        state.text_input = ""
                else:  # DELETE
                    name = state.text_input.strip()
                    if library.remove_combination(name):
                        persist(library, path)
                        print(f"Deleted combination '{name}'")
                    else:
                        print(f"Combination '{name}' not found")
                    state.mode = Mode.VIEW
                    state.text_input = ""
            continue

        if key == ord("q"):
            break
        if key == 27:
            state.selection = []
            continue
        if key == ord("c"):
            state.selection = []
        elif key == ord("n"):
            if len(state.selection) >= 2:
                state.mode = Mode.NAME
                state.text_input = ""
                print("Type a name for this combination, then Enter")
            else:
                print("Select at least 2 colors before naming a combination")
        elif key == ord("d"):
            state.mode = Mode.DELETE
            state.text_input = ""
            print("Type a combination name to delete, then Enter")
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            names = library.names()
            if idx < len(names):
                toggle_color_into_selection(state, names[idx])
        # Rebuild detector so the live viewer reflects new combinations.
        detector = build_detector(path)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
