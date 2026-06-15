#!/usr/bin/env python3
"""Interactive color calibration with click → adjust → name workflow.

Process:
  1. Click a color anywhere on the live camera image
  2. Narrow/widen the selection with +/- keys (live preview)
  3. Press V to validate → type a name tag → Enter to save
  4. Press P to paste a color (#RRGGBB or r,g,b) and create a new entry

Saved colors are written to JSON and highlighted in real time.

Keys:
  click     - pick color (starts adjust mode)
  +/-       - narrow/widen range (in adjust mode)
  v         - validate → enter name
  p         - paste new color from hex/rgb string
  d         - delete a saved color by name
  Esc       - cancel current step
  q         - quit
"""

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import cv2

from ozobot_bands.calibration import hsv_range_from_samples, sample_region_hsv
from ozobot_bands.color_library import (
    ColorLibrary,
    NamedColor,
    load_color_library,
    parse_color_string,
    save_color_library,
)
from ozobot_bands.colors import HSVRange
from ozobot_bands.detector import DetectionParams
from ozobot_bands.frame_source import add_source_args, open_checked
from ozobot_bands.visualization import (
    draw_calibration_ui,
    draw_named_color_highlights,
    draw_pending_highlight,
)


class Mode(str, Enum):
    VIEW = "view"
    ADJUST = "adjust"
    NAME = "name"
    PASTE = "paste"
    DELETE = "delete"


@dataclass
class PendingSelection:
    samples: List[Tuple[int, int, int]]
    click_point: Tuple[int, int]
    reference_bgr: Optional[Tuple[int, int, int]] = None
    hue_padding: int = 8
    sv_padding: int = 40

    def range(self) -> HSVRange:
        return hsv_range_from_samples(
            self.samples,
            hue_padding=self.hue_padding,
            sv_padding=self.sv_padding,
        )


@dataclass
class CalibrateState:
    mode: Mode = Mode.VIEW
    text_input: str = ""
    pending: Optional[PendingSelection] = None
    mouse: Dict[str, Any] = field(default_factory=lambda: {"click_xy": None})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Named color calibration UI")
    add_source_args(parser)
    parser.add_argument("--output", type=Path, default=Path("calibration.json"))
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--highlight-alpha", type=float, default=0.45)
    parser.add_argument("--sample-radius", type=int, default=20)
    return parser.parse_args()


def on_mouse(event: int, x: int, y: int, _flags: int, mouse: Dict[str, Any]) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse["click_xy"] = (x, y)


def start_click_selection(
    frame,
    x: int,
    y: int,
    sample_radius: int,
    hue_padding: int,
    sv_padding: int,
) -> PendingSelection:
    samples = sample_region_hsv(frame, (x, y), sample_radius)
    bgr = tuple(int(c) for c in frame[y, x])
    return PendingSelection(
        samples=samples,
        click_point=(x, y),
        reference_bgr=bgr,
        hue_padding=hue_padding,
        sv_padding=sv_padding,
    )


def start_paste_selection(
    bgr: Tuple[int, int, int],
    hue_padding: int,
    sv_padding: int,
) -> PendingSelection:
    pixel = np.array([[list(bgr)]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    samples = [(int(hsv[0]), int(hsv[1]), int(hsv[2]))]
    return PendingSelection(
        samples=samples,
        click_point=None,
        reference_bgr=bgr,
        hue_padding=hue_padding,
        sv_padding=sv_padding,
    )


def pending_from_entry(
    entry: NamedColor,
    hue_padding: int,
    sv_padding: int,
) -> PendingSelection:
    if entry.reference_bgr is not None:
        return start_paste_selection(entry.reference_bgr, hue_padding, sv_padding)
    mid_h = (entry.hsv_range.h_min + entry.hsv_range.h_max) // 2
    mid_s = (entry.hsv_range.s_min + entry.hsv_range.s_max) // 2
    mid_v = (entry.hsv_range.v_min + entry.hsv_range.v_max) // 2
    return PendingSelection(
        samples=[(mid_h, mid_s, mid_v)],
        click_point=entry.sample_point,
        reference_bgr=entry.reference_bgr,
        hue_padding=hue_padding,
        sv_padding=sv_padding,
    )


def persist(library: ColorLibrary, path: Path, params: DetectionParams) -> None:
    library.detection = {
        "min_segment_width_px": params.min_segment_width_px,
        "scan_strip_height_ratio": params.scan_strip_height_ratio,
        "min_band_colors": params.min_band_colors,
        "require_black_separators": params.require_black_separators,
        "roi_y_center_ratio": params.roi_y_center_ratio,
        "roi_width_ratio": params.roi_width_ratio,
    }
    save_color_library(path, library)


def commit_named_color(
    library: ColorLibrary,
    name: str,
    pending: PendingSelection,
    path: Path,
    params: DetectionParams,
) -> None:
    name = name.strip()
    if not name:
        print("Name cannot be empty")
        return
    if not re_valid_name(name):
        print("Name must be letters, numbers, underscore, hyphen only")
        return

    entry = NamedColor(
        name=name,
        hsv_range=pending.range(),
        sample_point=pending.click_point,
        reference_bgr=pending.reference_bgr,
    )
    library.upsert(entry)
    persist(library, path, params)
    print(f"Saved color '{name}' -> {path}")


def re_valid_name(name: str) -> bool:
    import re

    return bool(re.match(r"^[A-Za-z0-9_-]+$", name))


def handle_text_key(key: int, state: CalibrateState) -> bool:
    """Handle character input for name/paste/delete modes. Returns True if handled."""
    if key == 27:  # Esc
        state.mode = Mode.VIEW
        state.text_input = ""
        state.pending = None
        print("Cancelled")
        return True
    if key in (8, 127):  # backspace
        state.text_input = state.text_input[:-1]
        return True
    if key == 13:  # Enter
        return False  # let caller handle submit
    if 32 <= key <= 126:
        state.text_input += chr(key)
        return True
    return False


def main() -> None:
    args = parse_args()
    load_path = args.load or (args.output if args.output.exists() else None)

    if load_path and load_path.exists():
        library = load_color_library(load_path)
        params = DetectionParams.from_dict(library.detection)
        print(f"Loaded {load_path} ({len(library.colors)} colors)")
    else:
        library = ColorLibrary()
        params = DetectionParams()

    default_h = 8
    default_sv = 40
    state = CalibrateState()

    cap = open_checked(args)

    window = "Color Calibration"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse, state.mouse)

    print(__doc__)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mouse click in VIEW or ADJUST (restart selection)
        click_xy = state.mouse.get("click_xy")
        if click_xy and state.mode in (Mode.VIEW, Mode.ADJUST):
            state.mouse["click_xy"] = None
            x, y = click_xy
            state.pending = start_click_selection(
                frame, x, y, args.sample_radius, default_h, default_sv
            )
            state.mode = Mode.ADJUST
            print(f"Adjust selection at ({x}, {y}) — use +/- then V to validate")

        display = draw_named_color_highlights(
            frame, library, alpha=args.highlight_alpha
        )

        pending_point = None
        if state.pending and state.mode == Mode.ADJUST:
            display = draw_pending_highlight(display, state.pending.range())
            pending_point = state.pending.click_point

        display = draw_calibration_ui(
            display,
            library,
            state.mode.value,
            state.text_input,
            state.pending.hue_padding if state.pending else default_h,
            state.pending.sv_padding if state.pending else default_sv,
            pending_point,
            args.sample_radius,
            str(args.output),
        )

        cv2.imshow(window, display)
        key = cv2.waitKey(1)
        if key == -1:
            continue
        key &= 0xFF

        if key == ord("q"):
            break

        if state.mode in (Mode.NAME, Mode.PASTE, Mode.DELETE):
            if handle_text_key(key, state):
                continue
            if key == 13:
                if state.mode == Mode.NAME and state.pending:
                    commit_named_color(
                        library, state.text_input, state.pending, args.output, params
                    )
                    state.mode = Mode.VIEW
                    state.text_input = ""
                    state.pending = None
                elif state.mode == Mode.PASTE:
                    text = state.text_input.strip()
                    try:
                        if text in library.colors:
                            state.pending = pending_from_entry(
                                library.colors[text], default_h, default_sv
                            )
                            state.mode = Mode.ADJUST
                            state.text_input = ""
                            print(f"Copied '{text}' — adjust with +/- then V")
                        else:
                            bgr = parse_color_string(text)
                            state.pending = start_paste_selection(
                                bgr, default_h, default_sv
                            )
                            state.mode = Mode.ADJUST
                            state.text_input = ""
                            print(f"Pasted BGR{bgr} — adjust with +/- then V")
                    except ValueError as exc:
                        print(exc)
                elif state.mode == Mode.DELETE:
                    name = state.text_input.strip()
                    if library.remove(name):
                        persist(library, args.output, params)
                        print(f"Deleted '{name}'")
                    else:
                        print(f"Color '{name}' not found")
                    state.mode = Mode.VIEW
                    state.text_input = ""
            continue

        if key == 27:
            state.mode = Mode.VIEW
            state.pending = None
            state.text_input = ""
            print("Cancelled")
            continue

        if state.mode == Mode.VIEW:
            if key == ord("p"):
                state.mode = Mode.PASTE
                state.text_input = ""
                print("Paste mode: type #RRGGBB or r,g,b then Enter")
            elif key == ord("d"):
                state.mode = Mode.DELETE
                state.text_input = ""
                print("Delete mode: type color name then Enter")
            continue

        if state.mode == Mode.ADJUST and state.pending:
            if key == ord("+") or key == ord("="):
                state.pending.hue_padding = min(30, state.pending.hue_padding + 2)
                state.pending.sv_padding = min(80, state.pending.sv_padding + 5)
            elif key == ord("-"):
                state.pending.hue_padding = max(2, state.pending.hue_padding - 2)
                state.pending.sv_padding = max(10, state.pending.sv_padding - 5)
            elif key == ord("v"):
                state.mode = Mode.NAME
                state.text_input = ""
                print("Enter name tag for this color, then press Enter")
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
