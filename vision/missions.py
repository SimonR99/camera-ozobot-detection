"""Turn an ordered colour sequence into a mission (one action per colour).

The :class:`vision.detector.BandDetector` reads the colours along a scan line in
order (``BandDetectionResult.colors_sequence``). This module maps each colour, in
that same order, to a *correction action* phrased in French — producing a
:class:`Mission` the TTS layer can read aloud.

The colour -> action map is, in priority order:

1. an explicit override file passed to :func:`load_action_map`;
2. an ``"actions"`` block inside the calibration JSON (so actions live next to
   the colours they refer to);
3. the built-in French defaults below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# The separator colour carries no action — it just delimits tape strips.
SEPARATOR_NAME = "black"
UNKNOWN_NAME = "unknown"

# Default correction actions in French, keyed by calibrated colour name. These
# are phrased to match the motions executed by ``motion.controller`` (green =
# forward 1 m, blue = +45 deg, orange = wave, yellow = -45 deg). Override
# per-deployment via an actions file or an "actions" block in the calibration
# JSON. ``red`` is kept as a safe stop default.
DEFAULT_ACTIONS_FR: Dict[str, str] = {
    "green": "Avance d'un mètre",
    "blue": "Tourne de quarante-cinq degrés",
    "orange": "Salue de la main",
    "yellow": "Tourne de moins quarante-cinq degrés",
    "red": "Arrête-toi",
}


@dataclass
class MissionStep:
    """One colour in the detected order and the action it maps to."""

    index: int
    color: str
    action: Optional[str]

    @property
    def known(self) -> bool:
        return self.action is not None

    def phrase(self) -> str:
        """Spoken French phrase for this step (falls back when unmapped)."""
        if self.action:
            return self.action
        return f"Couleur inconnue : {self.color}"


@dataclass
class Mission:
    """An ordered list of actions derived from a detected colour sequence."""

    steps: List[MissionStep] = field(default_factory=list)

    @property
    def colors(self) -> List[str]:
        return [step.color for step in self.steps]

    @property
    def actions(self) -> List[str]:
        """French action phrases, in order (includes fallbacks for unknowns)."""
        return [step.phrase() for step in self.steps]

    @property
    def is_empty(self) -> bool:
        return not self.steps

    @property
    def has_unknown(self) -> bool:
        return any(not step.known for step in self.steps)

    def narration(self, intro: bool = True) -> str:
        """Full French sentence reading out the whole mission, step by step."""
        if self.is_empty:
            return "Aucune mission détectée."
        parts = []
        if intro:
            count = len(self.steps)
            word = "étape" if count == 1 else "étapes"
            parts.append(f"Mission détectée, {count} {word}.")
        for step in self.steps:
            parts.append(f"Étape {step.index}, {step.phrase()}.")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "colors": self.colors,
            "color_code": "-".join(self.colors),
            "steps": [
                {"index": s.index, "color": s.color, "action": s.action}
                for s in self.steps
            ],
            "narration": self.narration(),
            "has_unknown": self.has_unknown,
        }


def load_action_map(
    calibration_path: Optional[Path] = None,
    override_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Build the colour -> French action map from defaults + JSON sources.

    Later sources win: defaults are overlaid by an ``"actions"`` block in the
    calibration file, then by a dedicated override file. An override file may be
    either a flat ``{"colour": "action"}`` object or one wrapped as
    ``{"actions": {...}}``.
    """
    actions: Dict[str, str] = dict(DEFAULT_ACTIONS_FR)

    if calibration_path and Path(calibration_path).exists():
        payload = json.loads(Path(calibration_path).read_text())
        block = payload.get("actions")
        if isinstance(block, dict):
            actions.update({str(k): str(v) for k, v in block.items()})

    if override_path and Path(override_path).exists():
        payload = json.loads(Path(override_path).read_text())
        block = payload.get("actions", payload)
        if isinstance(block, dict):
            actions.update({str(k): str(v) for k, v in block.items()})

    return actions


def build_mission(
    colors_sequence: List[str],
    action_map: Dict[str, str],
    drop_separators: bool = True,
) -> Mission:
    """Build an ordered :class:`Mission` from a detected colour sequence.

    The sequence order is preserved, including repeated colours (each occurrence
    becomes its own step). ``black`` separators and ``unknown`` labels are
    dropped by default. A colour with no entry in ``action_map`` still becomes a
    step, but with ``action=None`` (spoken as "couleur inconnue").
    """
    steps: List[MissionStep] = []
    for color in colors_sequence:
        if drop_separators and color in (SEPARATOR_NAME, UNKNOWN_NAME):
            continue
        steps.append(
            MissionStep(
                index=len(steps) + 1,
                color=color,
                action=action_map.get(color),
            )
        )
    return Mission(steps=steps)
