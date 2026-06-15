"""French text-to-speech with graceful backend fallback.

Speaking is best-effort: the pipeline must keep running even with no audio stack.
Backends are tried in this order (first available wins unless one is forced):

* ``spd-say``   — speech-dispatcher CLI (present on most desktop Linux);
* ``espeak-ng`` / ``espeak`` — direct synthesiser CLI;
* ``pyttsx3``   — offline Python engine (optional dependency);
* ``gtts``      — Google online TTS, played via ``ffplay``/``aplay`` (optional);
* ``print``     — no audio, just echoes the phrase (always available).

Every backend speaks *synchronously* (``say`` blocks until the phrase finishes)
so consecutive mission steps do not overlap or clip each other.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

# Backends that do real audio, in preference order. ``print`` is appended last as
# the universal fallback and is intentionally not in this list.
_AUDIO_BACKEND_ORDER = ["spd-say", "espeak-ng", "espeak", "pyttsx3", "gtts"]


def _has_cli(name: str) -> bool:
    return shutil.which(name) is not None


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def available_backends() -> List[str]:
    """Names of TTS backends usable on this machine (most preferred first)."""
    found: List[str] = []
    for backend in _AUDIO_BACKEND_ORDER:
        if backend in ("spd-say", "espeak-ng", "espeak"):
            if _has_cli(backend):
                found.append(backend)
        elif backend == "pyttsx3":
            if _has_module("pyttsx3"):
                found.append(backend)
        elif backend == "gtts":
            # gTTS needs both the library and something to play the MP3.
            if _has_module("gtts") and (_has_cli("ffplay") or _has_cli("mpg123")):
                found.append(backend)
    found.append("print")
    return found


class FrenchTTS:
    """Speak French phrases through the first available (or forced) backend."""

    def __init__(
        self,
        backend: Optional[str] = None,
        lang: str = "fr",
        rate: Optional[int] = None,
        enabled: bool = True,
    ):
        self.lang = lang
        self.rate = rate
        self.enabled = enabled
        options = available_backends()
        if backend and backend != "auto":
            if backend not in options and backend != "print":
                raise ValueError(
                    f"TTS backend {backend!r} is not available. "
                    f"Choose from: {', '.join(options)}"
                )
            self.backend = backend
        else:
            self.backend = options[0]
        self._engine = None  # lazily built pyttsx3 engine

    def describe(self) -> str:
        state = self.backend if self.enabled else f"{self.backend} (muted)"
        return f"French TTS via {state}"

    def say(self, text: str) -> None:
        """Speak one phrase, blocking until it finishes. Never raises on failure."""
        text = (text or "").strip()
        if not text:
            return
        if not self.enabled:
            return
        try:
            self._dispatch(text)
        except Exception as exc:  # pragma: no cover - audio stacks vary by host
            print(f"[tts:{self.backend} failed: {exc}] {text}", file=sys.stderr)

    def say_many(self, texts: Iterable[str]) -> None:
        for text in texts:
            self.say(text)

    # -- backend implementations ------------------------------------------------

    def _dispatch(self, text: str) -> None:
        if self.backend == "spd-say":
            self._say_spd(text)
        elif self.backend in ("espeak-ng", "espeak"):
            self._say_espeak(text)
        elif self.backend == "pyttsx3":
            self._say_pyttsx3(text)
        elif self.backend == "gtts":
            self._say_gtts(text)
        else:
            self._say_print(text)

    def _say_spd(self, text: str) -> None:
        # -w waits for the phrase to finish; -l sets the language.
        cmd = ["spd-say", "-w", "-l", self.lang]
        if self.rate is not None:
            # spd-say rate is -100..100; clamp a words-per-minute-ish value down.
            cmd += ["-r", str(max(-100, min(100, self.rate)))]
        cmd.append(text)
        subprocess.run(cmd, check=True)

    def _say_espeak(self, text: str) -> None:
        cmd = [self.backend, "-v", self.lang]
        if self.rate is not None:
            cmd += ["-s", str(self.rate)]
        cmd.append(text)
        subprocess.run(cmd, check=True)

    def _say_pyttsx3(self, text: str) -> None:
        import pyttsx3

        if self._engine is None:
            self._engine = pyttsx3.init()
            for voice in self._engine.getProperty("voices"):
                if self.lang in (getattr(voice, "languages", []) or []) or \
                        self.lang in voice.id.lower() or "french" in voice.name.lower():
                    self._engine.setProperty("voice", voice.id)
                    break
            if self.rate is not None:
                self._engine.setProperty("rate", self.rate)
        self._engine.say(text)
        self._engine.runAndWait()

    def _say_gtts(self, text: str) -> None:
        from gtts import gTTS

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            path = Path(handle.name)
        try:
            gTTS(text=text, lang=self.lang).save(str(path))
            if _has_cli("ffplay"):
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
                    check=True,
                )
            else:
                subprocess.run(["mpg123", "-q", str(path)], check=True)
        finally:
            path.unlink(missing_ok=True)

    def _say_print(self, text: str) -> None:
        print(f"\N{SPEAKER WITH THREE SOUND WAVES}  [fr] {text}")
