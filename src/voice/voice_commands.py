"""Offline voice command listener (optional).

Uses ``vosk`` — a fully local, no-cloud speech recognizer — on a background
thread to turn spoken commands into actions.  Because it must never block the
render loop, decoded commands are pushed onto a thread-safe :class:`queue.Queue`
that the app drains each frame.

This layer is entirely optional: if vosk / the model is missing, the module
instantiates into a "disabled" state and the app simply runs without voice.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from ..utils.logger import get_logger

log = get_logger("voice.commands")

# Supported spoken commands -> canonical action names.
COMMANDS: dict[str, str] = {
    "spawn cube": "spawn_cube",
    "spawn pyramid": "spawn_pyramid",
    "spawn sphere": "spawn_sphere",
    "clear scene": "clear_scene",
    "undo": "undo",
    "save scene": "save_scene",
    "save report": "save_report",
    "rotate": "rotate",
    "calibrate": "calibrate",
}


def _match_command(text: str) -> str | None:
    """Match a normalized transcript against known commands (prefix-tolerant)."""
    lowered = " ".join(text.lower().split())
    best: str | None = None
    best_score = 0.0
    for phrase, action in COMMANDS.items():
        if phrase in lowered:
            score = len(phrase)
            if score > best_score:
                best_score = score
                best = action
    return best


class VoiceListener:
    """Background-thread vosk STT command listener."""

    def __init__(
        self,
        model_dir: str | Path = "assets/voice/vosk-model-small-en-us",
        sample_rate: int = 16000,
        continuous: bool = True,
        language: str = "en-us",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.sample_rate = sample_rate
        self.continuous = continuous
        self.language = language
        self.commands: queue.Queue[str] = queue.Queue()
        self.available = False
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._target_sink = None  # optional audio sink (set by app if desired)

    def attempt_connect(self) -> bool:
        """Try to start the listener; returns False (non-fatal) if unavailable."""
        try:
            import pyaudio  # type: ignore  # noqa: F401
            from vosk import KaldiRecognizer, Model  # type: ignore
        except Exception as exc:
            self.error = f"vosk/pyaudio not installed: {exc}"
            log.warning("Voice disabled: %s", self.error)
            return False

        if not self.model_dir.is_dir():
            self.error = f"Vosk model not found at {self.model_dir}"
            log.warning("Voice disabled: %s", self.error)
            return False

        try:
            self._model = Model(str(self.model_dir))
            self._recognizer = KaldiRecognizer(self._model, self.sample_rate)
            self._recognizer.SetWords(True)
            self.available = True
            self._start_thread()
            return True
        except Exception as exc:
            self.error = str(exc)
            log.warning("Voice disabled: %s", exc)
            return False

    def _start_thread(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="voice-listener", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        import pyaudio

        audio = pyaudio.PyAudio()
        self._audio = audio
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=4000,
            )
        except Exception as exc:
            self.error = str(exc)
            log.warning("Voice mic open failed: %s", exc)
            return
        try:
            while not self._stop.is_set():
                data = stream.read(4000, exception_on_overflow=False)
                if self._recognizer.AcceptWaveform(data):
                    result = json.loads(self._recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        action = _match_command(text)
                        if action:
                            self.commands.put(action)
                            log.info("Voice command: %r -> %s", text, action)
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            audio.terminate()

    def drain(self) -> list[str]:
        """Return all pending voice command actions (non-blocking)."""
        out: list[str] = []
        try:
            while True:
                out.append(self.commands.get_nowait())
        except queue.Empty:
            return out

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
