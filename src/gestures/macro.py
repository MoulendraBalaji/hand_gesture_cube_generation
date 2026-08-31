"""Gesture macro recording and replay.

A macro captures a time-stamped sequence of action events (e.g. ``spawn``,
``rotate`` with a delta, ``scale`` with a delta) plus the wall-clock delays
between them.  Replaying reproduces the sequence deterministically, which is
genuinely useful for recording consistent portfolio demo GIFs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MACRO_VERSION = 1


class MacroRecorder:
    """Records a sequence of capture frames for later replay."""

    def __init__(self) -> None:
        self.recording = False
        self._events: list[dict[str, Any]] = []
        self._t0: float = 0.0

    def start(self) -> None:
        """Begin a new recording session."""
        self.recording = True
        self._events = []
        self._t0 = time.monotonic()

    def stop(self) -> list[dict[str, Any]]:
        """Stop recording and return the captured events."""
        self.recording = False
        return list(self._events)

    def record(self, action: str, **payload: Any) -> None:
        """Record one action event with its relative timestamp."""
        if not self.recording:
            return
        now = time.monotonic() - self._t0
        self._events.append(
            {"t": round(now, 4), "action": action, **payload}
        )

    def save(self, path: str | Path) -> Path:
        """Persist the captured macro as JSON.

        Args:
            path: Destination file path.

        Returns:
            The path that was written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": MACRO_VERSION,
            "events": self._events,
        }
        with out.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        return out


class MacroPlayer:
    """Replays a macro by dispatching action events to a callback."""

    def __init__(self, dispatch: Callable[[str, dict[str, Any]], None]) -> None:
        self.dispatch = dispatch
        self._events: list[dict[str, Any]] = []
        self._playing = False
        self._idx = 0
        self._t0 = 0.0
        self.loop = False

    def load(self, path: str | Path) -> bool:
        """Load a macro from disk.

        Returns:
            ``True`` if a valid macro was loaded.
        """
        p = Path(path)
        if not p.is_file():
            return False
        try:
            with p.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._events = list(data.get("events", []))
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def start(self) -> None:
        """Begin replaying the loaded macro from scratch."""
        if not self._events:
            return
        self._idx = 0
        self._t0 = time.monotonic()
        self._playing = True

    def stop(self) -> None:
        self._playing = False
        self._idx = 0

    @property
    def playing(self) -> bool:
        return self._playing

    def tick(self) -> bool:
        """Advance the replay by one frame (emit due events).

        Call once per frame while playing.  Returns ``True`` while actively
        playing (``False`` when finished).
        """
        if not self._playing:
            return False
        now = time.monotonic() - self._t0
        while self._idx < len(self._events):
            event = self._events[self._idx]
            if event["t"] <= now:
                payload = {k: v for k, v in event.items() if k != "t" and k != "action"}
                self.dispatch(str(event["action"]), payload)
                self._idx += 1
            else:
                break
        if self._idx >= len(self._events):
            if self.loop:
                self.start()
            else:
                self._playing = False
            return self._playing
        return True
