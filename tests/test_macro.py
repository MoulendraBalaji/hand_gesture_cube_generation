"""Unit tests for gesture macro recording and replay."""

from __future__ import annotations

from pathlib import Path

from src.gestures.macro import MacroPlayer, MacroRecorder


class TestRecorder:
    def test_records_events_with_timestamps(self):
        r = MacroRecorder()
        r.start()
        r.record("spawn", kind="cube")
        assert len(r._events) == 1
        assert r._events[0]["action"] == "spawn"
        assert r._events[0]["kind"] == "cube"
        assert r._events[0]["t"] >= 0
        r.stop()

    def test_does_not_record_when_stopped(self):
        r = MacroRecorder()
        r.record("spawn")  # not recording
        assert r._events == []

    def test_save_writes_json(self, tmp_path: Path):
        r = MacroRecorder()
        r.start()
        r.record("rotate", deg=15.0)
        r.stop()
        path = r.save(tmp_path / "m.json")
        assert path.is_file()
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert len(data["events"]) == 1


class TestPlayer:
    def _write_macro(self, tmp_path: Path) -> Path:
        r = MacroRecorder()
        r.start()
        r.record("spawn", kind="cube")
        r.record("rotate", deg=15.0)
        r.stop()
        return r.save(tmp_path / "m.json")

    def test_load_and_play_dispatches(self, tmp_path: Path):
        path = self._write_macro(tmp_path)
        dispatched = []
        p = MacroPlayer(dispatch=lambda a, payload: dispatched.append((a, payload)))
        assert p.load(path)
        p.start()
        # Fake the real-time clock so events dispatch immediately.
        p._t0 -= 100.0  # rewind reference time so all are due
        p.tick()
        assert len(dispatched) == 2
        assert dispatched[0][0] == "spawn"
        assert dispatched[0][1]["kind"] == "cube"

    def test_load_missing_returns_false(self, tmp_path: Path):
        p = MacroPlayer(dispatch=lambda a, p: None)
        assert p.load(tmp_path / "nope.json") is False

    def test_finishes_when_all_events_emitted(self, tmp_path: Path):
        path = self._write_macro(tmp_path)
        p = MacroPlayer(dispatch=lambda a, p: None)
        p.load(path)
        p.start()
        p._t0 -= 100.0
        p.tick()
        assert not p.playing
