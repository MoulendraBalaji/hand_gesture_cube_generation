"""Unit tests for the analytics session logger."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analytics.session_logger import SessionLogger


class TestSessionLogger:
    def test_open_creates_schema_and_logs_event(self, tmp_path: Path):
        sl = SessionLogger(db_path=tmp_path / "s.db", csv_path=tmp_path / "s.csv")
        sl.open()
        sl.log_event("spawn", "cube", 1, x=0.5, y=0.4)
        events = sl.retrieve_events()
        assert len(events) == 1
        assert events[0][0] == "spawn"
        assert events[0][1] == "cube"
        sl.close()

    def test_csv_written(self, tmp_path: Path):
        sl = SessionLogger(db_path=tmp_path / "s.db", csv_path=tmp_path / "s.csv")
        sl.open()
        sl.log_event("gesture", "pinch")
        sl.close()
        text = (tmp_path / "s.csv").read_text(encoding="utf-8")
        assert text.startswith("session_id,ts,event_type")
        assert "pinch" in text

    def test_closed_logger_returns_empty(self, tmp_path: Path):
        sl = SessionLogger(db_path=tmp_path / "s.db", csv_path=tmp_path / "s.csv")
        sl.open()
        sl.log_event("a", "b")
        sl.close()
        assert sl.retrieve_events() == []

    def test_scene_snapshot_logs_kinds(self, tmp_path: Path):
        from src.scene.scene_manager import SceneManager

        s = SceneManager()
        s.spawn("cube", np.zeros(3))
        s.spawn("cube", np.zeros(3))
        s.spawn("sphere", np.zeros(3))
        sl = SessionLogger(db_path=tmp_path / "s.db", csv_path=tmp_path / "s.csv")
        sl.open()
        sl.save_scene_snapshot(s.objects)
        events = sl.retrieve_events()
        assert events[-1][0] == "scene_snapshot"
        assert "cube:2" in events[-1][1]
        sl.close()
