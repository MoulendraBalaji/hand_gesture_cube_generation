"""Unit tests for the end-of-session PNG/PDF report generator.

Report generation is headless (matplotlib with the Agg backend), so it runs in
CI without a display.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analytics.report_generator import _build_heatmap, generate_report


def sample_events():
    return [
        ("spawn", "cube", 0, 0.5, 0.5),
        ("spawn", "cube", 0, 0.2, 0.3),
        ("gesture", "pinch", 0, 0.6, 0.4),
        ("delete", "grabbed", 1, 0.9, 0.1),
    ]


class TestReport:
    def test_generates_png(self, tmp_path: Path):
        out = generate_report(sample_events(), duration=120.0, report_dir=tmp_path)
        png = Path(out["png"])
        assert png.exists() and png.stat().st_size > 0

    def test_returns_something(self, tmp_path: Path):
        out = generate_report(sample_events(), duration=60.0, report_dir=tmp_path)
        assert "png" in out
        assert len(out) >= 1

    def test_empty_session_no_crash(self, tmp_path: Path):
        out = generate_report([], duration=5.0, report_dir=tmp_path)
        assert Path(out["png"]).exists()

    def test_heatmap_accumulates_events(self):
        events = [("gesture", "pinch", 0, 0.5, 0.5) for _ in range(4)]
        grid = _build_heatmap(events, 32)
        assert grid.max() == pytest.approx(1.0)
        assert grid.sum() > 0
