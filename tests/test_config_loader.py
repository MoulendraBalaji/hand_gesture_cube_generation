"""Unit tests for config loading and merge behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config_loader import load_config


def test_defaults_load_without_file():
    cfg = load_config(None)
    assert cfg.get("app.name") == "GestureForge"
    assert cfg.get("vision.filter.beta") is not None


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_merge_overrides_partial(tmp_path: Path):
    f = tmp_path / "cfg.yaml"
    f.write_text("app:\n  name: Custom\n", encoding="utf-8")
    cfg = load_config(f)
    assert cfg.get("app.name") == "Custom"
    # Non-overridden keys preserved from defaults.
    assert cfg.get("vision.filter.beta") is not None


def test_section_view():
    cfg = load_config(None)
    vis = cfg.section("vision")
    assert vis.get("max_num_hands") == 2


def test_attr_access():
    cfg = load_config(None)
    assert cfg.vision.max_num_hands == 2


def test_get_missing_returns_default():
    cfg = load_config(None)
    assert cfg.get("nope.missing", "fallback") == "fallback"


def test_round_trip_keys_match_config_file():
    """The shipped default_config.yaml must satisfy the loader without error."""
    here = Path(__file__).resolve().parents[1]
    default = here / "config" / "default_config.yaml"
    assert default.is_file()
    cfg = load_config(default)
    assert cfg.get("camera.width") == 1280
