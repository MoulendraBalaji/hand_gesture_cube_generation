"""Configuration loading and access for GestureForge.

The entire application is parameterised by a YAML configuration file whose
contents are merged over sane in-code defaults.  This keeps every tunable
(thresholds, colours, camera index, bindings, feature toggles) out of the
source tree and in one auditable place.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Defaults mirroring config/default_config.yaml; used as a fallback if no file
# can be found so the app still runs with zero configuration.
_DEFAULTS: dict[str, Any] = {
    "app": {
        "name": "GestureForge",
        "fps_target": 30,
        "mirror": True,
        "window_title": "GestureForge",
        "camera_downscale": 1.0,
    },
    "camera": {"index": 0, "width": 1280, "height": 720},
    "vision": {
        "max_num_hands": 2,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "filter": {"min_cutoff": 1.0, "beta": 0.007, "d_cutoff": 1.0},
    },
    "calibration": {
        "auto_on_first_run": True,
        "reference_distance": 0.6,
        "hand_span_m": 0.19,
        "pinch_ratio": 0.35,
        "fist_ratio": 0.45,
        "samples": 30,
    },
    "gestures": {
        "use_ml": True,
        "ml_fallback_rule": True,
        "ml_min_confidence": 0.75,
        "hud_show_debug": True,
    },
    "physics": {
        "gravity": 9.8,
        "damping": 0.85,
        "restitution": 0.55,
        "air_resistance": 0.995,
        "floor": -3.0,
        "sphere_collision": True,
    },
    "renderer": {
        "default": "opencv",
        "cv": {
            "mesh_color": [200, 110, 40],
            "edge_color": [255, 235, 200],
            "light_dir": [0.4, 0.8, 0.6],
            "perspective_fov_deg": 60,
            "near": 0.1,
            "far": 100.0,
            "camera_z": 6.0,
        },
        "gl": {"clear_color": [0.12, 0.12, 0.14, 1.0]},
    },
    "air_draw": {
        "canvas_color": [80, 200, 255],
        "line_width": 3,
        "auto_save": True,
        "save_dir": "output/drawings",
    },
    "voice": {
        "enabled": False,
        "model_dir": "assets/voice/vosk-model-small-en-us",
        "sample_rate": 16000,
        "continuous": True,
    },
    "analytics": {
        "database": "output/session.db",
        "csv_path": "output/session.csv",
        "report_dir": "output/reports",
        "heatmap_size": 32,
    },
    "export": {"obj_dir": "output/scenes", "default_filename": "scene.obj"},
    "macros": {"save_dir": "output/macros", "default_name": "macro.json"},
    "keys": {
        "spawn": " ",
        "grab": "g",
        "release": "esc",
        "scale_up": "+",
        "scale_down": "-",
        "rotate_y_plus": "]",
        "rotate_y_minus": "[",
        "air_draw_toggle": "d",
        "undo": "ctrl+z",
        "redo": "ctrl+y",
        "save_report": "s",
        "calibrate": "c",
        "cycle_primitive": "tab",
        "record_macro": "r",
        "replay_macro": "p",
        "toggle_help": "h",
        "toggle_debug": "f",
        "clear_scene": "x",
        "quit": "q",
    },
    "actions": {
        "spawn": "spawn",
        "grab": "grab",
        "release": "release",
        "scale": "scale",
        "rotate": "rotate",
        "air_draw": "air_draw",
        "undo": "undo",
        "redo": "redo",
        "save_report": "save_report",
        "calibrate": "calibrate",
        "cycle_primitive": "cycle",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (returns a new dict)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """A thin, attribute-style wrapper over the merged configuration dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data: dict[str, Any] = data

    def section(self, name: str) -> Config:
        """Return a nested section as a new Config view."""
        return Config(self._data.get(name, {}))

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-path accessor, e.g. ``cfg.get("vision.max_num_hands")``."""
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def as_dict(self) -> dict[str, Any]:
        """Return the underlying plain dict (for introspection/serialisation)."""
        return self._data

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __getattr__(self, name: str) -> Any:
        # Allows `cfg.app.name` style access to nested sections/leaf values.
        if name in self._data:
            value = self._data[name]
            return Config(value) if isinstance(value, dict) else value
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config from *path*, merged over in-code defaults.

    Args:
        path: Optional path to a YAML file.  If ``None`` the in-code defaults
            are used directly.  Missing keys in the file are filled in from
            the defaults so a partial config remains valid.

    Returns:
        A fully-merged :class:`Config` instance.
    """
    data: dict[str, Any] = copy.deepcopy(_DEFAULTS)
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        data = _deep_merge(data, loaded)
    return Config(data)
