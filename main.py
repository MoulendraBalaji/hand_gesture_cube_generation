"""GestureForge CLI entry point.

Runs the touchless 3D interaction studio from a webcam with zero configuration
(a core install + a webcam is enough).  Behaviour is tuned via an optional YAML
config and a handful of argparse flags, several of which overlap with config
keys so the CLI can always win at runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gestureforge",
        description=(
            "GestureForge — real-time, gesture-controlled 3D scene editor "
            "driven entirely from the webcam (no browser)."
        ),
    )
    p.add_argument("--camera", type=int, default=None,
                   help="Webcam index (overrides config).")
    p.add_argument("--renderer", choices=["opencv", "opengl"], default=None,
                   help="Rendering backend (default: opencv; opengl optional).")
    p.add_argument("--voice", action="store_true",
                   help="Enable the optional offline voice command layer.")
    p.add_argument("--resolution", type=str, default=None,
                   help="Camera resolution as WxH, e.g. 1280x720.")
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file (overrides defaults).")
    p.add_argument("--headless-log-only", action="store_true",
                   help="Smoke-test mode for CI: initialize, log, and exit "
                        "without requiring a display/webcam.")
    p.add_argument("--no-ml", action="store_true",
                   help="Disable the trained ML gesture classifier.")
    return p


def _parse_resolution(text: str) -> tuple[int, int] | None:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python main.py``."""
    args = _parser().parse_args(argv)

    config_path = args.config
    if config_path is None:
        here = Path(__file__).resolve().parent
        default_cfg = here / "config" / "default_config.yaml"
        if default_cfg.is_file():
            config_path = str(default_cfg)

    try:
        config = load_config(config_path)
        applied_config = config_path or "built-in defaults"
    except FileNotFoundError as exc:
        print(f"[gestureforge] ERROR: {exc}", file=sys.stderr)
        return 2

    setup_logger(level="INFO")
    from src.utils.logger import get_logger

    log = get_logger("main")
    log.info("Using config: %s", applied_config)

    # CLI flags override config.
    if args.camera is not None:
        config.as_dict()["camera"]["index"] = args.camera
    if args.renderer is not None:
        config.as_dict()["renderer"]["default"] = args.renderer
    if args.voice:
        config.as_dict()["voice"]["enabled"] = True
    if args.no_ml:
        config.as_dict()["gestures"]["use_ml"] = False
    res = _parse_resolution(args.resolution) if args.resolution else None
    if res:
        config.as_dict()["camera"]["width"] = res[0]
        config.as_dict()["camera"]["height"] = res[1]

    from src.app import GestureForgeApp

    app = GestureForgeApp(config, headless_log_only=args.headless_log_only)
    camera_index = config.get("camera.index", 0)

    if args.headless_log_only:
        log.info("Headless smoke test mode")
        app.session.open()
        app.session.log_event("smoke_test", "ok")
        app.session.close()
        log.info("Headless smoke test passed")
        return 0

    try:
        app.run(camera_index, renderer_name=config.get("renderer.default", None))
        return 0
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down.")
        return 130
    except Exception:
        log.exception("Fatal error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
