"""Structured logging for GestureForge.

Provides a console handler plus a rotating file handler so the app emits
structured ``logging`` records (never bare ``print`` statements) and keeps a
bounded on-disk log for post-hoc debugging of a session.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


def setup_logger(
    name: str = "gestureforge",
    *,
    console: bool = True,
    file: bool = True,
    log_dir: str | Path = "output/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure the root GestureForge logger (idempotent).

    Args:
        name: Logger name to obtain.
        console: Whether to attach a console ``StreamHandler``.
        file: Whether to attach a rotating file handler.
        log_dir: Directory for the rotating log files.
        level: Base logging level.

    Returns:
        The configured logger instance.
    """
    global _configured
    root = logging.getLogger(name)

    if not (console or file):
        root.addHandler(logging.NullHandler())
        root.setLevel(logging.DEBUG)
        return root

    if not _configured:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root.addHandler(console_handler)
        if file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_path / "gestureforge.log",
                maxBytes=2 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        root.setLevel(level)
        _configured = True

    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of the ``gestureforge`` namespace."""
    return logging.getLogger(f"gestureforge.{name}" if name else "gestureforge")
