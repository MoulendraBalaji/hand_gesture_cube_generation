"""Session analytics logging (SQLite + CSV).

Every meaningful event — a detected gesture, an object spawn/delete/transform,
a spawned object's identity/hand, and the session itself — is appended to a
local SQLite database and mirrored to a CSV so analytics survive even if the
report generator is later extended.  Writes happen on the render thread but are
tiny (single inserts), so they keep the loop smooth.
"""

from __future__ import annotations

import csv
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

log = get_logger("analytics.logger")


class SessionLogger:
    """Logs events + session metadata to SQLite and CSV."""

    def __init__(
        self,
        db_path: str | Path = "output/session.db",
        csv_path: str | Path = "output/session.csv",
    ) -> None:
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.started_at = time.time()
        self._connection: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    def open(self) -> None:
        """Initialise the database schema and CSV header."""
        self._connection = sqlite3.connect(str(self.db_path))
        cur = self._connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started REAL,
                ended REAL,
                event_count INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                ts REAL,
                event_type TEXT,
                detail TEXT,
                hand_index INTEGER,
                x REAL,
                y REAL
            )
            """
        )
        self._connection.commit()
        cur.execute(
            "INSERT OR REPLACE INTO sessions (id, started, event_count) VALUES (?, ?, 0)",
            (self.session_id, self.started_at),
        )
        self._connection.commit()
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        header = ["session_id", "ts", "event_type", "detail", "hand_index", "x", "y"]
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    # ------------------------------------------------------------------
    def log_event(self, event_type: str, detail: str = "", hand_index: int = 0,
                  x: Any = None, y: Any = None) -> None:
        """Log one structured event atomically."""
        ts = time.time() - self.started_at
        if self._connection is not None:
            try:
                self._connection.execute(
                    """
                    INSERT INTO events
                        (session_id, ts, event_type, detail, hand_index, x, y)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (self.session_id, ts, event_type, detail, hand_index, x, y),
                )
                self._connection.execute(
                    "UPDATE sessions SET event_count = event_count + 1"
                    " WHERE id = ?",
                    (self.session_id,),
                )
                self._connection.commit()
            except sqlite3.Error as exc:
                log.warning("Failed to log event: %s", exc)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [self.session_id, round(ts, 4), event_type, detail, hand_index, x, y]
            )

    # ------------------------------------------------------------------
    def retrieve_events(self) -> list[tuple]:
        """Return this session's events (for report generation)."""
        if self._connection is None:
            return []
        cur = self._connection.execute(
            "SELECT event_type, detail, hand_index, x, y FROM events"
            " WHERE session_id = ? ORDER BY ts",
            (self.session_id,),
        )
        return list(cur.fetchall())

    def duration(self) -> float:
        return time.time() - self.started_at

    def close(self) -> None:
        """Finalize the session record and close the DB connection."""
        count = self._event_count()
        if self._connection is not None:
            self._connection.execute(
                "UPDATE sessions SET ended = ? WHERE id = ?",
                (time.time(), self.session_id),
            )
            self._connection.commit()
            self._connection.close()
            self._connection = None
        log.info("Session %s logged (%d events)", self.session_id, count)

    def _event_count(self) -> int:
        if self._connection is None:
            return 0
        try:
            cur = self._connection.execute(
                "SELECT event_count FROM sessions WHERE id = ?", (self.session_id,)
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0

    def save_scene_snapshot(self, objects) -> None:
        """Record the scene composition (count per kind) for the report."""
        from collections import Counter

        kinds = Counter(o.kind for o in objects)
        detail = ";".join(f"{k}:{v}" for k, v in kinds.items())
        self.log_event("scene_snapshot", detail)
