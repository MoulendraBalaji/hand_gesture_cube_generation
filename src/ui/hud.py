"""On-screen HUD for GestureForge.

Draws the FPS, active gesture + confidence, calibration state, a small FPS
sparkline, and a toggleable gesture cheat-sheet (the §7 table) onto the camera
/ render frame.  Pure OpenCV drawing — no state beyond the provided arguments.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..utils.logger import get_logger

log = get_logger("ui.hud")

# The on-screen cheat sheet mirrors §7 of the spec.
CHEAT_SHEET: tuple[tuple[str, str, str], ...] = (
    ("Gesture", "Action", "Key"),
    ("Pinch (thumb+index)", "Spawn / grab", "Space"),
    ("Fist held", "Lock / grab", "G"),
    ("Open palm", "Release / menu", "Esc"),
    ("Two-hand pinch (dist)", "Scale", "+ / -"),
    ("Two-hand twist (angle)", "Rotate Y", "[ / ]"),
    ("Index only, moving", "Air-draw", "D"),
    ("Swipe left / right", "Undo / redo", "Ctrl+Z / Y"),
    ("Thumbs up 1s", "Save + report", "S"),
    ("OK sign", "Calibrate", "C"),
    ("Peace", "Cycle primitive", "Tab"),
    ("T", "Train gesture", "T"),
    ("R / P", "Record / replay macro", "R / P"),
    ("H", "Toggle this help", "H"),
    ("X", "Clear scene", "X"),
)


class HUD:
    """Draws status panels and the cheat sheet onto a BGR frame."""

    def __init__(
        self,
        title: str = "GestureForge",
        sparkline_size: int = 60,
    ) -> None:
        self.title = title
        self.show_help = False
        self.show_debug = True
        self._fps_history: deque[float] = deque(maxlen=sparkline_size)
        self._last_fps: float = 0.0

    # ------------------------------------------------------------------
    def update_fps(self, fps: float) -> None:
        self._last_fps = fps
        self._fps_history.append(fps)

    @property
    def fps(self) -> float:
        return self._last_fps

    # ------------------------------------------------------------------
    def draw(
        self,
        frame: np.ndarray,
        *,
        active_gesture: str,
        confidence: float,
        calibration_state: str,
        mode: str,
        scene_info: str,
    ) -> np.ndarray:
        """Compose every HUD element onto *frame* and return it."""
        h, w = frame.shape[:2]
        self._draw_top_bar(frame, w)
        self._draw_fps(frame, w, h)
        self._draw_status_frame(
            frame,
            active_gesture=active_gesture,
            confidence=confidence,
            calibration_state=calibration_state,
            mode=mode,
            scene_info=scene_info,
        )
        if self.show_help:
            self._draw_cheat_sheet(frame, w, h)
        return frame

    # ------------------------------------------------------------------
    def _draw_top_bar(self, frame: np.ndarray, w: int) -> None:
        overlay = frame.copy()
        cv_rect(overlay, (0, 0), (w, 36), (30, 30, 40), -1)
        cv_add_text(overlay, f" {self.title}", (8, 24), (255, 255, 255), 0.7, 2)
        alpha = 0.6
        cv_blend(frame, overlay, alpha)

    def _draw_fps(self, frame: np.ndarray, w: int, h: int) -> None:
        hist = list(self._fps_history)
        base_x, base_y = w - 140, 18
        cv_add_text(frame, f"FPS {self._last_fps:5.1f}", (base_x, base_y), (0, 255, 0), 0.6, 2)
        if len(hist) >= 2:
            n = len(hist)
            x0, y0 = w - 120, base_y + 20
            pts = []
            for i, v in enumerate(hist):
                px = x0 + (i / (n - 1)) * 100
                py = y0 - int(min(60, v) / 60 * 30)
                pts.append((int(px), int(py)))
            cv_polyline(frame, pts, (0, 255, 0), False)

    def _draw_status_frame(
        self,
        frame: np.ndarray,
        *,
        active_gesture: str,
        confidence: float,
        calibration_state: str,
        mode: str,
        scene_info: str,
    ) -> None:
        lines = [
            f"Gesture: {active_gesture}",
            f"Confidence: {confidence:.2f}",
            f"Calibration: {calibration_state}",
            f"Mode: {mode}",
            f"Scene: {scene_info}",
        ]
        x, y = 8, 46
        cv_add_text(frame, lines[0], (x, y), (0, 255, 255), 0.6, 2)
        y += 22
        cv_add_text(frame, lines[1], (x, y), (200, 200, 200), 0.6, 1)
        y += 20
        cal_color = (0, 255, 0) if calibration_state.lower().startswith(("ok", "calib", "l")) else (0, 165, 255)
        cv_add_text(frame, lines[2], (x, y), cal_color, 0.6, 1)
        y += 20
        cv_add_text(frame, lines[3], (x, y), (170, 170, 255), 0.6, 1)
        y += 20
        cv_add_text(frame, lines[4], (x, y), (120, 200, 120), 0.6, 1)

    def _draw_cheat_sheet(self, frame: np.ndarray, w: int, h: int) -> None:
        panel_w = 620
        panel_h = len(CHEAT_SHEET) * 20 + 40
        x0, y0 = 20, 60
        overlay = frame.copy()
        cv_rect(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 24), -1)
        cv_add_text(overlay, "  KEYBOARD / GESTURE CHEAT SHEET (press H to hide)", (x0 + 10, y0 + 20), (0, 255, 0), 0.6, 1)
        yy = y0 + 38
        for i, (g, a, k) in enumerate(CHEAT_SHEET):
            if i == 0:
                col = (0, 255, 0)
            else:
                col = (210, 210, 210)
            cv_add_text(overlay, f"  {g:<26} {a:<22} {k}", (x0 + 10, yy), col, 0.5, 1)
            yy += 20
        frame[:] = cv_blend(overlay, frame, 0.92, invert=True)


# ---------------------------------------------------------------------------
# Thin cv2 indirection so HUD can be imported where cv2 is optional.
# ---------------------------------------------------------------------------
_ctx = {"cv": None}


def _cv():
    if _ctx["cv"] is None:
        import cv2

        _ctx["cv"] = cv2
    return _ctx["cv"]


def cv_rect(img, p0, p1, color, thickness):
    return _cv().rectangle(img, p0, p1, color, thickness)


def cv_add_text(img, text, org, color, scale, thick):
    return _cv().putText(img, text, org, _cv().FONT_HERSHEY_SIMPLEX, scale, color, thick)


def cv_add_weighted(a, alpha, b, beta):
    return _cv().addWeighted(a, alpha, b, beta)


def cv_blend(dst, overlay, alpha, invert: bool = False) -> np.ndarray:
    """Blend *overlay* onto *dst* with opacity *alpha* (replacing dst pixels).

    When *invert* is True the roles are swapped so the overlay dominates.
    """
    if invert:
        return _cv().addWeighted(overlay, alpha, dst, 1.0 - alpha, 0)
    return _cv().addWeighted(dst, alpha, overlay, 1.0 - alpha, 0)


def cv_polyline(img, pts, color, closed):
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    return _cv().polylines(img, [arr], closed, color, 1, _cv().LINE_AA)
