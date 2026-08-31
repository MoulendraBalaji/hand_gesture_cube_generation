"""MediaPipe hand tracking wrapper.

Wraps ``mediapipe.solutions.hands`` behind a small, test-friendly interface
and adds:
  * support for up to two hands simultaneously,
  * per-hand handedness (Left/Right) with a stable colour,
  * connectivity constants and a named-landmark reference so downstream
    gesture/geometry code never needs to hard-code landmark indices,
  * a run-on-background convenience via a thread-safe queue (see
    :class:`HandTrackerPipeline`).
"""

from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass, field

import numpy as np

try:  # MediaPipe is a hard runtime dependency; import guarded for clarity.
    import mediapipe as mp
except Exception:  # pragma: no cover - exercised only on exotic installs
    mp = None

from ..utils.logger import get_logger

log = get_logger("vision.hand_tracker")


# Landmark index constants (MediaPipe hand topology) — keep as module-level
# names so callers read `lm.INDEX_TIP` instead of the magic number 8.
class LM:
    """Named landmark indices for a single hand."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Canonical connectivity (pairs of landmark indices) for drawing the hand
# skeleton.  Most of this equals MediaPipe's HAND_CONNECTIONS; defined locally
# so the render path does not depend on a MediaPipe import.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# One stable colour (BGR) per distinguishable hand role: right or left.
HAND_COLORS: dict[str, tuple[int, int, int]] = {
    "Right": (0, 200, 255),   # cyan-ish
    "Left": (0, 165, 255),    # orange
    "Unknown": (200, 200, 200),
}


@dataclass
class HandLandmark:
    """A single 2D/3D hand landmark sample."""

    x: float              # normalized [0,1] horizontal (frame-relative)
    y: float              # normalized [0,1] vertical
    z: float              # roughly-normalized depth (MediaPipe's rough Z)

    @property
    def pos(self) -> tuple[float, float, float]:
        """Return the landmark as an ``(x, y, z)`` tuple."""
        return (self.x, self.y, self.z)


@dataclass
class Hand:
    """A full tracked hand with its landmarks and metadata."""

    label: str = "Unknown"          # "Left" | "Right" | "Unknown"
    handedness_score: float = 0.0
    landmarks: list[HandLandmark] = field(default_factory=list)
    # Normalized-screen positions of the 21 landmarks, smoothed by the
    # One-Euro filter (set by the tracker after filtering).
    screen: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def color(self) -> tuple[int, int, int]:
        """BGR colour used to draw this hand."""
        return HAND_COLORS.get(self.label, HAND_COLORS["Unknown"])

    def landmark(self, index: int) -> HandLandmark | None:
        """Return the landmark at *index*, or ``None`` if out of range."""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return ``(x0, y0, x1, y1)`` of the hand's normalized bounding box.

        Uses the smoothed ``screen`` coordinates when available, falling back
        to the raw landmarks.
        """
        if self.screen:
            xs = [p[0] for p in self.screen]
            ys = [p[1] for p in self.screen]
        else:
            xs = [p.x for p in self.landmarks]
            ys = [p.y for p in self.landmarks]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    def bbox_size(self) -> float:
        """Return the diagonal length of the hand bounding box (normalized)."""
        (x0, y0, x1, y1) = self.bounding_box()
        return math.hypot(x1 - x0, y1 - y0)


class HandTracker:
    """High-level MediaPipe Hands wrapper.

    Handles the underlying model lifecycle, per-frame ``process`` calls, and
    converts raw landmark results into :class:`Hand` objects with per-landmark
    One-Euro smoothing applied.  Instantiation is lazy: construction never
    raises if MediaPipe is missing — the first ``update`` call surfaces a
    controlled error instead.
    """

    def __init__(
        self,
        *,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        filter_min_cutoff: float = 1.0,
        filter_beta: float = 0.007,
        filter_d_cutoff: float = 1.0,
    ) -> None:
        self.max_num_hands = max_num_hands
        self._model = _create_hands_model(
            max_num_hands,
            min_detection_confidence,
            min_tracking_confidence,
        )
        from .landmark_filter import LandmarkSmoother  # local import (pure module)

        self._smoothers: dict[int, LandmarkSmoother] = {}
        self._filter_min_cutoff = filter_min_cutoff
        self._filter_beta = filter_beta
        self._filter_d_cutoff = filter_d_cutoff

    # ------------------------------------------------------------------
    def update(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> list[Hand]:
        """Detect and smooth hands in a BGR frame.

        Args:
            frame_bgr: A BGR ``numpy`` frame from the camera.
            timestamp: Optional sample time (seconds) for One-Euro filtering.

        Returns:
            A list of :class:`Hand` objects, one per detected hand.
        """
        if self._model is None:
            raise RuntimeError(
                "MediaPipe is not installed. Run `pip install gestureforge[core]` "
                "or install mediapipe to enable hand tracking."
            )
        rgb = cv2_cvt_bgr2rgb(frame_bgr)
        results = self._model.process(rgb)
        hands: list[Hand] = []
        if results.multi_hand_landmarks is None:
            self._smoothers.clear()
            return hands

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand = self._build_hand(hand_landmarks, results, i, timestamp)
            hands.append(hand)

        # Drop smoothers for hands that vanished this frame.
        active = {id(h) for h in hands}
        self._smoothers = {k: v for k, v in self._smoothers.items() if k in active}
        return hands

    # ------------------------------------------------------------------
    def _build_hand(
        self, hand_landmarks, results, index: int, timestamp: float | None
    ) -> Hand:
        label, score = _label_and_score(results, index)

        # Raw normalized coordinates.
        raw: list[tuple[float, float, float]] = [
            (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
        ]

        # One-Euro smooth the 3D coordinates.
        smoother = self._smoothers.get(index)
        if smoother is None:
            smoother = self._create_smoother()
            self._smoothers[index] = smoother
        smoothed = smoother.smooth(raw, timestamp)

        landmarks = [HandLandmark(*p) for p in smoothed]
        hand = Hand(label=label, handedness_score=float(score), landmarks=landmarks)
        hand.screen = smoothed
        return hand

    def _create_smoother(self):
        from .landmark_filter import LandmarkSmoother

        return LandmarkSmoother(
            21,
            min_cutoff=self._filter_min_cutoff,
            beta=self._filter_beta,
            d_cutoff=self._filter_d_cutoff,
        )

    def close(self) -> None:
        """Release model resources."""
        if self._model is not None:
            try:
                self._model.close()
            except Exception:  # pragma: no cover
                pass
            self._model = None


def _create_hands_model(
    max_num_hands: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
):
    """Construct the MediaPipe Hands model, returning ``None`` if unavailable."""
    if mp is None:
        return None
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def _label_and_score(results, index: int) -> tuple[str, float]:
    """Extract handedness label + confidence from MediaPipe results, defensively."""
    if results.multi_handedness is not None and index < len(results.multi_handedness):
        classification = results.multi_handedness[index]
        if classification and len(classification.classification):
            entry = classification.classification[0]
            return str(entry.label), float(entry.score)
    return "Unknown", 0.0


def cv2_cvt_bgr2rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to RGB (guarded against missing OpenCV)."""
    import cv2

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


class HandTrackerPipeline:
    """Producer-consumer wrapper running MediaPipe off the render thread.

    Inference is pushed onto a bounded FIFO queue consumed by a dedicated
    worker thread, so a slow MediaPipe frame never blocks the render loop.
    The *latest* frame is always read via ``get()``; stale frames are dropped.
    """

    def __init__(self, tracker: HandTracker | None = None, queue_size: int = 2) -> None:
        self._tracker = tracker or HandTracker()
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._result_store: queue.Queue[list[Hand]] = queue.Queue(maxsize=2)
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background inference thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="hand-inference", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame is None:
                break
            try:
                hands = self._tracker.update(frame)
                # Keep only the newest result; drop stale ones.
                try:
                    self._result_store.get_nowait()
                except queue.Empty:
                    pass
                self._result_store.put(hands)
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("Hand inference error: %s", exc)

    def submit(self, frame_bgr: np.ndarray) -> None:
        """Hand a frame to the pipeline (non-blocking; drops if full)."""
        try:
            self._q.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def get(self) -> list[Hand]:
        """Return the latest detected hands (empty list if none yet)."""
        try:
            return self._result_store.get_nowait()
        except queue.Empty:
            return []

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to exit."""
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)
