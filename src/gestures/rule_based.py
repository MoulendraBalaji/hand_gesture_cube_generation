"""Geometric / angle-based gesture detection.

A fast, deterministic, camera-agnostic recognizer covering the full GestureForge
vocabulary (§7 of the spec).  All thresholds are *scale-invariant*: distances
are expressed as ratios of the current hand bounding-box size, so detection is
identical whether the user is near the camera or far from it.  It never
requires a trained model and thus always provides a reliable fallback when the
ML classifier has no data yet.

This module contains no camera / OpenCV I/O so it is fully unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..vision.hand_tracker import LM, Hand
from . import GestureEvent

# ---------------------------------------------------------------------------
# Small pure-math helpers reused by the recognizer.
# ---------------------------------------------------------------------------


def distance(a, b) -> float:
    """Euclidean 2D distance between two landmarks/points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_at(a, b, c) -> float:
    """Angle (degrees) at vertex *b* formed by points *a*, *b*, *c*."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


@dataclass
class HandGeometry:
    """Precomputed, normalized geometry for one hand used by all detectors."""

    hand: Hand
    bbox: float                      # diagonal of hand bounding box (normalized)
    palm: tuple[float, float] | None
    palm_center: tuple[float, float] | None
    fingertip: tuple[float, float] | None          # index tip
    thumb_tip: tuple[float, float] | None
    pinch: float | None                            # thumb-index distance ratio
    extended_fingers: list[int]                       # indices of extended tips
    angles: dict[int, float]                          # knuckle angles per finger
    # Index positions for helpers below.
    index_tip: tuple[float, float] | None


class RuleBasedGestureRecognizer:
    """Detects the full gesture vocabulary from a list of :class:`Hand` objects.

    The recognizer is *stateless* between hand objects except for a lightweight
    swipe detector that needs the previous frame's wrist positions.  It returns
    a list of :class:`GestureEvent` (one per hand, plus combined two-hand
    gestures).
    """

    # Number of recent x/y samples kept for velocity-based gestures (swipe).
    _VELOCITY_WINDOW = 5

    def __init__(self, pinch_ratio: float = 0.35, fist_ratio: float = 0.45) -> None:
        self.pinch_ratio = pinch_ratio
        self.fist_ratio = fist_ratio
        self._prev_wrists: dict[int, list[tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    def detect(self, hands: list[Hand]) -> list[GestureEvent]:
        """Detect gestures from the current frame's hands.

        Args:
            hands: The list of detected :class:`Hand` objects.

        Returns:
            A list of :class:`GestureEvent` (may be empty).
        """
        events: list[GestureEvent] = []
        geoms = [self._geometry(h) for h in hands]

        # Per-hand gestures.
        for idx, geom in enumerate(geoms):
            ev = self._detect_single(geom, idx)
            if ev is not None and ev.name != "none":
                events.append(ev)

        # Combined two-hand gestures (scale via pinch spread, rotate via twist).
        if len(geoms) >= 2:
            events.extend(self._detect_two_hand(geoms))

        # Record wrist history for swipe detection (after detection uses it).
        self._update_wrists(geoms)
        return events

    # ------------------------------------------------------------------
    def _geometry(self, hand: Hand) -> HandGeometry:
        """Extract normalized geometry from a hand using its *smoothed* screen coords."""
        s = hand.screen
        if not s or len(s) < 21:
            return HandGeometry(hand, 1.0, None, None, None, None, None, [], {}, None)

        bbox = hand.bbox_size()
        bbox = bbox if bbox > 1e-6 else 1.0

        def at(i: int) -> tuple[float, float] | None:
            if i < len(s):
                return (s[i][0], s[i][1])
            return None

        palm = at(LM.WRIST)
        palm_center = _mean([at(LM.WRIST), at(LM.INDEX_MCP), at(LM.PINKY_MCP)])
        index_tip = at(LM.INDEX_TIP)
        thumb_tip = at(LM.THUMB_TIP)

        pinch = None
        if index_tip and thumb_tip:
            pinch = distance(index_tip, thumb_tip) / bbox

        # Which fingers are extended?  A finger is extended when its tip is
        # farther from the wrist than its own PIP joint — i.e. the finger is
        # actually straight rather than curled into the palm.  Because both
        # distances originate at the wrist and are compared as a ratio, the
        # test is scale-invariant (robust to camera distance) and orientation-
        # invariant (works whether the hand is pointing up, down, or sideways).
        # This is the classic robust MediaPipe finger-extension heuristic.
        extended = []
        angles: dict[int, float] = {}
        palm_ref = palm if palm is not None else (0.0, 0.0)
        for tip, pip, mcp, finger in (
            (LM.THUMB_TIP, LM.THUMB_IP, LM.THUMB_MCP, 0),
            (LM.INDEX_TIP, LM.INDEX_PIP, LM.INDEX_MCP, 1),
            (LM.MIDDLE_TIP, LM.MIDDLE_PIP, LM.MIDDLE_MCP, 2),
            (LM.RING_TIP, LM.RING_PIP, LM.RING_MCP, 3),
            (LM.PINKY_TIP, LM.PINKY_PIP, LM.PINKY_MCP, 4),
        ):
            t, p, m = at(tip), at(pip), at(mcp)
            if t and p and m:
                d_tip_w = distance(t, palm_ref)
                d_pip_w = distance(p, palm_ref)
                # A straight finger keeps its tip clearly beyond its PIP joint.
                if d_tip_w > d_pip_w * 1.05:
                    extended.append(finger)
                angles[finger] = angle_at(p, m, t)

        return HandGeometry(
            hand=hand,
            bbox=bbox,
            palm=palm,
            palm_center=palm_center,
            fingertip=index_tip,
            thumb_tip=thumb_tip,
            pinch=pinch,
            extended_fingers=extended,
            angles=angles,
            index_tip=index_tip,
        )

    # ------------------------------------------------------------------
    def _detect_single(self, g: HandGeometry, idx: int) -> GestureEvent | None:
        """Detect a single-hand gesture from its geometry."""
        if g.pinch is None or g.palm is None:
            return None

        extended = set(g.extended_fingers)
        pinch = g.pinch

        # --- OK sign: thumb+index tips touching (pinch) with other fingers up.
        if (
            pinch < self.pinch_ratio
            and 0 in extended
            and 1 in extended
            and {2, 3, 4} <= extended
        ):
            return GestureEvent(
                name="ok_sign",
                confidence=0.8,
                hand_index=idx,
            )

        # --- Thumbs up: only thumb extended.
        if extended == {0}:
            return GestureEvent(name="thumbs_up", confidence=0.9, hand_index=idx)

        # --- Peace: index + middle extended, others folded.
        if extended == {1, 2}:
            return GestureEvent(name="peace", confidence=0.9, hand_index=idx)

        # --- Point: only index extended (air-draw mode).
        if extended == {1}:
            return GestureEvent(name="point", confidence=0.85, hand_index=idx)

        # --- Fist: no fingers extended.
        if not extended:
            if self._is_fist(g):
                return GestureEvent(name="fist", confidence=0.9, hand_index=idx)

        # --- Open palm: all five extended.
        if extended == {0, 1, 2, 3, 4}:
            return GestureEvent(name="open_palm", confidence=0.9, hand_index=idx)

        # --- Pinch: thumb+index near (regardless of other fingers).
        if 0 in extended and 1 in extended and pinch < self.pinch_ratio:
            return GestureEvent(name="pinch", confidence=0.95, hand_index=idx)

        # --- Swipe (velocity based), evaluated independently.
        swipe = self._detect_swipe(g, idx)
        if swipe is not None:
            return swipe

        return None

    def _is_fist(self, g: HandGeometry) -> bool:
        """Heuristic: fist when the four fingertips cluster near the palm."""
        s = g.hand.screen
        if not s or g.palm_center is None:
            return False
        tips = [LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP]
        spread = 0.0
        count = 0
        for t in tips:
            if t < len(s):
                spread += distance((s[t][0], s[t][1]), g.palm_center)
                count += 1
        if count == 0:
            return False
        return (spread / count) / g.bbox < self.fist_ratio

    # ------------------------------------------------------------------
    def _detect_swipe(self, g: HandGeometry, idx: int) -> GestureEvent | None:
        """Detect fast lateral wrist motion -> swipe left/right."""
        if g.palm is None:
            return None
        hist = self._prev_wrists.get(idx, [])
        if len(hist) < self._VELOCITY_WINDOW - 1:
            return None
        first = hist[0]
        vx = g.palm[0] - first[0]          # normalized delta over window
        # Only a mostly-lateral fast motion counts as a swipe.
        if abs(vx) < 0.10:
            return None
        if abs(vx) < abs(g.palm[1] - first[1]) * 2.5:
            return None  # vertical movement dominates -> not a swipe
        return GestureEvent(
            name="swipe_left" if vx < 0 else "swipe_right",
            confidence=min(0.95, abs(vx) * 6.0),
            hand_index=idx,
        )

    def _update_wrists(self, geoms: list[HandGeometry]) -> None:
        for idx, g in enumerate(geoms):
            if g.palm is None:
                continue
            self._prev_wrists.setdefault(idx, []).append(g.palm)
            if len(self._prev_wrists[idx]) > self._VELOCITY_WINDOW:
                self._prev_wrists[idx].pop(0)

    # ------------------------------------------------------------------
    def _detect_two_hand(self, geoms: list[HandGeometry]) -> list[GestureEvent]:
        """Detect scale (two-hand pinch spread) and rotate (two-hand twist).

        These single-shot events are emitted as stream deltas that the rest of
        the app can consume continuously; the ``meta`` dict carries the signed
        delta for scaling/rotation.
        """
        events: list[GestureEvent] = []
        g0, g1 = geoms[0], geoms[1]

        # Pinch-spread scaling: both hands pinching, distance between the two
        # pinch points changes.
        if (
            g0.pinch is not None
            and g1.pinch is not None
            and g0.pinch < self.pinch_ratio * 1.5
            and g1.pinch < self.pinch_ratio * 1.5
        ):
            events.append(
                GestureEvent(name="two_hand_pinch", confidence=0.7, delta=0.0)
            )
        # Store raw spread for the caller to compute deltas.
        p0 = g0.palm_center or (0, 0)
        p1 = g1.palm_center or (0, 0)
        spread = distance(p0, p1)
        # Twist: difference in the vertical angles of the two palms.
        a0 = _palm_angle(g0)
        a1 = _palm_angle(g1)
        events.append(
            GestureEvent(
                name="twist",
                confidence=0.1,
                delta=math.degrees(a1 - a0),
                meta={"spread": spread, "wrist0": p0, "wrist1": p1},
            )
        )
        return events


def _mean(points) -> tuple[float, float] | None:
    pts = [p for p in points if p is not None]
    if not pts:
        return None
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def _palm_angle(g: HandGeometry) -> float:
    """Vertical angle of the hand's wrist->middle-mcp vector (radians)."""
    s = g.hand.screen
    palm = g.palm
    if palm is None or not s or LM.MIDDLE_MCP >= len(s):
        return 0.0
    m = (s[LM.MIDDLE_MCP][0], s[LM.MIDDLE_MCP][1])
    return math.atan2(m[1] - palm[1], m[0] - palm[0])
