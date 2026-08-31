"""Shared gesture vocabulary types and data structures.

Defines the canonical gesture names + actions used across the rule-based
detector, the ML classifier, macro recording, and the HUD.  Keeping these in
one place ensures the gesture table shown on screen (§7 of the spec) stays in
sync with what the engine actually detects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical gesture names.  These map 1:1 to rows of the on-screen cheat sheet.
GESTURE_NAMES: tuple[str, ...] = (
    "pinch",
    "fist",
    "open_palm",
    "two_hand_pinch",   # distance change -> scale
    "twist",            # relative angle change -> rotate
    "point",            # index finger only extended -> air-draw
    "swipe_left",
    "swipe_right",
    "thumbs_up",
    "ok_sign",
    "peace",            # two fingers -> cycle primitive
    "none",
)


@dataclass
class GestureEvent:
    """A single detected gesture with its confidence and optional payload."""

    name: str = "none"
    confidence: float = 0.0
    # Payload fields used by specific gestures (e.g. scale delta, angle delta).
    delta: float = 0.0
    hand_index: int = 0
    source: str = "rule"          # "rule" | "ml" | "mixed"
    meta: dict = field(default_factory=dict)
