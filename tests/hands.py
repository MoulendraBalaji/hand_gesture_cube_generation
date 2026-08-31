"""Synthetic hand builder used by tests (headless, no MediaPipe needed).

Produces a 21-landmark :class:`Hand` with a specified set of extended fingers,
optional pinch, and optional open-thumb spread.
"""

from __future__ import annotations

import numpy as np

from src.vision.hand_tracker import Hand

FINGERS: dict[int, tuple[int, int, int, int]] = {
    0: (1, 2, 3, 4),
    1: (5, 6, 7, 8),
    2: (9, 10, 11, 12),
    3: (13, 14, 15, 16),
    4: (17, 18, 19, 20),
}
DIR: dict[int, tuple[float, float]] = {
    0: (0.55, 0.20),
    1: (0.35, 0.40),
    2: (0.10, 0.60),
    3: (-0.15, 0.40),
    4: (-0.35, 0.20),
}
LEN: dict[int, float] = {0: 0.45, 1: 0.55, 2: 0.62, 3: 0.58, 4: 0.48}


def build_hand(
    fingers: list[int],
    *,
    pinch: bool = False,
    open_thumb: bool = False,
    wrist: tuple[float, float] = (0.5, 0.6),
) -> Hand:
    """Build a synthetic hand where ``fingers[f]`` is whether finger f extends."""
    w = np.array(wrist, dtype=float)
    lm = np.zeros((21, 3))
    lm[0] = [w[0], w[1], 0.0]
    for f in range(5):
        dx, dy = DIR[f]
        cmc, mcp, pip, tip = FINGERS[f]
        lm[cmc] = [w[0] + dx * 0.08, w[1] + dy * 0.08, 0.0]
        lm[mcp] = [w[0] + dx * 0.18, w[1] + dy * 0.18, 0.0]
    for f in range(5):
        dx, dy = DIR[f]
        cmc, mcp, pip, tip = FINGERS[f]
        ext = bool(fingers[f])
        lm[pip] = [w[0] + dx * 0.30, w[1] + dy * 0.30, 0.0]
        if ext:
            lm[tip] = [w[0] + dx * LEN[f], w[1] + dy * LEN[f], 0.0]
        else:
            lm[tip] = [w[0] + dx * 0.10, w[1] + dy * 0.10, 0.0]
    if open_thumb:
        lm[4] = [w[0] - 0.22, w[1] - 0.10, 0.0]
    if pinch:
        lm[8] = lm[4].copy()
    hand = Hand(label="Right", landmarks=[])
    hand.screen = [tuple(float(v) for v in p) for p in lm]
    return hand
