"""One-Euro filter for per-landmark smoothing.

MediaPipe hand landmarks are noisy at the sub-pixel level; naive per-frame
consumption produces visibly jittery cursors and garbage gesture angles.
The *One-Euro filter* (Casiez et al., 2012) is the standard production
technique for exactly this problem: a low-pass filter with an adaptive
smoothing factor that drops as cursor speed rises, giving a good trade-off
between lag and noise across a wide range of motion speeds.

This module is deliberately pure math (no OpenCV / MediaPipe imports) so it
can be unit-tested headlessly and reused anywhere landmarks are consumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


@dataclass
class LowPassFilter:
    """A single-variable exponential low-pass filter."""

    alpha: float = 0.5
    y: float = 0.0
    has_been_set: bool = False

    def filter(self, value: float, alpha: float | None = None) -> float:
        """Feed one sample and return the filtered value.

        Args:
            value: Raw input sample.
            alpha: Smoothing factor (0..1).  When omitted the value supplied
                at construction (or the last assigned value) is used.

        Returns:
            Filtered output.
        """
        if alpha is not None:
            self.alpha = alpha
        if not self.has_been_set:
            self.has_been_set = True
            self.y = value
        else:
            self.y = self.alpha * value + (1.0 - self.alpha) * self.y
        return self.y


@dataclass
class OneEuroFilter:
    """Adaptive, velocity-aware low-pass filter for a single signal."""

    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0

    x_filter: LowPassFilter = field(default_factory=LowPassFilter)
    dx_filter: LowPassFilter = field(default_factory=LowPassFilter)
    last_time: float | None = None

    # ------------------------------------------------------------------
    def _filtering_alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau)

    def __call__(self, x: float, timestamp: float | None = None) -> float:
        """Filter a single incoming sample.

        Args:
            x: The raw value to smooth.
            timestamp: Optional time (seconds) of the sample; used to compute
                an accurate derivative.  When ``None`` a fixed alpha is used.

        Returns:
            The smoothed value.
        """
        if self.last_time is None or timestamp is None:
            # No temporal reference available: fall back to a fixed alpha
            # derived from min_cutoff.
            alpha = self._filtering_alpha(self.min_cutoff)
            self.x_filter.filter(x, alpha)
            self.last_time = timestamp
            return self.x_filter.y

        dt = timestamp - self.last_time
        self.last_time = timestamp
        if dt <= 0:
            return self.x_filter.y

        # Estimate derivative of the raw signal and smooth it.
        dx = (x - self.x_filter.y) / dt if self.x_filter.has_been_set else 0.0
        edx = self.dx_filter.filter(dx, self._filtering_alpha(self.d_cutoff))

        # Adaptive cutoff: increase with speed to reduce lag on fast motion.
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._filtering_alpha(cutoff)
        return self.x_filter.filter(x, alpha)


class LandmarkSmoother:
    """Applies one independent One-Euro filter per (x, y, z) axis per landmark.

    ``MediaPipe`` produces 21 landmarks per hand; a full hand therefore needs
    21 * 3 = 63 independent filters.  This class owns that matrix so callers
    can smooth a whole hand in one call.
    """

    def __init__(
        self,
        num_landmarks: int = 21,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._x = [
            [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(num_landmarks)]
            for _ in range(3)
        ]
        self._num_landmarks = num_landmarks
        self.has_been_initialised = False

    def reset(self) -> None:
        """Clear all internal filter state (e.g. when a hand disappears)."""
        self._x = [
            [OneEuroFilter() for _ in range(self._num_landmarks)] for _ in range(3)
        ]
        self.has_been_initialised = False

    def smooth(
        self, landmarks: list[tuple[float, float, float]], timestamp: float | None = None
    ) -> list[tuple[float, float, float]]:
        """Smooth a sequence of (x, y, z) landmarks.

        Args:
            landmarks: A list of ``(x, y, z)`` tuples, one per landmark.
            timestamp: Optional sample time in seconds.

        Returns:
            A new list of smoothed ``(x, y, z)`` tuples, same length.
        """
        out: list[tuple[float, float, float]] = []
        for i, point in enumerate(landmarks):
            x = self._x[0][i](point[0], timestamp)
            y = self._x[1][i](point[1], timestamp)
            z = self._x[2][i](point[2], timestamp)
            out.append((x, y, z))
        return out
