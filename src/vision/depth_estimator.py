"""Monocular depth estimation from a calibrated hand span.

A single webcam cannot triangulate depth the way a stereo rig can, but we can
still get a useful, *calibrated* Z coordinate by leveraging a known physical
constant: the width of the user's palm.  Given the focal length (derived from
the user's measured hand span at a known reference distance) we estimate how
far a hand is from the camera using the standard pinhole relation::

    distance ≈ (known_world_width * focal_length_px) / (observed_width_px)

This turns MediaPipe's unreliable rough-Z into a real metric estimate that is
stable enough to drive 6-DOF manipulation and physics.
"""

from __future__ import annotations

import math

from ..utils.logger import get_logger

log = get_logger("vision.depth_estimator")


class DepthEstimator:
    """Estimates 3D world position from 2D landmarks + a calibrated hand span.

    Calibration measures the hand's on-screen span (in pixels) while the user
    holds their palm at a known reference distance.  From that single sample
    the effective focal length is recovered; subsequent frames convert any
    observed span into a metric Z.  Handedness is not considered — the average
    of the palm span is used.
    """

    def __init__(
        self,
        *,
        hand_span_m: float = 0.19,
        reference_distance_m: float = 0.6,
        focal_length_px: float | None = None,
    ) -> None:
        self.hand_span_m = hand_span_m
        self.reference_distance_m = reference_distance_m
        # Effective focal length in pixels; derived on calibration.
        self.focal_length_px: float | None = focal_length_px
        self.is_calibrated = focal_length_px is not None

    # ------------------------------------------------------------------
    def calibrate_span(self, span_px: float) -> None:
        """Recover the effective focal length from one calibration sample.

        Args:
            span_px: On-screen hand span in pixels at the reference distance.

        Returns:
            None; sets :attr:`focal_length_px` and :attr:`is_calibrated`.
        """
        if span_px <= 0:
            log.warning("Ignoring non-positive calibration span %.3f px", span_px)
            return
        # pinhole: focal / z = span_px / world_span  =>  focal = z * span_px / world
        self.focal_length_px = (
            self.reference_distance_m * span_px / self.hand_span_m
        )
        self.is_calibrated = True
        log.info(
            "Calibrated focal length = %.1f px (span %.1f px @ %.2f m)",
            self.focal_length_px,
            span_px,
            self.reference_distance_m,
        )

    # ------------------------------------------------------------------
    def z_from_span(self, span_px: float) -> float | None:
        """Estimate absolute distance in metres from an observed span in pixels.

        Args:
            span_px: Observed on-screen hand span in pixels.

        Returns:
            Estimated distance in metres, or ``None`` if not yet calibrated.
        """
        if not self.is_calibrated or self.focal_length_px is None or span_px <= 0:
            return None
        return (self.focal_length_px * self.hand_span_m) / span_px

    # ------------------------------------------------------------------
    def pan_span_px(self, hand, frame_width: float) -> float:
        """Compute the palm's on-screen span in pixels for a detected hand.

        The "span" is the pixel distance between the thumb CMC and pinky MCP
        landmarks, which is a reasonable proxy for palm width and is stable
        across finger poses.

        Args:
            hand: A :class:`~gestureforge.src.vision.hand_tracker.Hand`.
            frame_width: Width of the frame in pixels (to denormalize).

        Returns:
            The palm span in pixels (>= 0).
        """
        from .hand_tracker import LM

        a = hand.landmark(LM.THUMB_CMC)
        b = hand.landmark(LM.PINKY_MCP)
        if a is None or b is None:
            return 0.0
        ax, ay = a.x * frame_width, a.y * frame_width
        bx, by = b.x * frame_width, b.y * frame_width
        return math.hypot(bx - ax, by - ay)
