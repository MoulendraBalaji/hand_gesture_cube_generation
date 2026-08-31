"""Unit tests for the calibrated monocular depth estimator."""

from __future__ import annotations

import pytest

from src.vision.depth_estimator import DepthEstimator


class TestCalibration:
    def test_not_calibrated_by_default(self):
        d = DepthEstimator()
        assert d.is_calibrated is False
        assert d.z_from_span(100.0) is None

    def test_calibrate_sets_focal_and_flag(self):
        d = DepthEstimator(hand_span_m=0.19, reference_distance_m=0.6)
        d.calibrate_span(200.0)
        assert d.is_calibrated is True
        assert d.focal_length_px == pytest.approx(0.6 * 200.0 / 0.19)

    def test_ignores_nonpositive_span(self):
        d = DepthEstimator()
        d.calibrate_span(0)
        assert d.is_calibrated is False


class TestDepth:
    def test_closer_hand_is_nearer_camera(self):
        d = DepthEstimator()
        d.calibrate_span(200.0)
        far = d.z_from_span(100.0)
        near = d.z_from_span(300.0)
        # Larger on-screen span (near hand) => smaller distance.
        assert near < far

    def test_distance_inverse_of_span(self):
        d = DepthEstimator(hand_span_m=0.19, reference_distance_m=0.6)
        d.calibrate_span(200.0)
        z1 = d.z_from_span(200.0)
        z2 = d.z_from_span(100.0)
        assert z1 == pytest.approx(0.6)
        assert z2 == pytest.approx(1.2)

    def test_reference_sanity(self):
        d = DepthEstimator(hand_span_m=0.19, reference_distance_m=0.6)
        d.calibrate_span(200.0)
        # At the calibrated span, distance == reference distance.
        assert d.z_from_span(200.0) == pytest.approx(d.reference_distance_m)
