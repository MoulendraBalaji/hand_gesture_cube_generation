"""Unit tests for the One-Euro landmark filter."""

from __future__ import annotations

import numpy as np

from src.vision.landmark_filter import LandmarkSmoother, LowPassFilter, OneEuroFilter


class TestLowPass:
    def test_first_value_passes_through(self):
        f = LowPassFilter()
        assert f.filter(3.0) == 3.0

    def test_smooths_impulse(self):
        f = LowPassFilter(alpha=0.5)
        f.filter(0.0)
        # alpha 0.5 -> first filtered 0, then 1 -> 0.5, converges.
        f.filter(10.0)
        y = f.filter(10.0)
        assert y < 10.0


class TestOneEuro:
    def test_low_noise_not_degenerate(self):
        f = OneEuroFilter(min_cutoff=2.0, beta=0.0, d_cutoff=1.0)
        vals = [f(x, i * 0.016) for i, x in enumerate([1.0, 1.0, 1.0, 1.0, 1.0])]
        for v in vals:
            assert abs(v - 1.0) < 0.5

    def test_fast_motion_tracked(self):
        f = OneEuroFilter(min_cutoff=4.0, beta=0.0, d_cutoff=2.0)
        # A constant ramp should be tracked with no sign flips.
        out = [f(float(i), i * 0.016) for i in range(10)]
        diffs = [out[i] - out[i - 1] for i in range(1, len(out))]
        assert all(d >= 0 for d in diffs)

    def test_smoother_reduces_noise(self):
        rng = np.random.default_rng(0)
        true = 5.0
        noisy = [true + rng.normal(0, 1) for _ in range(120)]
        sm = LandmarkSmoother(1, min_cutoff=1.0, beta=0.5)
        pts = [[(v, 0.0, 0.0)] for v in noisy]
        out = [sm.smooth(p, i * 0.016)[0][0] for i, p in enumerate(pts)]
        # Smoothed signal should be much less variable than the raw noise.
        assert np.std(out) < np.std(noisy)

    def test_smoother_shape_preserved(self):
        sm = LandmarkSmoother(3)
        inp = [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)]
        out = sm.smooth(inp, 0.0)
        assert len(out) == 3
        assert len(out[0]) == 3

    def test_reset(self):
        sm = LandmarkSmoother(2)
        sm.smooth([(1.0, 1.0, 1.0)] * 2, 0.0)
        sm.reset()
        assert sm.has_been_initialised is False
