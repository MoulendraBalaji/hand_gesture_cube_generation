"""Unit tests for the ML gesture classifier (feature extraction + training)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hands import build_hand

from src.gestures.ml_classifier import (
    NUM_FEATURES,
    GestureCapture,
    GestureClassifier,
    extract_features,
)


class TestFeatures:
    def test_feature_vector_constant_length(self):
        h = build_hand([1, 1, 1, 1, 1])
        feats = extract_features(h)
        assert len(feats) == NUM_FEATURES

    def test_different_gestures_different_features(self):
        open_f = extract_features(build_hand([1, 1, 1, 1, 1]))
        fist_f = extract_features(build_hand([0, 0, 0, 0, 0]))
        assert open_f != fist_f

    def test_degenerate_hand_zero_features(self):
        from src.vision.hand_tracker import Hand

        h = Hand(label="Right")
        h.screen = []
        feats = extract_features(h)
        assert all(f == 0.0 for f in feats)


@pytest.mark.skipif(
    not GestureClassifier().has_sklearn, reason="scikit-learn not installed"
)
class TestClassifier:
    def test_train_and_predict(self):
        clf = GestureClassifier()
        # Build proper two-class samples.
        samples = []
        for _ in range(8):
            samples.append(("open_palm", extract_features(build_hand([1, 1, 1, 1, 1]))))
            samples.append(("fist", extract_features(build_hand([0, 0, 0, 0, 0]))))
        clf.train(samples)
        assert clf.is_trained
        ev = clf.predict(build_hand([1, 1, 1, 1, 1]))
        assert ev is not None
        assert ev.name in ("open_palm", "fist")

    def test_train_raises_with_single_class(self):
        clf = GestureClassifier()
        samples = [("open_palm", extract_features(build_hand([1, 1, 1, 1, 1]))) for _ in range(5)]
        with pytest.raises(ValueError):
            clf.train(samples)

    def test_save_load_roundtrip(self, tmp_path: Path):
        clf = GestureClassifier()
        samples = []
        for _ in range(6):
            samples.append(("open_palm", extract_features(build_hand([1, 1, 1, 1, 1]))))
            samples.append(("fist", extract_features(build_hand([0, 0, 0, 0, 0]))))
        clf.train(samples)
        path = tmp_path / "model.pkl"
        clf.save(path)
        loaded = GestureClassifier()
        assert loaded.load(path) is True
        assert loaded.is_trained

    def test_load_missing_returns_false(self, tmp_path: Path):
        assert GestureClassifier().load(tmp_path / "missing.pkl") is False

    def test_untrained_predict_none(self):
        assert GestureClassifier().predict(build_hand([1, 1, 1, 1, 1])) is None


class TestCapture:
    def test_capture_adds_labelled_samples(self):
        cap = GestureCapture()
        cap.add("wave", build_hand([1, 1, 1, 1, 1]))
        cap.add("wave", build_hand([1, 1, 1, 1, 1]))
        assert len(cap.samples) == 2
        assert cap.labels() == ["wave", "wave"]

    def test_clear(self):
        cap = GestureCapture()
        cap.add("a", build_hand([1, 1, 1, 1, 1]))
        cap.clear()
        assert cap.samples == []
