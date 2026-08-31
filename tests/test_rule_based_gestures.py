"""Unit tests for the rule-based gesture recognizer (full §7 vocabulary)."""

from __future__ import annotations

from hands import build_hand

from src.gestures.rule_based import RuleBasedGestureRecognizer
from src.vision.hand_tracker import Hand


def recognizer(**kw) -> RuleBasedGestureRecognizer:
    return RuleBasedGestureRecognizer(**kw)


def names(events) -> list[str]:
    return [e.name for e in events]


class TestExtension:
    def test_open_palm_all_extended(self):
        h = build_hand([1, 1, 1, 1, 1], open_thumb=True)
        g = RuleBasedGestureRecognizer()._geometry(h)
        assert set(g.extended_fingers) == {0, 1, 2, 3, 4}

    def test_fist_none_extended(self):
        h = build_hand([0, 0, 0, 0, 0])
        g = RuleBasedGestureRecognizer()._geometry(h)
        assert g.extended_fingers == []

    def test_point_only_index(self):
        h = build_hand([0, 1, 0, 0, 0])
        g = RuleBasedGestureRecognizer()._geometry(h)
        assert g.extended_fingers == [1]


class TestDetection:
    def test_open_palm(self):
        r = recognizer()
        assert "open_palm" in names(r.detect([build_hand([1, 1, 1, 1, 1], open_thumb=True)]))

    def test_fist(self):
        r = recognizer()
        assert "fist" in names(r.detect([build_hand([0, 0, 0, 0, 0])]))

    def test_peace(self):
        r = recognizer()
        assert "peace" in names(r.detect([build_hand([0, 1, 1, 0, 0])]))

    def test_point(self):
        r = recognizer()
        assert "point" in names(r.detect([build_hand([0, 1, 0, 0, 0])]))

    def test_thumbs_up(self):
        r = recognizer()
        assert "thumbs_up" in names(r.detect([build_hand([1, 0, 0, 0, 0])]))

    def test_pinch(self):
        r = recognizer()
        assert "pinch" in names(r.detect([build_hand([1, 1, 0, 0, 0], pinch=True)]))

    def test_ok_sign(self):
        r = recognizer()
        assert "ok_sign" in names(r.detect([build_hand([1, 1, 1, 1, 1], pinch=True)]))

    def test_none_detects_nothing(self):
        r = recognizer()
        # An empty hand list yields no events.
        assert r.detect([]) == []


class TestAdaptiveThresholds:
    def test_pinch_detection_is_scale_invariant(self):
        """A pinch at two vastly different hand sizes must still detect."""
        small = build_hand([1, 1, 0, 0, 0], pinch=True, wrist=(0.5, 0.5))
        big = build_hand([1, 1, 0, 0, 0], pinch=True, wrist=(0.5, 0.5))
        # Scale up all of `big` coordinates.
        big.screen = [(x * 5.0, y * 5.0, z) for (x, y, z) in big.screen]
        # Ensure both register a pinch despite totally different pixel extents.
        for hand in (small, big):
            evs = recognizer().detect([hand])
            assert "pinch" in names(evs)

    def test_gesture_does_not_depend_on_absolute_position(self):
        """The same gesture translated across the frame still detects."""
        def translate(hand: Hand, dx: float, dy: float) -> Hand:
            h2 = Hand(label=hand.label)
            h2.screen = [(x + dx, y + dy, z) for (x, y, z) in hand.screen]
            return h2

        base = build_hand([1, 0, 0, 0, 0])
        for dx, dy in [(0, 0), (0.2, 0.1), (-0.3, 0.25)]:
            moved = translate(base, dx, dy)
            assert "thumbs_up" in names(recognizer().detect([moved]))


class TestConfidence:
    def test_junk_hand_returns_none_or_low(self):
        r = recognizer()
        # A hand with degenerate geometry (single landmark) should not raise.
        bogus = Hand(label="Right")
        bogus.screen = [(0.5, 0.5, 0.0)] * 3
        assert r.detect([bogus]) == []
