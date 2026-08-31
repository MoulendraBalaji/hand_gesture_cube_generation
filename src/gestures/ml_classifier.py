"""Trainable gesture classifier + feature extraction.

A collection step records normalized feature vectors (distances and angles
relative to the wrist, all scale-invariant) for a labelled gesture across many
frames.  A small ``scikit-learn`` RandomForest is then trained in a couple of
seconds and persisted to ``assets/gestures/model.pkl``.

At runtime the classifier casts a vote that is merged with the rule-based
detector (rule always remains a safety fallback when no model is trained).
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

from ..vision.hand_tracker import LM, Hand
from . import GestureEvent

try:
    from sklearn.ensemble import RandomForestClassifier

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - sklearn is optional for tests
    _HAS_SKLEARN = False
    RandomForestClassifier = None  # type: ignore[assignment]


# Number of features produced by :func:`extract_features`.
NUM_FEATURES = 26


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a, b, c) -> float:
    """Angle in degrees at vertex b."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def extract_features(hand: Hand) -> list[float]:
    """Compute a normalized feature vector for one hand.

    All features are expressed relative to the wrist so they are translation
    invariant, and normalized by the hand bbox size so they are scale invariant
    (robust to camera distance).

    Args:
        hand: A detected :class:`Hand` (smoothed screen coords expected).

    Returns:
        A length-:data:`NUM_FEATURES` list of floats.
    """
    s = hand.screen
    if not s or len(s) < 21:
        return [0.0] * NUM_FEATURES
    wrist = (s[LM.WRIST][0], s[LM.WRIST][1])
    bbox = hand.bbox_size() or 1.0

    def rel(i: int) -> tuple[float, float]:
        return ((s[i][0] - wrist[0]) / bbox, (s[i][1] - wrist[1]) / bbox)

    # 21 2D landmarks relative to wrist -> 42 dims is too many for a tiny
    # dataset, so we reduce to a compact but discriminative set:
    #   5 tip-to-wrist distances, 5 tip-to-palm-center distances,
    #   5 fingertip spread (tip-to-tip adjacent) distances,
    #   5 knuckle bend angles, 1 pinch (thumb-index) ratio,
    #   and 7 additional relative coordinates. Total 28.
    feats: list[float] = []

    tips = [LM.THUMB_TIP, LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP]

    # 1. Tip-to-wrist distances (5)
    for t in tips:
        feats.append(_dist(rel(t), (0.0, 0.0)))

    # 2. Tip spread (adjacent tip-to-tip) distances (4)
    for i in range(4):
        feats.append(_dist(s[tips[i]], s[tips[i + 1]]) / bbox)

    # 3. Pinch (thumb-index) distance ratio (1)
    feats.append(_dist(s[tips[0]], s[tips[1]]) / bbox)

    # 4. Finger-tip to middle-MCP vectors (5) — captures curl direction
    mid_mcp = rel(LM.MIDDLE_MCP)
    for t in tips:
        feats.append((rel(t)[0] - mid_mcp[0]) + (rel(t)[1] - mid_mcp[1]))

    # 5. Knuckle bend angles at each MCP (4, skip thumb)
    for tip, _pip, mcp in (
        (LM.INDEX_TIP, LM.INDEX_PIP, LM.INDEX_MCP),
        (LM.MIDDLE_TIP, LM.MIDDLE_PIP, LM.MIDDLE_MCP),
        (LM.RING_TIP, LM.RING_PIP, LM.RING_MCP),
        (LM.PINKY_TIP, LM.PINKY_PIP, LM.PINKY_MCP),
    ):
        feats.append(_angle(s[mcp], s[mcp], s[tip]) / 180.0)

    # 6. wrist offsets of index/middle/ring tips (6) as positional context
    for t in tips[:3]:
        rx, ry = rel(t)
        feats.append(rx)
        feats.append(ry)

    # 7. bbox width/height ratio (1)
    (x0, y0, x1, y1) = hand.bounding_box()
    feats.append(((x1 - x0) + 1e-6) / ((y1 - y0) + 1e-6))

    # Normalize to fixed length (defensive).
    if len(feats) < NUM_FEATURES:
        feats += [0.0] * (NUM_FEATURES - len(feats))
    return feats[:NUM_FEATURES]


# ---------------------------------------------------------------------------
# Trainer (capture buffer)
# ---------------------------------------------------------------------------
class GestureCapture:
    """Collects labelled feature samples for training."""

    def __init__(self) -> None:
        self.samples: list[tuple[str, list[float]]] = []

    def add(self, label: str, hand: Hand) -> None:
        """Append one labelled feature vector from a hand."""
        feats = extract_features(hand)
        if any(math.isfinite(f) for f in feats):
            self.samples.append((label, feats))

    def labels(self) -> list[str]:
        return [lbl for lbl, _ in self.samples]

    def clear(self) -> None:
        self.samples.clear()


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class GestureClassifier:
    """RandomForest wrapper that can train, save, load, and predict."""

    def __init__(
        self,
        n_estimators: int = 80,
        seed: int = 4,
        min_confidence: float = 0.55,
    ) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self.min_confidence = min_confidence
        self._model: Any = None
        self._classes: list[str] = []
        self.is_trained = False
        self.training_error: str | None = None

    @property
    def has_sklearn(self) -> bool:
        return _HAS_SKLEARN

    def train(self, samples: list[tuple[str, list[float]]]) -> None:
        """Train a RandomForest on collected labelled samples.

        Args:
            samples: List of ``(label, features)`` tuples.

        Raises:
            RuntimeError: If sklearn is unavailable.
            ValueError: If there are fewer than two distinct classes or no data.
        """
        if not _HAS_SKLEARN or RandomForestClassifier is None:
            self.training_error = "scikit-learn is not installed"
            raise RuntimeError(
                "scikit-learn is required for the ML classifier. "
                "Install with `pip install gestureforge[ml]` or `pip install scikit-learn`."
            )
        if not samples:
            raise ValueError("No training samples collected")
        labels = [lbl for lbl, _ in samples]
        if len(set(labels)) < 2:
            raise ValueError("Need samples from at least two distinct gestures")
        classes = sorted(set(labels))
        if len(classes) < 2:
            raise ValueError("Need at least two classes to train a classifier")

        X = [feats for _, feats in samples]
        y = labels
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            class_weight="balanced",
            n_jobs=1,
        )
        model.fit(X, y)
        self._model = model
        self._classes = classes
        self.is_trained = True
        self.training_error = None

    def predict(self, hand: Hand) -> GestureEvent | None:
        """Predict a gesture class for a hand.

        Args:
            hand: The detected hand.

        Returns:
            A :class:`GestureEvent` if the model is trained and confident
            enough, otherwise ``None`` (caller may fall back to rule-based).
        """
        if not self.is_trained or self._model is None:
            return None
        feats = extract_features(hand)
        probs = self._model.predict_proba([feats])[0]
        idx = int(probs.argmax())
        conf = float(probs[idx])
        label = self._classes[idx]
        if conf < self.min_confidence or label == "none":
            return None
        return GestureEvent(
            name=label, confidence=conf, source="ml", meta={"features": feats}
        )

    def save(self, path: str | Path) -> None:
        """Persist the trained model and its class list to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as handle:
            pickle.dump(
                {
                    "model": self._model,
                    "classes": self._classes,
                    "features": NUM_FEATURES,
                },
                handle,
            )

    def load(self, path: str | Path) -> bool:
        """Load a trained model from disk.

        Returns:
            ``True`` if the model was loaded, else ``False``.
        """
        p = Path(path)
        if not p.is_file():
            return False
        try:
            with p.open("rb") as handle:
                data = pickle.load(handle)
            self._model = data["model"]
            self._classes = list(data["classes"])
            self.is_trained = True
            self.training_error = None
            return True
        except Exception as exc:  # pragma: no cover - corrupted file
            self.training_error = str(exc)
            return False
