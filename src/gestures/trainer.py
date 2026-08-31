"""Interactive gesture capture + training workflow.

Provides the ready-to-wire components for the ``T`` training mode: a capture
phase where the user repeats a labelled gesture N times while the system logs
feature vectors, followed by an optional train-and-save step.  The actual frame
loop lives in the app; this module supplies the state machine helpers and the
``GestureTrainer`` façade that keeps capture state, training, and persistence
in one place.
"""

from __future__ import annotations

from pathlib import Path

from ..utils.logger import get_logger
from .ml_classifier import GestureCapture, GestureClassifier

log = get_logger("gestures.trainer")


class GestureTrainer:
    """Coordinate capture -> train -> save for custom gestures.

    The app drives the capture by calling :meth:`capture_frame` for every frame
    while the user performs the gesture.  Once the required number of samples
    per label is met the app may train and persist.
    """

    def __init__(
        self,
        classifier: GestureClassifier | None = None,
        samples_per_label: int = 40,
        model_path: str | Path = "assets/gestures/model.pkl",
    ) -> None:
        self.classifier = classifier or GestureClassifier()
        self.capture = GestureCapture()
        self.samples_per_label = samples_per_label
        self.model_path = Path(model_path)
        self.current_label: str | None = None
        self.recording = False

    # ------------------------------------------------------------------
    def start_recording(self, label: str) -> None:
        """Begin collecting samples under a label."""
        self.current_label = label
        self.recording = True

    def stop_recording(self) -> None:
        """Stop the current capture session."""
        self.recording = False
        self.current_label = None

    def capture_frame(self, hand) -> bool:
        """Record one sample from a hand if currently recording.

        Args:
            hand: A detected :class:`Hand`.

        Returns:
            ``True`` if a sample was appended.
        """
        if not self.recording or hand is None or self.current_label is None:
            return False
        self.capture.add(self.current_label, hand)
        return True

    def count_for(self, label: str) -> int:
        """Number of samples already collected for *label*."""
        return sum(1 for lbl, _ in self.capture.samples if lbl == label)

    def label_complete(self, label: str) -> bool:
        """Whether *label* has enough samples to consider its capture done."""
        return self.count_for(label) >= self.samples_per_label

    # ------------------------------------------------------------------
    def train(self) -> str | None:
        """Train and save the classifier.

        Returns:
            A human-readable status/error string for the HUD, or ``None`` on
            success.
        """
        try:
            self.classifier.train(self.capture.samples)
            self.classifier.save(self.model_path)
            log.info("Trained and saved classifier to %s", self.model_path)
            return None
        except (ValueError, RuntimeError) as exc:
            log.warning("Training failed: %s", exc)
            return str(exc)

    def load_if_available(self) -> bool:
        """Attempt to load a previously trained model.

        Returns:
            ``True`` if a model was loaded.
        """
        ok = self.classifier.load(self.model_path)
        if ok:
            log.info("Loaded classifier from %s", self.model_path)
        return ok

    def clear_capture(self) -> None:
        """Drop all collected samples and reset recording state."""
        self.capture.clear()
        self.recording = False
        self.current_label = None
