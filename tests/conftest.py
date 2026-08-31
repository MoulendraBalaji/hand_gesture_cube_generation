"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys

import pytest

# Ensure this directory is importable so `from hands import build_hand` works
# regardless of how pytest resolves the rootdir / import mode.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hands import build_hand  # noqa: E402

__all__ = ["build_hand"]


@pytest.fixture
def hand_factory():
    """Fixture yielding the synthetic hand builder."""
    return build_hand
