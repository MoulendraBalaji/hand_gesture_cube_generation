"""Abstract renderer interface.

Both the default OpenCV wireframe renderer and the optional OpenGL backend
implement this protocol so the app can swap backends via ``--renderer`` with no
changes to the scene/drawing logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from ..scene.primitives import Mesh
from ..scene.scene_manager import SceneObject


class BaseRenderer(ABC):
    """Interface every renderer must satisfy.

    The renderer is responsible for turning the current scene into something
    drawable.  The OpenCV renderer draws directly into a BGR frame; the OpenGL
    renderer maintains its own window.  Both expose :meth:`project` so the app
    can convert world coordinates to screen coordinates for gesture logic.
    """

    @abstractmethod
    def begin_frame(self, frame: np.ndarray) -> None:
        """Start rendering a new frame (clear any per-frame state)."""

    @abstractmethod
    def draw_mesh(self, mesh: Mesh, color: tuple[int, int, int], transform=None) -> None:
        """Draw a mesh, optionally transformed, in *color* (BGR)."""

    @abstractmethod
    def draw_object(self, obj: SceneObject) -> None:
        """Draw a scene object in world space (applies its transform)."""

    @abstractmethod
    def project(self, world_point: Sequence[float]) -> tuple[float, float] | None:
        """Project a world-space point to screen pixel coordinates.

        Returns ``None`` if the point is behind the camera.
        """

    @abstractmethod
    def finish(self) -> np.ndarray:
        """Finalize the frame and return the composited BGR image."""

    def close(self) -> None:  # noqa: B027
        """Release renderer resources (overridden by backends that need it)."""
