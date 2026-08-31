"""Optional PyOpenGL-accelerated renderer backend.

Provides the same :class:`BaseRenderer` interface as the OpenCV renderer but
renders via modern-ish OpenGL (immediate mode for simplicity and portability).
This module imports heavy optional deps lazily so the app runs fine without
them; any import failure simply disables the backend.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..scene import transform as T
from ..scene.primitives import Mesh
from ..scene.scene_manager import SceneObject
from .base_renderer import BaseRenderer

_GL_AVAILABLE = False
_gl, _glu, _glfw = None, None, None


def _import_gl() -> bool:
    """Best-effort import of PyOpenGL + GLFW helpers."""
    global _GL_AVAILABLE, _gl, _glu, _glfw
    if _GL_AVAILABLE:
        return True
    try:
        import glfw as _glfw  # type: ignore
        from OpenGL import GL as _gl  # type: ignore
        from OpenGL import GLU as _glu  # type: ignore

        _GL_AVAILABLE = True
        return True
    except Exception:
        return False


class GLRenderer(BaseRenderer):
    """OpenGL backend. Creation silently degrades to ``None`` if unavailable."""

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        title: str = "GestureForge GL",
        fov_deg: float = 60.0,
        near: float = 0.1,
        far: float = 100.0,
        camera_z: float = 6.0,
        clear_color: Sequence[float] = (0.12, 0.12, 0.14, 1.0),
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.fov_deg = fov_deg
        self.near = near
        self.far = far
        self.camera_z = camera_z
        self.clear_color = tuple(clear_color)
        self.window = None
        self._initialised = False
        self.flip = _import_gl()

        if not _import_gl():
            raise RuntimeError(
                "OpenGL backend unavailable: install with "
                "`pip install gestureforge[opengl]` (PyOpenGL + glfw)."
            )
        self._setup_window()

    def _setup_window(self) -> None:
        _glfw.init()
        _glfw.window_hint(_glfw.VISIBLE, _glfw.TRUE)
        self.window = _glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            raise RuntimeError("glfw failed to create a window")
        _glfw.make_context_current(self.window)
        _gl.glEnable(_gl.GL_DEPTH_TEST)
        _gl.glEnable(_gl.GL_CULL_FACE)
        _gl.glClearColor(*self.clear_color)
        self._initialised = True

    # ------------------------------------------------------------------
    def _proj_setup(self) -> None:
        _gl.glMatrixMode(_gl.GL_PROJECTION)
        _gl.glLoadIdentity()
        _glu.gluPerspective(self.fov_deg, self.width / self.height, self.near, self.far)
        _gl.glMatrixMode(_gl.GL_MODELVIEW)
        _gl.glLoadIdentity()
        _glu.gluLookAt(0, 0, self.camera_z, 0, 0, 0, 0, 1, 0)

    def begin_frame(self, frame: np.ndarray) -> None:
        if not self._initialised:
            return
        _glfw.poll_events()
        _gl.glClear(_gl.GL_COLOR_BUFFER_BIT | _gl.GL_DEPTH_BUFFER_BIT)
        self._proj_setup()

    def draw_mesh(self, mesh: Mesh, color: tuple[int, int, int], transform=None) -> None:
        if not self._initialised:
            return
        _gl.glMatrixMode(_gl.GL_MODELVIEW)
        _gl.glPushMatrix()
        if transform is not None:
            # Column-major 4x4 for OpenGL.
            flat = np.asarray(transform, dtype=np.float32).T
            _gl.glMultMatrixf(flat.flatten())
        r, g, b = (c / 255.0 for c in color)
        _gl.glColor3f(r, g, b)
        # Draw filled triangles with (1,0,0)->(0,1,0)->(0,0,1) light.
        for face in mesh.faces:
            _gl.glBegin(_gl.GL_POLYGON)
            for i in face:
                v = mesh.vertices[i]
                _gl.glVertex3f(v[0], v[1], v[2])
            _gl.glEnd()
        _gl.glPopMatrix()

    def draw_object(self, obj: SceneObject) -> None:
        m = T.compose(
            T.translation(*obj.position.tolist()),
            T.rotation_x(obj.rotation_deg[0]),
            T.rotation_y(obj.rotation_deg[1]),
            T.rotation_z(obj.rotation_deg[2]),
            T.uniform_scaling(obj.scale),
        )
        self.draw_mesh(obj.mesh, obj.color, m)

    def project(self, world_point: Sequence[float]) -> tuple[float, float] | None:
        # Projection handled analytically to avoid GL readback complexity.
        _gl.glGetDoublev(_gl.GL_MODELVIEW_MATRIX, mv := np.eye(4, dtype=np.float64))
        _gl.glGetDoublev(_gl.GL_PROJECTION_MATRIX, pj := np.eye(4, dtype=np.float64))
        viewport = np.array([0, 0, self.width, self.height], dtype=np.int32)
        win = _glu.gluProject(
            float(world_point[0]), float(world_point[1]), float(world_point[2]),
            mv, pj, viewport,
        )
        try:
            return (float(win[0]), float(win[1]))
        except Exception:
            return None

    def finish(self) -> np.ndarray:
        if self._initialised:
            _glfw.swap_buffers(self.window)
        return np.zeros((0, 0, 3), np.uint8)

    def close(self) -> None:
        if self.window:
            _glfw.destroy_window(self.window)
            _glfw.terminate()
            self.window = None
        self._initialised = False
