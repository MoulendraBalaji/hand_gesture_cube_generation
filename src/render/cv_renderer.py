"""Pure-OpenCV 3D perspective renderer — the headline feature.

Most webcam gesture demos "fake" a 3D cube with two offset rectangles (as the
original prototype did).  This renderer instead does *real* camera math:

    model (rotation/scale/translation) -> view -> perspective -> screen

and draws a genuinely projected wireframe mesh with:

  * perspective-correct vertex projection,
  * painter's-algorithm depth sorting of faces (back-to-front fill),
  * simple Lambertian shading from a fixed light direction for solid faces.

No OpenGL, no GPU — everything is numpy + OpenCV primitives, satisfying
:class:`BaseRenderer` so the OpenGL backend can drop in later.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..scene import transform as T
from ..scene.primitives import Mesh, edge_connectivity
from ..scene.scene_manager import SceneObject
from .base_renderer import BaseRenderer


class OpenCVRenderer(BaseRenderer):
    """Perspective wireframe/shaded renderer drawing into a BGR frame."""

    def __init__(
        self,
        *,
        fov_deg: float = 60.0,
        near: float = 0.1,
        far: float = 100.0,
        camera_z: float = 6.0,
        mesh_color: tuple[int, int, int] = (200, 110, 40),
        edge_color: tuple[int, int, int] = (255, 235, 200),
        light_dir: Sequence[float] = (0.4, 0.8, 0.6),
    ) -> None:
        self.fov_deg = fov_deg
        self.near = near
        self.far = far
        self.camera_z = camera_z
        self.mesh_color = tuple(mesh_color)
        self.edge_color = tuple(edge_color)
        self._light = np.asarray(light_dir, dtype=float)
        self._light = self._light / (np.linalg.norm(self._light) + 1e-9)

        self._frame: np.ndarray | None = None
        self._width = 0
        self._height = 0
        self._aspect = 1.0
        self._proj: np.ndarray | None = None
        self._view: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _size(self) -> tuple[int, int]:
        if self._frame is None:
            return (0, 0)
        h, w = self._frame.shape[:2]
        return (w, h)

    def begin_frame(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._width, self._height = self._size()
        self._aspect = self._width / max(1, self._height)
        self._proj = T.perspective(self.fov_deg, self._aspect, self.near, self.far)
        self._view = T.look_at(
            np.array([0.0, 0.0, self.camera_z]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )

    # ------------------------------------------------------------------
    def _to_screen(self, world_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to screen; return (pixels, view_z) arrays."""
        if self._proj is None or self._view is None:
            return np.zeros((0, 2)), np.zeros(0)
        view = T.apply_matrix(self._view, world_points)
        clip = (self._proj @ np.c_[view, np.ones(len(view))].T).T
        w = clip[:, 3:4]
        w[w == 0] = 1e-12
        ndc = clip[:, :3] / w
        px = (ndc[:, 0] + 1.0) * 0.5 * self._width
        py = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * self._height
        return np.column_stack([px, py]), view[:, 2]

    # ------------------------------------------------------------------
    def project(self, world_point: Sequence[float]) -> tuple[float, float] | None:
        """Project a single world point to a pixel tuple (or None if behind)."""
        pts = np.asarray([world_point], dtype=float)
        screen, z = self._to_screen(pts)
        if len(screen) == 0 or z[0] >= self.near - 1e-6:
            return None
        return (float(screen[0, 0]), float(screen[0, 1]))

    # ------------------------------------------------------------------
    def draw_mesh(self, mesh: Mesh, color: tuple[int, int, int], transform=None) -> None:
        """Draw a mesh, optionally transformed, in *color*.

        Args:
            mesh: The mesh (world space if ``transform`` is None).
            color: BGR fill color for solid faces.
            transform: Optional 4x4 model matrix applied to vertices.
        """
        if self._frame is None or self._proj is None or self._view is None:
            return
        verts = mesh.vertices
        if transform is not None:
            verts = T.apply_matrix(transform, verts)
        self._draw_vertices(verts, mesh.faces, color)

    def draw_object(self, obj: SceneObject) -> None:
        """Draw a scene object using its own transform and per-object color."""
        self.draw_mesh(obj.mesh, obj.color)

    # ------------------------------------------------------------------
    def _draw_vertices(
        self, world_verts: np.ndarray, faces: list[list[int]], color
    ) -> None:
        pixels, view_z = self._to_screen(world_verts)

        # Draw filled, depth-sorted, Lambert-shaded faces (painter's algorithm).
        face_records: list[tuple[float, list[np.ndarray], np.ndarray]] = []
        for face in faces:
            idx = np.asarray(face, dtype=int)
            if np.any(idx >= len(pixels)) or len(idx) < 3:
                continue
            face_px = pixels[idx]
            face_z = view_z[idx]
            avg_z = float(np.mean(face_z))
            normal = self._face_normal(world_verts[idx])
            face_records.append((avg_z, face_px, normal))

        # Painter's algorithm: draw farthest faces first (most negative view
        # Z, since the camera looks down -Z), nearest faces last so they
        # occlude correctly.
        face_records.sort(key=lambda r: r[0])  # ascending Z = far -> near
        vertex_colors = {}

        def vertex_light(i: int) -> float:
            if i not in vertex_colors:
                n = world_verts[i]
                ln = np.linalg.norm(n)
                if ln < 1e-9:
                    vertex_colors[i] = 0.7
                else:
                    vertex_colors[i] = max(0.15, float((n / ln) @ self._light))
            return vertex_colors[i]

        for avg_z, face_px, normal in face_records:
            pts = np.round(face_px).astype(int)
            # Lambertian term from a fixed light direction.
            lambert = max(0.12, float(normal @ self._light))
            # Atmospheric depth cue: faces farther from the camera are dimmer.
            dist = min(self.camera_z, max(0.0, -avg_z))
            depth_fade = 1.0 - 0.55 * (dist / self.camera_z)
            shade = lambert * depth_fade
            face_color = tuple(int(max(0, min(255, c * shade))) for c in color)
            cv_fill_polygon(self._frame, pts, face_color)
            # Edges slightly brighter.
            cv_polyline(self._frame, pts, self.edge_color, closed=True)
            _ = vertex_light  # keep reference for potential future outline shading

    def _face_normal(self, verts_face: np.ndarray) -> np.ndarray:
        """Average of triangle cross products over a (possibly concave) face."""
        # Signed area-weighted normal for a polygon.
        p0 = verts_face[0]
        normal = np.zeros(3)
        for i in range(1, len(verts_face) - 1):
            normal += np.cross(verts_face[i] - p0, verts_face[i + 1] - p0)
        n = np.linalg.norm(normal)
        if n < 1e-9:
            return np.array([0.0, 1.0, 0.0])
        return normal / n

    # ------------------------------------------------------------------
    def draw_edges(self, world_verts: np.ndarray, faces: list[list[int]]) -> None:
        """Draw only the wireframe edges of a mesh (for line-only scene)."""
        if self._frame is None:
            return
        pixels, view_z = self._to_screen(world_verts)
        for e0, e1 in edge_connectivity(Mesh(world_verts, faces)):
            if e0 < len(pixels) and e1 < len(pixels):
                p0 = tuple(np.round(pixels[e0]).astype(int))
                p1 = tuple(np.round(pixels[e1]).astype(int))
                cv_line(self._frame, p0, p1, self.edge_color, 1)

    def finish(self) -> np.ndarray:
        """Return the composited BGR frame."""
        frame = self._frame
        self._frame = None
        return frame if frame is not None else np.zeros((0, 0, 3), np.uint8)


# ---------------------------------------------------------------------------
# Thin cv2 indirection so the module can be imported where cv2 is optional.
# ---------------------------------------------------------------------------
def _cv():
    import cv2

    return cv2


def cv_fill_polygon(frame: np.ndarray, pts: np.ndarray, color) -> None:
    _cv().fillPoly(frame, [pts.reshape(-1, 1, 2)], color, lineType=_cv().LINE_AA)


def cv_polyline(frame: np.ndarray, pts: np.ndarray, color, closed: bool) -> None:
    _cv().polylines(
        frame, [pts.reshape(-1, 1, 2)], closed, color, 1, lineType=_cv().LINE_AA
    )


def cv_line(frame: np.ndarray, p0, p1, color, thickness: int) -> None:
    _cv().line(frame, p0, p1, color, thickness, lineType=_cv().LINE_AA)
