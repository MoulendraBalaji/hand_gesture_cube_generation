"""Headless tests for the OpenCV (software) 3D renderer.

These need no display or GPU: everything is projected and drawn into a plain
numpy/OpenCV canvas, so they run fine in CI.
"""

from __future__ import annotations

import numpy as np

from src.render.cv_renderer import OpenCVRenderer
from src.scene.scene_manager import SceneManager


def _canvas(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestOpenCVRenderer:
    def test_projects_origin_to_center(self):
        r = OpenCVRenderer(camera_z=6.0)
        r.begin_frame(_canvas(640, 480))
        px = r.project([0.0, 0.0, 0.0])
        assert px is not None
        # Camera at +Z looking toward the origin; origin projects to frame center.
        assert abs(px[0] - 320.0) < 2.0
        assert abs(px[1] - 240.0) < 2.0

    def test_draw_cube_produces_non_blank_frame(self):
        r = OpenCVRenderer(camera_z=6.0)
        frame = _canvas()
        r.begin_frame(frame)
        s = SceneManager()
        obj = s.spawn("cube", np.zeros(3), scale=2.0, color=(200, 110, 40))
        r.draw_object(obj)
        out = r.finish()
        assert out.shape == (480, 640, 3)
        assert bool(np.any(out > 0))

    def test_background_survives_roundtrip(self):
        r = OpenCVRenderer(camera_z=6.0)
        frame = _canvas()
        frame[:] = (10, 20, 30)
        r.begin_frame(frame)
        s = SceneManager()
        r.draw_object(s.spawn("sphere", np.zeros(3), scale=1.5))
        out = r.finish()
        # The background corner colour should be preserved where nothing is drawn.
        assert tuple(out[5, 5]) == (10, 20, 30)

    def test_project_null_behind_camera(self):
        r = OpenCVRenderer(camera_z=6.0)
        r.begin_frame(_canvas())
        # A point behind the camera (large +Z, beyond the eye) is not projectable.
        assert r.project([0.0, 0.0, 40.0]) is None
