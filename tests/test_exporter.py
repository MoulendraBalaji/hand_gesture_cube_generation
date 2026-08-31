"""Unit tests for OBJ scene export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.scene.exporter import export_scene, mesh_to_obj
from src.scene.primitives import cube_mesh
from src.scene.scene_manager import SceneManager


class TestObjExport:
    def test_writes_valid_obj(self, tmp_path: Path):
        out = tmp_path / "scene.obj"
        mesh = cube_mesh()
        mesh_to_obj([("test", mesh, (200, 110, 40))], out)
        text = out.read_text(encoding="utf-8")
        assert text.startswith("#")
        assert "mtllib scene.mtl" in text
        assert "o test" in text
        assert "v " in text
        assert any(line.startswith("f ") for line in text.splitlines())

    def test_exports_multiple_objects(self, tmp_path: Path):
        s = SceneManager()
        s.spawn("cube", np.array([0, 0, 0]))
        s.spawn("pyramid", np.array([2, 0, 0]))
        out = tmp_path / "out.obj"
        export_scene(s.objects, out)
        text = out.read_text(encoding="utf-8")
        assert "o obj_" in text
        assert text.count("o ") == 2

    def test_mtl_emitted(self, tmp_path: Path):
        out = tmp_path / "scene.obj"
        mesh_to_obj([("cube", cube_mesh(), (200, 110, 40))], out)
        mtl = out.with_suffix(".mtl")
        assert mtl.is_file()
        assert "newmtl mat_cube" in mtl.read_text(encoding="utf-8")

    def test_world_transform_applied(self, tmp_path: Path):
        s = SceneManager()
        o = s.spawn("cube", np.array([10, 0, 0]))
        out = tmp_path / "moved.obj"
        export_scene([o], out)
        text = out.read_text(encoding="utf-8")
        # Transformed cube vertices are far from origin (offset +10 in x).
        xs = [float(line.split()[1]) for line in text.splitlines() if line.startswith("v ")]
        assert min(xs) > 7.0
