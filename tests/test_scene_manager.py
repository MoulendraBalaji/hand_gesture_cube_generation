"""Unit tests for the scene manager (spawn, select, transform, undo/redo)."""

from __future__ import annotations

import numpy as np
import pytest

from src.scene.scene_manager import SceneManager


def make_scene() -> SceneManager:
    return SceneManager()


class TestSpawn:
    def test_spawn_adds_object(self):
        s = make_scene()
        o = s.spawn("cube", np.array([0, 1, 0]))
        assert len(s.objects) == 1
        assert o.kind == "cube"

    def test_spawn_selects_new_object(self):
        s = make_scene()
        o = s.spawn("cube", np.zeros(3))
        assert s.selected_id == o.obj_id

    def test_spawn_supports_all_primitives(self):
        s = make_scene()
        for kind in ("cube", "pyramid", "sphere"):
            o = s.spawn(kind, np.zeros(3))
            assert o.kind == kind

    def test_ids_monotonically_increment(self):
        s = make_scene()
        a = s.spawn("cube", np.zeros(3))
        b = s.spawn("cube", np.zeros(3))
        assert b.obj_id > a.obj_id


class TestTransform:
    def test_move_changes_position(self):
        s = make_scene()
        o = s.spawn("cube", np.array([0, 0, 0]))
        s.move(o.obj_id, np.array([1, 2, 3]))
        np.testing.assert_allclose(o.position, [1, 2, 3])

    def test_rotate_accumulates(self):
        s = make_scene()
        o = s.spawn("cube", np.zeros(3))
        s.rotate(o.obj_id, "y", 30)
        s.rotate(o.obj_id, "y", 15)
        assert o.rotation_deg[1] == pytest.approx(45)

    def test_scale_bounded(self):
        s = make_scene()
        o = s.spawn("cube", np.zeros(3))
        s.scale_by(o.obj_id, 100)
        assert o.scale <= 8.0
        s.scale_by(o.obj_id, 0.0001)
        assert o.scale >= 0.1


class TestSelectionAndDelete:
    def test_select_toggles_flag(self):
        s = make_scene()
        a = s.spawn("cube", np.zeros(3))
        b = s.spawn("cube", np.zeros(3))
        s.select(a.obj_id)
        assert a.selected is True and b.selected is False

    def test_delete_removes_object(self):
        s = make_scene()
        o = s.spawn("cube", np.zeros(3))
        s.delete(o.obj_id)
        assert len(s.objects) == 0

    def test_delete_selected(self):
        s = make_scene()
        s.spawn("cube", np.zeros(3))
        s.delete_selected()
        assert len(s.objects) == 0


class TestUndoRedo:
    def test_undo_spawn(self):
        s = make_scene()
        s.spawn("cube", np.zeros(3))
        s.undo()
        assert len(s.objects) == 0

    def test_undo_then_redo(self):
        s = make_scene()
        o = s.spawn("cube", np.zeros(3))
        s.move(o.obj_id, np.array([5, 0, 0]))
        s.undo()
        np.testing.assert_allclose(s.objects[0].position, [0, 0, 0])
        s.redo()
        np.testing.assert_allclose(s.objects[0].position, [5, 0, 0])

    def test_undo_stack_empty_returns_false(self):
        s = make_scene()
        assert s.undo() is False

    def test_clear_is_undoable(self):
        s = make_scene()
        s.spawn("cube", np.zeros(3))
        s.spawn("cube", np.zeros(3))
        s.clear()
        assert len(s.objects) == 0
        s.undo()
        assert len(s.objects) == 2
