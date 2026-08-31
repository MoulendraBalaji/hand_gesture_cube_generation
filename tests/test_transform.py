"""Unit tests for the 3D transform / linear-algebra module."""

from __future__ import annotations

import numpy as np

from src.scene import transform as T


class TestBasicMatrices:
    def test_identity_is_identity(self):
        result = T.apply_matrix(np.eye(4), np.array([[0, 0, 0]]))
        np.testing.assert_allclose(result[0], [0, 0, 0], atol=1e-8)

    def test_translation(self):
        m = T.translation(1, 2, 3)
        out = T.apply_matrix(m, np.array([[0, 0, 0]]))
        np.testing.assert_allclose(out[0], [1, 2, 3])

    def test_uniform_scaling(self):
        m = T.uniform_scaling(2.0)
        out = T.apply_matrix(m, np.array([[1, 0, 0]]))
        np.testing.assert_allclose(out[0], [2, 0, 0])

    def test_nonuniform_scaling(self):
        m = T.scaling(1, 2, 3)
        out = T.apply_matrix(m, np.array([[1, 1, 1]]))
        np.testing.assert_allclose(out[0], [1, 2, 3])


class TestRotation:
    def test_rotation_y_90z(self):
        m = T.rotation_y(90)
        out = T.apply_matrix(m, np.array([[1, 0, 0]]))
        np.testing.assert_allclose(out[0], [0, 0, -1], atol=1e-8)

    def test_rotation_x_90(self):
        m = T.rotation_x(90)
        out = T.apply_matrix(m, np.array([[0, 1, 0]]))
        np.testing.assert_allclose(out[0], [0, 0, 1], atol=1e-8)

    def test_rotation_z_90(self):
        m = T.rotation_z(90)
        out = T.apply_matrix(m, np.array([[1, 0, 0]]))
        np.testing.assert_allclose(out[0], [0, 1, 0], atol=1e-8)

    def test_rotation_preserves_length(self):
        m = T.rotation_y(37)
        v = np.array([[3, -1, 2]])
        out = T.apply_matrix(m, v)
        assert np.isclose(np.linalg.norm(out[0]), np.linalg.norm(v[0]))


class TestQuaternion:
    def test_spin_preserves_length(self):
        q = T.quat_from_axis_angle(np.array([1, 0, 0]), 45)
        v = np.array([0.5, 1.0, -2.0])
        out = T.rotate_quat(q, v)
        assert np.isclose(np.linalg.norm(out), np.linalg.norm(v))

    def test_quat_angle_matches_matrix(self):
        axis = np.array([0, 1, 0])
        q = T.quat_from_axis_angle(axis, 90)
        v = np.array([1.0, 0.0, 0.0])
        qv = T.rotate_quat(q, v)
        mv = T.apply_matrix(T.rotation_y(90), np.array([v]))
        np.testing.assert_allclose(qv, mv[0], atol=1e-6)

    def test_zero_axis_returns_identity(self):
        q = T.quat_from_axis_angle(np.array([0, 0, 0]), 90)
        np.testing.assert_allclose(q, [1, 0, 0, 0])

    def test_quat_multiply_identity(self):
        q = T.quat_from_axis_angle(np.array([0, 0, 1]), 30)
        i = np.array([1.0, 0, 0, 0])
        np.testing.assert_allclose(T.quat_multiply(i, q), q)


class TestProjection:
    def test_perspective_ndc_sane(self):
        # A point directly in center of view should land near NDC (0,0).
        m = T.perspective(60, 1.0, 0.1, 100)
        view = np.array([[0.0, 0.0, -6.0]])  # looking down -z
        clip = (m @ np.c_[view, np.ones(1)].T).T
        w = clip[:, 3:4]
        ndc = clip[:, :3] / w
        np.testing.assert_allclose(ndc[0, :2], [0, 0], atol=1e-6)

    def test_look_at_places_eye(self):
        v = T.look_at(np.array([0, 0, 6]), np.array([0, 0, 0]), np.array([0, 1, 0]))
        # Camera looks down -z; the origin (in front) lands at view z = -6.
        out = T.apply_matrix(v, np.array([[0, 0, 0]]))
        np.testing.assert_allclose(out[0], [0, 0, -6], atol=1e-6)

    def test_compose_order_translate_then_rotate(self):
        # compose(m1, m2) = m2 @ m1, so m1 (the first argument) is applied
        # first.  With a rotation first then a translation, the origin moves
        # to the translation offset (rotation of the origin is a no-op).
        m = T.compose(T.rotation_y(90), T.translation(1, 0, 0))
        out = T.apply_matrix(m, np.array([[0, 0, 0]]))
        np.testing.assert_allclose(out[0], [1, 0, 0], atol=1e-8)

    def test_applies_arbitrary_pointset(self):
        m = T.translation(5, -2, 1)
        pts = np.array([[0.0, 0, 0], [1, 1, 1], [-3, 4, 2]])
        out = T.apply_matrix(m, pts)
        np.testing.assert_allclose(out, pts + np.array([5, -2, 1]))
