"""Unit tests for the physics engine (gravity, floor, collisions, velocity)."""

from __future__ import annotations

import numpy as np
import pytest

from src.scene.physics import BodyCacher, PhysicsEngine, RigidBody


def make_engine(**kw) -> PhysicsEngine:
    defaults = dict(gravity=9.8, floor_y=-3.0, restitution=0.55, damping=0.85,
                    air_resistance=0.995, sphere_collision=True)
    defaults.update(kw)
    return PhysicsEngine(**defaults)


class TestGravity:
    def test_falling_object_accelerates_down(self):
        e = make_engine()
        b = RigidBody()
        b.dynamic = True
        b.position = np.array([0.0, 1.0, 0.0])
        v0 = b.velocity[1]
        e.step(b, 0.016)
        assert b.velocity[1] < v0  # sped up downward
        assert b.position[1] < 1.0

    def test_static_object_never_moves(self):
        e = make_engine()
        b = RigidBody()  # dynamic=False by default
        b.position = np.array([0.0, 1.0, 0.0])
        e.step(b, 0.5)
        np.testing.assert_allclose(b.position, [0.0, 1.0, 0.0])

    def test_gravity_magnitude_approximate(self):
        e = make_engine(air_resistance=1.0)
        b = RigidBody(dynamic=True)
        b.position = np.array([0.0, 0.0, 0.0])
        # Small dt so the object stays above the floor (no bounce).
        e.step(b, 0.1)
        assert b.velocity[1] == pytest.approx(-0.98, abs=1e-6)


class TestFloor:
    def test_bounces_above_floor(self):
        e = make_engine()
        b = RigidBody(dynamic=True)
        b.position = np.array([0.0, -3.5, 0.0])
        b.velocity = np.array([0.0, -5.0, 0.0])
        e.step(b, 0.016)
        assert b.position[1] >= e.floor_y
        assert b.velocity[1] >= 0  # bounced back up

    def test_never_sinks_below_floor(self):
        e = make_engine()
        b = RigidBody(dynamic=True)
        b.position = np.array([0.0, -100.0, 0.0])
        b.velocity = np.array([0.0, -1.0, 0.0])
        e.step(b, 0.016)
        assert b.position[1] >= e.floor_y - 1e-6

    def test_eventually_rests(self):
        e = make_engine()
        b = RigidBody(dynamic=True)
        b.position = np.array([0.0, 0.0, 0.0])
        for _ in range(500):
            e.step(b, 0.016)
        assert abs(b.velocity[1]) < 0.05


class TestVelocityEstimate:
    def test_estimate_velocity_constant_motion(self):
        eps = []
        for i in range(5):
            eps.append(np.array([i * 0.1, 0.0, 0.0]))
        v = PhysicsEngine.estimate_velocity(eps, 0.1)
        np.testing.assert_allclose(v, [1.0, 0.0, 0.0], atol=1e-6)

    def test_estimate_needs_two_samples(self):
        assert np.sum(PhysicsEngine.estimate_velocity([np.zeros(3)], 0.1)) == 0


class TestSpheres:
    def test_separates_overlapping(self):
        e = make_engine()
        a = RigidBody(dynamic=True, radius=1.0)
        b = RigidBody(dynamic=True, radius=1.0)
        a.position = np.zeros(3)
        b.position = np.array([1.0, 0.0, 0.0])
        e.collide_spheres([a, b])
        sep = np.linalg.norm(a.position - b.position)
        assert sep >= 2.0 - 1e-6

    def test_nonoverlapping_untouched(self):
        e = make_engine()
        a = RigidBody(dynamic=True, radius=1.0, position=np.zeros(3))
        b = RigidBody(dynamic=True, radius=1.0, position=np.array([10.0, 0, 0]))
        pa = a.position.copy()
        pb = b.position.copy()
        e.collide_spheres([a, b])
        np.testing.assert_allclose(a.position, pa)
        np.testing.assert_allclose(b.position, pb)

    def test_static_body_still_separates(self):
        e = make_engine()
        a = RigidBody(dynamic=True, radius=1.0, position=np.zeros(3))
        b = RigidBody(dynamic=False, radius=1.0, position=np.array([1.0, 0, 0]))
        e.collide_spheres([a, b])
        assert np.linalg.norm(a.position - b.position) >= 2.0 - 1e-6


class TestBodyCacher:
    def test_velocity_follows_history(self):
        c = BodyCacher(window=8)
        for i in range(10):
            c.push(1, np.array([i * 0.05, 0.0, 0.0]))
        v = c.velocity(1, 0.05)
        np.testing.assert_allclose(v[0], 1.0, atol=0.2)

    def test_clear_removes_history(self):
        c = BodyCacher()
        c.push(1, np.zeros(3))
        c.clear(1)
        assert np.sum(c.velocity(1, 0.05)) == 0
